//! CPU sampler for MLX — temperature / top-p / top-k / seed.
//!
//! For greedy (temperature=0), argmax is done in-graph on GPU — zero logits readback.
//! For temperature > 0, logits are read back via a single `to_vec_*` call
//! (bf16 preferred for smallest transfer, 300KB vs 600KB f32 for 150k vocab).

use anyhow::{Context, Result};

/// Sampling params. Temperature 0 => greedy (argmax).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0, // greedy
            top_p: 1.0,
            top_k: 0, // 0 = disabled
            seed: 42,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self::default()
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p.clamp(0.0, 1.0);
        self
    }
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Simple xorshift64* PRNG — no external crate.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_f32(&mut self) -> f32 {
        // [0, 1)
        const DIV: f64 = (1u64 << 53) as f64;
        let v = (self.next_u64() >> 11) as f64 / DIV;
        v as f32
    }
}

/// Sample from MLX logits array via single readback + CPU sampling.
///
/// Reads back logits ONCE (bf16 first for smallest transfer, then f16/f32
/// fallback). The previous implementation read all three dtypes + ran GPU
/// argmax (4x readback + 1 sync); now it's 1 readback + 1 sync.
pub fn sample_from_logits(
    ctx: &engine_mlx_ffi::MlxCtx,
    logits: engine_mlx_ffi::mlx_array,
    params: &SamplingParams,
    rng: &mut SimpleRng,
) -> Result<u32> {
    // Single readback — try bf16 (smallest), then f16, then f32.
    let vals = ctx
        .to_vec_bf16_as_f32(logits)
        .or_else(|_| ctx.to_vec_f16_as_f32(logits))
        .or_else(|_| ctx.to_vec_f32(logits))
        .context("read logits for sampling")?;
    sample_token(&vals, params, rng)
}

/// Sample a token id from logits slice.
/// Returns the sampled token id (index into vocab).
pub fn sample_token(logits: &[f32], params: &SamplingParams, rng: &mut SimpleRng) -> Result<u32> {
    if logits.is_empty() {
        anyhow::bail!("sample_token: empty logits");
    }
    // Greedy
    if params.temperature < 1e-6 {
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .context("empty logits")?;
        return Ok(max_idx as u32);
    }

    // Temperature scaling
    let temp = params.temperature.max(1e-6);
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Top-k filtering: keep only k largest
    let mut indices: Vec<usize> = (0..scaled.len()).collect();
    let mut filtered = scaled.clone();
    if params.top_k > 0 && params.top_k < scaled.len() {
        // Find k-th largest threshold (partial sort)
        indices.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
        let kth = scaled[indices[params.top_k - 1]];
        for v in &mut filtered {
            if *v < kth {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max = filtered
        .iter()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = filtered.iter().map(|&x| if x.is_finite() { (x - max).exp() } else { 0.0 }).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        // fallback to argmax
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .context("empty logits")?;
        return Ok(max_idx as u32);
    }
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    // Top-p (nucleus) filtering
    let mut p_sorted: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    p_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0;
    let mut keep = vec![false; probs.len()];
    for (idx, p) in &p_sorted {
        cum += *p;
        keep[*idx] = true;
        if cum >= params.top_p {
            break;
        }
    }
    // Ensure at least one kept
    if !keep.iter().any(|&k| k) {
        if let Some((idx, _)) = p_sorted.first() {
            keep[*idx] = true;
        }
    }
    // Zero out non-kept and renormalize
    let mut filtered_probs = vec![0.0; probs.len()];
    let mut filtered_sum = 0.0;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            filtered_probs[i] = probs[i];
            filtered_sum += probs[i];
        }
    }
    if filtered_sum > 0.0 {
        for v in &mut filtered_probs {
            *v /= filtered_sum;
        }
    } else {
        filtered_probs = probs;
    }

    // Categorical sample
    let r = rng.next_f32();
    let mut cumsum = 0.0;
    for (i, &p) in filtered_probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(i as u32);
        }
    }
    // Fallback to last kept
    for (i, &k) in keep.iter().enumerate().rev() {
        if k {
            return Ok(i as u32);
        }
    }
    Ok(0)
}
