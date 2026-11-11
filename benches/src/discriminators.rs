//! Pure, testable discriminators for the 5 standard bench tests.
//!
//! All three are preamble-tolerant: they search for the expected answer
//! *anywhere* in the output, so a thinking model that emits reasoning before
//! the final answer still matches. They are still meaningful: empty /
//! whitespace-only / wrong-answer output fails.

use regex::Regex;

/// Factual test discriminator: response contains "Paris" (case-insensitive,
/// word-bounded, searched anywhere).
pub fn factual_match(content: &str) -> bool {
    Regex::new(r"(?i)\bparis\b")
        .map(|re| re.is_match(content))
        .unwrap_or(false)
}

/// Instruction test discriminator: response contains "DONE" (case-insensitive,
/// word-bounded, searched anywhere). No length constraint — a thinking model
/// may prepend reasoning before the answer.
pub fn instruction_match(content: &str) -> bool {
    Regex::new(r"(?i)\bdone\b")
        .map(|re| re.is_match(content))
        .unwrap_or(false)
}

/// Coherence test discriminator: answer 18 present with coherent working.
///
/// `ANSWER: 18` is the primary signal. The loose arm is deliberately
/// restricted: `18` must appear in an answer context (preceded by a
/// response keyword like `equals`/`is`/`answer`, or followed by `left`/
/// `remaining`) *and* there must be working (72 or 12). This rejects false
/// positives like "…leaves 12… no wait, 18 is prime…" where 18 is a side
/// remark, not the final answer — while staying preamble-tolerant.
pub fn coherence_match(content: &str) -> bool {
    let answer_line = Regex::new(r"(?i)answer\s*[:=]\s*\$?\s*0*18\b")
        .map(|re| re.is_match(content))
        .unwrap_or(false);
    // Loose arm: 18 in an answer context (not a side remark like "18 is prime").
    let loose = Regex::new(
        r"(?i)(?:(?:answer|is|equals|=\s*|result|leaves|remaining)\s*\D*18\b)|(?:\b18\s+(?:left|remaining)\b)",
    )
    .map(|re| re.is_match(content))
    .unwrap_or(false);
    let shows_work = Regex::new(r"\b(72|12)\b")
        .map(|re| re.is_match(content))
        .unwrap_or(false);
    answer_line || (loose && shows_work)
}
