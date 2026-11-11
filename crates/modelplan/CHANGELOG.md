# Changelog — nxm-modelplan

All notable changes to this project will be documented in this format.

## [0.1.0] - 2026-07-31

### Added
- feat: crate scaffold — `nxm-modelplan` lib + bin, workspace member
- feat: manifest types — `ModelManifest`, `ModelInfo`, `AttentionConfig`, `EngineHints`, `OpsRequired`, etc. 
- feat: parser — `parser::parse_model_dir` reads config.json + tokenizer_config + safetensors headers
- feat: ops detector — `ops_detector::detect_ops` derives ops_required, engine_hints, layers config
- feat: plan — `plan::from_manifest` builds an `EnginePlan` (layer plans, scratch sizes, kernels)
- feat: CLI placeholder (subcommands `inspect`, `save`, `check`, `plan`)
