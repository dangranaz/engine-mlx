# RULES

Coding rules for this project. Every contributor (human or AI) must follow these.

## Public repository policy (mandatory)

This is a public repository. It MUST NOT contain:

- **Any project documents** — plans, designs, ADRs, research notes, roadmaps.
  Only `README.md`, `LICENSE`, `STORY.md` (see below), and the technical repo
  files below are allowed.
- **Any proprietary/experimental technology** — keep advanced or unpublished
  techniques out of this repo entirely.
- **Any internal references** — no private hostnames, internal repo URLs,
  internal tooling names, access tokens, or private paths.

Allowed technical repo files: `README.md`, `LICENSE`, `CONTEXT.md`,
`STATUS.md`, `CHANGELOG.md`, `RULES.md`.

**The one narrative exception:** `STORY.md` is the single project document
allowed in public — the story of how and why the engine was built, in English
only. Nothing else document-like goes here.

## File structure

- **Max 300 lines per file**. If a file grows beyond, split by functionality.
- Unit tests live in the `tests/` directory at crate root, one file per module.
- Integration tests live in `tests/integration/` when applicable.
- Tests are never mixed with production code (no `#[cfg(test)]` inline modules).

## Logging

- Public functions should have structured logging (`tracing`).
- Use per-functionality targets, e.g. `tracing::info!(target: "prefill", ...)`.
- Entry/exit logging for complex operations, with timing.

## Error handling

- No `unwrap()` / `expect()` in production code (tests only).
- `anyhow::Result` for application-level errors; `thiserror` for library errors.
- Always add context: `.context("what was happening when this failed")`.

## Naming & style

- English for code, comments, and documentation.
- Rust naming: `snake_case`, `PascalCase`, `SCREAMING_SNAKE_CASE`.
- Modules named after functionality, not implementation detail.

## Dependencies

- Pin exact versions in Cargo.toml for external crates.
- Prefer workspace dependencies declared in the root `Cargo.toml`.

## Testing

- New features come with unit tests.
- Integration tests for cross-module or cross-crate interactions.
- Use `#[ignore]` for tests requiring hardware (GPU), with a comment.

## Documentation

- Update `CHANGELOG.md` with every meaningful change.
- Public APIs must have `///` doc comments with at least one example.

## Commits

- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `perf:`, `chore:`.
- One logical change per commit.

## Versioning

- The version lives in `Cargo.toml` (`[workspace.package] version`); crates
  inherit it via `version.workspace = true`.
- **Bump the patch (`x`) on every code upgrade** that lands in the integration
  branch — one coherent change (feature, fix, or refactor) = one patch bump —
  in the same change that ships the code, so the running binary's reported
  version always identifies what is deployed.
- Moving to the next **minor** line (e.g. `0.1.x → 0.2.x`) is a deliberate
  milestone decision, not automatic per commit.
