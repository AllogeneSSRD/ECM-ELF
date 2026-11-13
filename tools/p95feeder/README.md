# Archived Prime95 feeder

This directory preserves the legacy `ecm_p95feeder` and its version 6
MIDSTAGE reader/writer, inspection tool and fixtures. The archived build and
test scripts describe the old integration; they are not build entry points
for the current repository. The feeder used a legacy `.tmp` input contract
and is not shipped or built by the root project.

`ecm_edwards_save_test.cpp` is preserved byte for byte as the source of the
historical point used by the Edwards large-bound comparison.

Current Stage1 handoff uses `p95_worktodo_path` and `p95_sync_dir`; see
[Stage1 usage](../../docs/usage/STAGE1.md). The CPU Edwards Stage1 checkpoint
codec remains in `src/io/prime95_save.cpp` and is separate from this archive.
