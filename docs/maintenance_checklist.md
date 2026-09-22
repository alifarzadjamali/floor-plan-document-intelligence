# Maintenance checklist

Use this checklist when making a documentation or tooling-only update after the
frozen evaluation runs have been completed.

1. Keep the held-out metrics, checkpoints, and generated result directories
   unchanged.
2. Confirm that documentation still distinguishes qualitative examples from
   measured evaluation evidence.
3. Run the focused test suite when a documented interface or schema changes.
4. Record any change to the structured-output contract in `output_schema.md`
   before updating examples or downstream consumers.
5. Keep dataset archives, OCR installations, and local virtual environments
   outside version control.
