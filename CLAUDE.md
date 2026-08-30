# Project rules

- Load models/data once in dedicated setup; never reload without asking.
- Save plots to disk as PNGs.
- Never delete or overwrite a checkpoint or activation cache without asking.
- Functions take inputs as arguments; no module-level globals.
- Minimal code. No abstraction layers, no config frameworks, no defensive
  wrappers. Simplest thing that runs.
- Don't write verification code. Flag where a check is needed and the user
  will write it.
- Commit often, small commits. Never mention Claude, AI, or agents in commit
  messages or code comments.
- Any number in a result must be reproducible by re-running a named script.
