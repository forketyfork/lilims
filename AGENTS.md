# Coding Agent Guidelines

This repository may be edited by automated coding agents. Please follow these rules when contributing automatically generated changes:

1. Keep Swift code formatted with Xcode defaults (4-space indentation).
2. Use Swift 6.1 features when compiling the app.
3. Keep Python scripts compatible with Python 3.13.
4. Place new documentation under `Docs/` and keep README up to date if the project structure changes.
5. When adding Swift files, include minimal DocC comments for public APIs.
6. Always run `swift test` before committing.
7. Run `ruff check Scripts` and `pytest Scripts/tests` to lint and test the Python utilities.
8. Mark completed tasks as done in `PLAN.md` when you finish a work item.

