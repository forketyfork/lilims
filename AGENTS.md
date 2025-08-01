# Coding Agent Guidelines

This repository may be edited by automated coding agents. 

## Important documentation
The documentation to the project is located in the `Docs` directory:
- `PLAN.md` — implementation plan.
- `Glossary.md` — glossary of important terms.

Please follow these rules when contributing automatically generated changes:

1. Keep Swift code formatted with Xcode defaults (4-space indentation).
2. Use Swift 6.0 features when compiling the app.
3. Keep Python scripts compatible with Python 3.13.
4. Place new documentation under `Docs/` and keep README.md up to date if the project structure changes.
5. When adding Swift files, include minimal DocC comments for public APIs.
6. Always run `swift test` before committing.
7. Run `ruff check Scripts` and `pytest Scripts/tests` to lint and test the Python utilities.
8. Mark completed tasks as done in `Docs/PLAN.md` when you finish a work item.

