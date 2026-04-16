# Package Import Guide for transformer-from-scratch

## Why the problem happened

When you run a Python file directly like this:

```powershell
python experiments/phase_03_single_head_attention/task_01_similarity_scores.py
```

Python sets `sys.path` so the directory containing that script is the only usable root for imports. That means the top-level `experiments` package is not automatically visible, so this import fails:

```python
from experiments.phase_02_positional_encoding.task_02_add_position_to_embedding import embeddings_with_pos
```

The code is trying to import from a package named `experiments`, but Python does not treat the project root as an import root in that mode.

## What we changed

To make the package import work:

- Added `__init__.py` files in `experiments/` and its phase subfolders.
- Updated module imports inside the package to use relative package imports where appropriate.

That means modules inside `experiments/` can now be imported as part of the package hierarchy.

## Permanent fix

For package-style execution, run from the project root using `-m`:

```powershell
cd "J:\My Drive\Coding\Data Science Project\transformer-from-scratch"
python -m experiments.phase_03_single_head_attention.task_01_similarity_scores
```

This tells Python to load the module as part of the `experiments` package.

## Direct-run support for Ctrl+Enter

Some editors run the current file directly as a script (for example, VS Code's `Run Python File in Dedicated Terminal` / Ctrl+Enter). In that case, the file can still work by bootstrapping the project root into `sys.path` before package imports.

The current task file now includes a small startup guard that inserts the project root into `sys.path` when it is executed directly. That means you can run the file with Ctrl+Enter and the package import will still resolve.

## What to do if it happens again

1. Check for missing `__init__.py` files.
   - Add empty `__init__.py` files to `experiments/` and the relevant subpackage folders.
2. Confirm the import path is valid from the project root.
   - `from experiments...` works only when the project root is on `PYTHONPATH`.
3. Prefer package module execution, not direct script execution.
   - Use `python -m package.module` instead of `python path/to/module.py`.
4. If using VS Code, make sure the workspace folder is the current working directory.
   - In `launch.json`, set `cwd` to the project root or use a `python` path that includes the root.

## Useful rule

- If a file belongs to a package and imports sibling modules, run it as a package module.
- If you want to allow direct script execution, add a small guard and manage import paths carefully.

Example guard:

```python
if __name__ == "__main__":
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    from experiments.phase_02_positional_encoding.task_02_add_position_to_embedding import embeddings_with_pos
```

That is a fallback only if you need direct file execution, but the cleaner solution is `python -m ...`.

## Summary

- The error happened because the file was run as a standalone script, not as part of the package.
- The permanent solution is to use package imports with `__init__.py` and run with `python -m` from the project root.
- If it happens again, verify package structure, confirm root import path, and use module execution.
