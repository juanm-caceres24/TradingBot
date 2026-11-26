# Git Workflow Guide

This repository uses a **Simplified Solo Git Flow** mirrored directly from the companion `NeuralNetwork` project, balancing a clean history with safe, structured releases.

---

## 1. Branch Model

| Branch | Purpose | Rules |
| :--- | :--- | :--- |
| `main` | Stable, tested milestones/releases. | **Never commit directly.** Only receives merges from `development` (releases) or `hotfix/*`. Tagged as `vX.Y.Z`. |
| `development` | Active integration branch. | Must always pass `./scripts/test.sh`. |
| `feat/<topic>` | New features or modules (e.g., `feat/sqlite-storage`, `feat/binance-provider`). | Branched from `development`; merged back to `development` and deleted once complete. |
| `fix/<topic>` | Bug fixes during development (e.g., `fix/gap-alignment`). | Branched from `development`; merged back and deleted. |
| `perf/<topic>` | Performance optimizations (e.g., `perf/indicator-cache`). | Branched from `development`; merged back and deleted. |
| `exp/<topic>` | Throwaway strategy or feature experiments. | Branched from `development`; delete freely if discarded. |
| `hotfix/<topic>` | Urgent fix on a released version in `main`. | Branched from `main`; merged back into both `main` and `development`. |

---

## 2. Commit Message Convention (Conventional Commits)

All commit messages, branch names, code, comments, and documentation are written in **English**.

Format:

```text
<type>(<scope>): <short imperative description>
```

### Common Types
- `feat`: New functionality (`feat(storage): add SQLite candle store and gap detector`)
- `fix`: Bug fix (`fix(config): reject boolean values for integer fields`)
- `test`: Adding or updating tests (`test(storage): cover internal and verified gap cases`)
- `docs`: Documentation changes (`docs: add project specification and git workflow`)
- `perf`: Performance improvement (`perf(indicators): vectorize sliding regression`)
- `refactor`: Code restructuring without changing behavior
- `build`: Packaging, scripts, or dependency changes (`build: add pyproject.toml and test script`)
- `chore`: Maintenance tasks (`.gitignore`, cleanup)

---

## 3. Standard Commands

### Start a New Feature
```bash
git checkout development
git checkout -b feat/my-feature
```

### Commit Incrementally
```bash
./scripts/test.sh
git status
git add <files>
git commit -m "feat(core): implement feature summary"
```

### Merge a Feature into `development`
Use `--no-ff` to preserve the feature grouping in history:
```bash
git checkout development
git merge --no-ff feat/my-feature -m "merge: integrate feat/my-feature into development"
git branch -d feat/my-feature
```

### Cut a Release to `main`
Once a milestone in `development` is complete and all tests pass:
```bash
./scripts/test.sh

git checkout main
git merge --no-ff development -m "release: v0.1.0 project setup, strict config, models, and sqlite storage"
git tag -a v0.1.0 -m "v0.1.0: project setup, strict config, models, and sqlite storage"
git checkout development
```

### Push to Remote
```bash
git push origin development
git push origin main --tags
```

---

## 4. Repository Hygiene Rules
- Never commit `.venv/`, `__pycache__/`, `data/`, SQLite `.db` files, exported `.csv` datasets, trained models, or `.env` secrets (enforced by `.gitignore`).
- Keep small example configs and provider presets versioned in `configs/`.
- Update relevant files in `docs/` and unit tests in `tests/` inside the same feature branch as the code they document/verify.
