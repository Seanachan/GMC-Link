# Stage D Gatekeeper — TempRMOT upstream acquisition

Date: 2026-04-22

## Probe

```
curl -sI -o /dev/null -w "%{http_code}" https://github.com/zyn213/TempRMOT
→ 200
```

`zyn213/TempRMOT` is reachable. `HELLORPG/TempRMOT` → 404 (not the intended
fork; keep `zyn213` as canonical).

## Decision

**REACHABLE — acquisition gated on user consent.** The spec plan Task 15 step 1
adds the repo as a git submodule and installs it editable. Because this
modifies shared state (`.gitmodules`, `third_party/`, pip-installs a package
from unvetted code into the conda env), we pause before executing.

## Next

When user approves:

```bash
mkdir -p third_party
git submodule add https://github.com/zyn213/TempRMOT third_party/TempRMOT
cd third_party/TempRMOT
git log -1 --format='%H %s'  # record pinned SHA in commit message
pip install -e .
cd -
git add .gitmodules third_party/TempRMOT
git commit -m "chore(exp37): pin TempRMOT submodule <pinned-sha>"
```

Per spec §10 kill switch #3, if the code has unfixable dependency conflicts
with our RMOT conda env, Stage D becomes "portability scope deferred" and
Exp 37 headline narrows to Stages A/B/C aligner-quality + ATE only.
