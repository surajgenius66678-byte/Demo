"""
Repo-root pytest bootstrap.

`pythonpath = . backend` in pytest.ini already puts both sys.path roots in
place for import *resolution*. This file additionally guarantees
*class identity* for the one module every part imports under two different
names: Part 2 imports it bare, as `shared.schemas`; Parts 4 and 5 import the
identical file as `backend.shared.schemas`. Left alone, Python would load
backend/shared/schemas.py twice under those two names, producing two
non-identical `Evidence`/`TaskType`/etc. classes — harmless within any one
part's own test suite (each is internally consistent), but exactly the bug
that would bite a test exercising the real integration (a Part 4 `Evidence`
handed to Part 2's planner, e.g. tests/integration/*). Aliasing here once,
before any test module imports either spelling, keeps one identity
everywhere. Same block, for the live app, in backend/api/main.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_BACKEND_DIR = _REPO_ROOT / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared.schemas as _schemas_mod  # noqa: E402
import shared as _shared_pkg  # noqa: E402
import backend as _backend_pkg  # noqa: E402

sys.modules.setdefault("backend.shared", _shared_pkg)
sys.modules.setdefault("backend.shared.schemas", _schemas_mod)
_backend_pkg.shared = _shared_pkg
