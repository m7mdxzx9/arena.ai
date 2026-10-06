"""Run learner Python code in a separate, resource-limited process.

Defence in depth for a *local, single-player* game:
  * separate subprocess (isolated mode `-I`, empty environment, temporary working directory),
  * wall-clock timeout + RLIMIT_CPU, RLIMIT_AS (memory), RLIMIT_FSIZE, RLIMIT_NOFILE,
  * a Python audit hook that blocks networking, process spawning and writes outside the temp dir,
  * output size caps.
This is NOT a hardened multi-tenant sandbox. If you host Neural Forge for untrusted users, run the code runner
inside a container/VM with no network (see docs/SECURITY.md).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

TIMEOUT_S = 10
CPU_S = 6
MEM_BYTES = 2 * 1024 ** 3
MAX_OUT = 20_000

HOOK = r'''
BLOCK_PREFIX = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.fork", "os.posix_spawn", "os.kill",
                "ctypes.", "shutil.rmtree", "os.rmdir", "winreg.", "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.")
def _hook(event, args):
    if event.startswith(BLOCK_PREFIX):
        raise PermissionError(f"Blocked in the Neural Forge sandbox: {event}")
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else "r"
        if isinstance(mode, str) and any(c in mode for c in "wax+") and isinstance(path, (str, bytes)):
            p = os.path.abspath(path if isinstance(path, str) else path.decode())
            if not p.startswith(TMP):
                raise PermissionError("Blocked: writing files outside the sandbox directory")
    if event in ("os.remove", "os.unlink", "os.rename") and args and isinstance(args[0], str):
        if not os.path.abspath(args[0]).startswith(TMP):
            raise PermissionError("Blocked: modifying files outside the sandbox directory")

'''

HARNESS = r'''
import sys, json, io, contextlib, math, traceback, os
TMP = os.getcwd()
import numpy as np            # pre-import libraries before the audit hook is installed
_spec_txt = open("spec.json").read() + open("user_code.py").read()
if "pandas" in _spec_txt or "pd." in _spec_txt:
    import pandas as pd
if "sklearn" in _spec_txt:
    try:
        import sklearn.linear_model, sklearn.model_selection, sklearn.datasets, sklearn.metrics, sklearn.tree, sklearn.neighbors, sklearn.ensemble, sklearn.preprocessing
    except Exception:
        pass
@@HOOK@@
def approx(a, b, tol):
    if isinstance(b, float) or isinstance(a, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
        except Exception:
            return False
    if isinstance(b, (list, tuple)):
        try:
            a = list(a)
        except Exception:
            return False
        return len(a) == len(b) and all(approx(x, y, tol) for x, y in zip(a, b))
    if isinstance(b, dict):
        return isinstance(a, dict) and set(map(str, a)) == set(map(str, b)) and all(approx(a[k] if k in a else a[int(k)] if str(k).isdigit() and int(k) in a else None, b[k], tol) for k in b)
    if isinstance(b, bool):
        return bool(a) == b and isinstance(a, (bool, np.bool_))
    try:
        return a == b or (hasattr(a, "item") and a.item() == b)
    except Exception:
        return False

def short(v):
    try:
        if isinstance(v, np.ndarray):
            v = v.tolist()
        s = repr(v)
    except Exception:
        s = "<unprintable>"
    return s[:300]

spec = json.load(open("spec.json"))
code = open("user_code.py").read()
out = io.StringIO()
result = {"ok": False, "tests": [], "stdout": "", "error": None}
sys.addaudithook(_hook)
ns = {"__name__": "__main__"}
try:
    with contextlib.redirect_stdout(out):
        exec(compile(spec.get("setup", ""), "setup", "exec"), ns)
        exec(compile(code, "your_code.py", "exec"), ns)
        for t in spec.get("tests", []):
            entry = {"name": t["name"]}
            try:
                if t.get("setup"):
                    exec(t["setup"], ns)
                got = eval(t["expr"], ns)
                entry["got"] = short(got)
                if "expected" in t:
                    entry["expected"] = short(t["expected"])
                    entry["passed"] = bool(approx(got, t["expected"], t.get("tol", 1e-6)))
                else:
                    entry["passed"] = bool(got)
            except Exception as e:
                entry["passed"] = False
                entry["error"] = f"{type(e).__name__}: {e}"
            result["tests"].append(entry)
    result["ok"] = all(t["passed"] for t in result["tests"]) if result["tests"] else True
except Exception as e:
    tb = traceback.format_exc()
    if 'File "your_code.py"' in tb:
        tb = "Traceback (most recent call last):\n  " + tb[tb.index('File "your_code.py"'):]
    result["error"] = tb[-1500:]
result["stdout"] = out.getvalue()[:8000]
sys.__stdout__.write("\n@@RESULT@@" + json.dumps(result))
'''

TRACE_HARNESS = r'''
import sys, json, os
TMP = os.getcwd()
@@HOOK@@
code = open("user_code.py").read()
sys.addaudithook(_hook)
steps = []
def tracer(frame, event, arg):
    if frame.f_code.co_filename != "snippet":
        return tracer
    if event == "line":
        vars_ = {k: repr(v)[:60] for k, v in frame.f_locals.items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
        steps.append({"line": frame.f_lineno, "vars": vars_})
        if len(steps) > 300:
            raise RuntimeError("Trace limit reached (300 steps) — is there an infinite loop?")
    return tracer
g = {"__name__": "__main__"}
err = None
import io, contextlib
out = io.StringIO()
try:
    with contextlib.redirect_stdout(out):
        sys.settrace(tracer)
        exec(compile(code, "snippet", "exec"), g)
        sys.settrace(None)
except Exception as e:
    sys.settrace(None)
    err = f"{type(e).__name__}: {e}"
final = {k: repr(v)[:60] for k, v in g.items() if not k.startswith("__") and isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
sys.__stdout__.write("\n@@RESULT@@" + json.dumps({"steps": steps, "final": final, "error": err, "stdout": out.getvalue()[:4000]}))
'''


HARNESS = HARNESS.replace("@@HOOK@@", HOOK)
TRACE_HARNESS = TRACE_HARNESS.replace("@@HOOK@@", HOOK)


def _limits():  # runs in the child before exec
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_S, CPU_S))
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 ** 2, 2 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    os.setsid()


def _run(harness: str, code: str, spec: dict | None = None) -> dict:
    if len(code) > 20_000:
        return {"ok": False, "error": "Code too long (20,000 characters max).", "tests": []}
    with tempfile.TemporaryDirectory(prefix="nf_sandbox_") as tmp:
        with open(os.path.join(tmp, "user_code.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmp, "spec.json"), "w") as f:
            json.dump(spec or {}, f)
        with open(os.path.join(tmp, "harness.py"), "w") as f:
            f.write(harness)
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run([sys.executable, "-I", "harness.py"], cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=TIMEOUT_S, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Time limit exceeded ({TIMEOUT_S}s). Infinite loop?", "tests": [], "stdout": ""}
        out = proc.stdout[-MAX_OUT * 5:]
        if "@@RESULT@@" in out:
            try:
                return json.loads(out.split("@@RESULT@@")[-1])
            except json.JSONDecodeError:
                pass
        err = (proc.stderr or "").strip()[-1500:]
        if proc.returncode < 0 or "MemoryError" in err:
            err = (err + "\n" if err else "") + "Process was stopped by the sandbox (CPU/memory limit)."
        return {"ok": False, "error": err or "Your code crashed the runner.", "tests": [], "stdout": proc.stdout[:4000]}


def run_tests(code: str, spec: dict) -> dict:
    return _run(HARNESS, code, spec)


def run_free(code: str) -> dict:
    return _run(HARNESS, code, {"tests": []})


def trace(code: str) -> dict:
    return _run(TRACE_HARNESS, textwrap.dedent(code))
