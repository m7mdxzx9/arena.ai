from neural_forge import code_exercises, sandbox


def test_runs_code_and_captures_stdout():
    r = sandbox.run_free("print(sum(range(5)))")
    assert r["ok"] and r["stdout"].strip() == "10"


def test_blocks_dangerous_operations():
    for code in ("import socket; socket.socket()", "import subprocess; subprocess.run(['ls'])", "import os; os.system('ls')", "open('/etc/passwd').read()", "import os; os.listdir('/')", "open('/tmp/x.txt', 'w').write('x')"):
        r = sandbox.run_free(code)
        assert not r["ok"] or r["error"], code


def test_lazy_imports_and_local_files_still_work():
    r = sandbox.run_free("import json, statistics, collections\nopen('notes.txt', 'w').write('hi')\nprint(open('notes.txt').read(), statistics.mean([1, 2, 3]))")
    assert r["ok"] and r["stdout"].split() == ["hi", "2"], r
    r = sandbox.run_free("import pandas as pd\nprint(pd.DataFrame({'a': [1, 2]}).a.sum())")
    assert r["ok"] and r["stdout"].strip() == "3", r


def test_timeout():
    r = sandbox.run_free("while True: pass")
    assert not r["ok"] and r["error"]


def test_trace_steps():
    t = sandbox.trace("x = 1\nfor i in range(3):\n    x = x * 2\nprint(x)")
    assert t["stdout"].strip() == "8"
    assert t["steps"][-1]["vars"]["x"] == "8"


def test_every_exercise_solution_passes_and_starter_fails():
    for e in code_exercises.EXERCISES:
        spec = dict(setup=e.get("setup", ""), tests=e["tests"])
        good = sandbox.run_tests(e["solution"], spec)
        assert good["tests"] and all(t["passed"] for t in good["tests"]), (e["id"], good)
        bad = sandbox.run_tests(e["starter"], spec)
        assert not (bad["tests"] and all(t["passed"] for t in bad["tests"])), e["id"]
