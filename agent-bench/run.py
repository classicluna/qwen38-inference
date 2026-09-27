#!/usr/bin/env python3
"""Run the coding-agent benchmark: one fresh repo copy per task, `omp -p` on task.md, then grade by
copying hidden_tests/ into <copy>/tests_hidden and running pytest on it (pass = all hidden tests pass).

  run.py --tag bonsai --model llama.cpp/bonsai-2-27b            # all tasks
  run.py --dry-run                                               # grade untouched repos (expect 0/N)
  run.py --reference                                             # grade reference fixes (expect N/N)
"""
import argparse, json, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/home/evank/dev/inference/.venv-eval/bin/python"


def grade(work, hidden):
    shutil.copytree(hidden, work / "tests_hidden", dirs_exist_ok=True)
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests_hidden"], cwd=work,
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0, r.stdout[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dry")
    ap.add_argument("--model")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--max-time", default="900")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reference", action="store_true")
    a = ap.parse_args()
    out_root = HERE.parent / "results" / "agent-bench" / a.tag
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p for p in (HERE / "tasks").iterdir() if p.is_dir())
    if a.tasks:
        tasks = [t for t in tasks if any(t.name.startswith(x) for x in a.tasks.split(","))]
    rows = []
    for t in tasks:
        work = Path(tempfile.mkdtemp(prefix=f"ab-{t.name}-"))
        shutil.copytree(t / "repo", work, dirs_exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=work)
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=ab", "add", "-A"], cwd=work)
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=ab", "commit", "-qm", "init"], cwd=work)
        od = out_root / t.name
        od.mkdir(parents=True, exist_ok=True)
        wall, tools, rc = 0.0, None, None
        if a.reference:
            shutil.copytree(t / "reference", work, dirs_exist_ok=True)
        elif not a.dry_run:
            t0 = time.time()
            p = subprocess.run(["omp", "-p", (t / "task.md").read_text(), "--cwd", str(work), "--model", a.model,
                                "--no-session", "--no-title", "--max-time", a.max_time, "--mode", "json"],
                               capture_output=True, text=True, timeout=int(a.max_time) + 120)
            wall, rc = time.time() - t0, p.returncode
            (od / "stdout.jsonl").write_text(p.stdout)
            (od / "stderr.txt").write_text(p.stderr)
            tools = sum(1 for line in p.stdout.splitlines() if '"tool_execution_start"' in line
                        or '"type":"toolCall"' in line.replace(" ", ""))
        diff = subprocess.run(["git", "status", "--porcelain"], cwd=work, capture_output=True, text=True).stdout
        ok, log = grade(work, t / "hidden_tests")
        (od / "grade.txt").write_text(log)
        (od / "diff.txt").write_text(subprocess.run(["git", "diff"], cwd=work, capture_output=True, text=True).stdout)
        row = {"task": t.name, "pass": ok, "wall_s": round(wall, 1), "tool_events": tools, "rc": rc,
               "changed": diff.count("\n")}
        rows.append(row)
        print(json.dumps(row), flush=True)
        shutil.rmtree(work, ignore_errors=True)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"== {a.tag}: {sum(r['pass'] for r in rows)}/{len(rows)} passed, "
          f"{sum(r['wall_s'] for r in rows)/60:.1f} min total")


if __name__ == "__main__":
    sys.exit(main())
