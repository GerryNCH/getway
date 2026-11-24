"""
Pre-push smoke test — Nov 2026.

Rule (Gerry, Nov 2026): before every push that touches the backend, actually
IMPORT main.py (or briefly start the server), not just py_compile it.
py_compile only catches syntax errors. It does NOT catch a bug like the one
that crashed production on Nov 24 2026: a FastAPI route decorator
(`dependencies=[Depends(some_func)]`) is evaluated at *import time*, in
source order. If `some_func` is defined further down in the file than the
route that references it, the whole app fails to import with a NameError —
and Railway then shows every route as 502, not just the broken one.
py_compile cannot see this because it only checks syntax, never executes or
resolves names.

What this script does:
  1. Actually `import main` (catches the above, plus any other import-time
     error: missing dependency, typo in a module-level name, bad decorator,
     etc).
  2. Briefly starts the real uvicorn server and waits for
     "Application startup complete" (catches startup-hook errors that a
     plain import wouldn't trigger).

Usage (run from the backend/ folder, same folder as main.py):
    python3 smoke_test.py

Needs: everything in requirements.txt installed, plus a real
ANTHROPIC_API_KEY in backend/.env (or exported) if you want step 2 to fully
match production — a dummy key is fine for just catching import/startup
errors, since the key is only read, never called, during startup.

Exit code 0 = safe to push. Non-zero = DO NOT PUSH, read the traceback above.
"""
import os
import subprocess
import sys
import tempfile

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def check_import():
    print("=== Step 1/2: import main.py ===")
    with tempfile.TemporaryDirectory() as scratch_data_dir:
        env = dict(os.environ)
        env.setdefault("ANTHROPIC_API_KEY", "sk-dummy-smoke-test-key")
        env["DATA_DIR"] = scratch_data_dir
        result = subprocess.run(
            [sys.executable, "-c", "import main; print('IMPORT OK')"],
            cwd=BACKEND_DIR,
            env=env,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            print("\n*** IMPORT FAILED — do not push. ***")
            return False
        print("Import OK.\n")
        return True


def check_server_start():
    print("=== Step 2/2: brief server start (uvicorn) ===")
    with tempfile.TemporaryDirectory() as scratch_data_dir:
        env = dict(os.environ)
        env.setdefault("ANTHROPIC_API_KEY", "sk-dummy-smoke-test-key")
        env["DATA_DIR"] = scratch_data_dir
        try:
            result = subprocess.run(
                [sys.executable, "-m", "uvicorn", "main:app", "--port", "8322"],
                cwd=BACKEND_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=6,
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired as e:
            # Expected: uvicorn runs forever until we kill it. A 6s timeout
            # is our "briefly start it" window; we just check what it
            # printed before being killed.
            output = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            output += (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")

        print(output)
        if "Application startup complete" in output:
            print("Server started cleanly.\n")
            return True
        print("*** SERVER FAILED TO START — do not push. ***")
        return False


if __name__ == "__main__":
    ok_import = check_import()
    # Only attempt the server-start check if the import itself succeeded —
    # a failed import will fail the server start the same way, with a
    # noisier traceback, so there's nothing more to learn from it.
    ok_server = check_server_start() if ok_import else False

    if ok_import and ok_server:
        print("PASS — safe to push.")
        sys.exit(0)
    else:
        print("FAIL — fix the error above before pushing.")
        sys.exit(1)
