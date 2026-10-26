"""Wait until a Space runs its latest commit; restart it once if the upload didn't start a rebuild.

Usage: python huggingface/wait_for_space.py prteek/sparse-panel-demo [timeout_seconds]
"""
import sys
import time

from huggingface_hub import HfApi

ERRORS = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
RESTART_AFTER = 120  # seconds still serving the old commit before forcing a restart


def main(repo: str, timeout: float = 900) -> None:
    api = HfApi()
    want = api.space_info(repo).sha
    start, restarted = time.time(), False
    while time.time() - start < timeout:
        runtime = api.get_space_runtime(repo)
        sha = (runtime.raw or {}).get("sha")
        print(f"{time.time() - start:5.0f}s  {runtime.stage}  running {str(sha)[:8]}  want {want[:8]}", flush=True)
        if runtime.stage in ERRORS:
            sys.exit(f"{repo} failed: {runtime.stage} {(runtime.raw or {}).get('errorMessage', '')}")
        if runtime.stage == "RUNNING" and sha == want:
            return
        if runtime.stage == "RUNNING" and not restarted and time.time() - start > RESTART_AFTER:
            print("still on the old commit; restarting the Space", flush=True)
            api.restart_space(repo)
            restarted = True
        time.sleep(15)
    sys.exit(f"{repo} not running {want[:8]} after {timeout:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], *(float(a) for a in sys.argv[2:]))
