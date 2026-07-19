import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weave.merge import _is_locked_file_error, _retry_on_lock


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label


def make_win_lock_error():
    e = OSError("The requested operation cannot be performed on a file with a user-mapped section open")
    e.winerror = 1224
    return e


# --- detects the specific Windows lock error ---
check("recognizes WinError 1224", _is_locked_file_error(make_win_lock_error()))
check("recognizes by message text even without winerror attr", _is_locked_file_error(OSError("...user-mapped section open...")))
check("does not misfire on unrelated OSError", not _is_locked_file_error(OSError("file not found")))
check("does not misfire on non-OSError", not _is_locked_file_error(ValueError("nope")))

# --- retries transient failures and eventually succeeds ---
calls = {"n": 0}


def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise make_win_lock_error()
    return "ok"


result = _retry_on_lock(flaky, retries=6, base_delay=0)
check("retries until success", result == "ok")
check("took exactly the expected number of attempts", calls["n"] == 3)

# --- gives up cleanly after exhausting retries, with a clear message ---
def always_fails():
    raise make_win_lock_error()


try:
    _retry_on_lock(always_fails, retries=3, base_delay=0)
    check("raises after exhausting retries", False)
except RuntimeError as e:
    check("raises a clear RuntimeError (not the raw WinError) after exhausting retries", "WinError 1224" in str(e))

# --- non-lock errors are never retried, just re-raised immediately ---
def unrelated_error():
    raise OSError("disk full")


try:
    _retry_on_lock(unrelated_error, retries=5, base_delay=0)
    check("unrelated OSError propagates", False)
except OSError as e:
    check("unrelated OSError propagates unchanged, not swallowed", str(e) == "disk full")

print("\nAll retry-helper tests passed.")
