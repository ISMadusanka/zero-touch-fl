"""Tests for the CUDA-lib preload shim in rl/policy.py.

`_preload_pip_cuda_libs` fixes the "libnvJitLink.so.13: cannot open shared object
file" failure that bitsandbytes (pulled in by unsloth) hits when pip's nvidia-*
wheels aren't on the linker path. These tests validate the *discovery* logic (find
the .so under a site-packages `nvidia/*/lib` layout on sys.path) and that the
preload is best-effort and never raises. They run on any OS — no torch, no CUDA,
no real .so (a fake, non-loadable file stands in).

Run:  python tests/test_cuda_preload.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.policy import _find_pip_cuda_libs, _preload_pip_cuda_libs  # noqa: E402


def _make_fake_wheel(root, soname="libnvJitLink.so.13"):
    """Create <root>/nvidia/nvjitlink/lib/<soname> and return the file path."""
    libdir = os.path.join(root, "nvidia", "nvjitlink", "lib")
    os.makedirs(libdir, exist_ok=True)
    path = os.path.join(libdir, soname)
    with open(path, "wb") as f:
        f.write(b"\x00")           # not a real ELF — CDLL will refuse it
    return path


def test_discovers_lib_on_syspath(tmp_path):
    fake = _make_fake_wheel(str(tmp_path))
    sys.path.insert(0, str(tmp_path))
    try:
        found = _find_pip_cuda_libs(("libnvJitLink.so.13",))
    finally:
        sys.path.remove(str(tmp_path))
    assert fake in found, (fake, found)
    print("ok: discovers lib under nvidia/*/lib on sys.path")


def test_no_nvidia_dir_returns_empty(tmp_path):
    # tmp_path has no nvidia/ subdir -> nothing found, no error.
    sys.path.insert(0, str(tmp_path))
    try:
        assert _find_pip_cuda_libs(("libnvJitLink.so.13",)) == []
    finally:
        sys.path.remove(str(tmp_path))
    print("ok: absent nvidia dir -> empty, no crash")


def test_unmatched_soname_returns_empty(tmp_path):
    _make_fake_wheel(str(tmp_path))               # installs the .13
    sys.path.insert(0, str(tmp_path))
    try:
        # A different soname must not match the .13 file.
        assert _find_pip_cuda_libs(("libnvJitLink.so.99",)) == []
    finally:
        sys.path.remove(str(tmp_path))
    print("ok: unmatched soname -> empty")


def test_preload_is_best_effort_and_silent(tmp_path):
    # The fake file exists but is not a valid shared object -> CDLL raises OSError,
    # which the shim must swallow, returning [] without propagating.
    _make_fake_wheel(str(tmp_path))
    sys.path.insert(0, str(tmp_path))
    try:
        loaded = _preload_pip_cuda_libs(("libnvJitLink.so.13",))
    finally:
        sys.path.remove(str(tmp_path))
    assert loaded == [], loaded          # invalid .so -> nothing successfully loaded
    print("ok: preload swallows load errors (best-effort)")


def test_preload_default_sonames_no_crash():
    # Called with the real default set on a box without the libs -> must not raise.
    result = _preload_pip_cuda_libs()
    assert isinstance(result, list)
    print("ok: default preload never raises")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            import inspect
            if inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            passed += 1
    print(f"\nAll {passed} cuda-preload tests passed.")
