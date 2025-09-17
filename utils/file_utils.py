import os
import glob
from typing import List

def _is_hidden_or_system(path: str) -> bool:
    # Windows: use file attributes
    if os.name == "nt":
        import ctypes
        FILE_ATTRIBUTE_HIDDEN = 0x2
        FILE_ATTRIBUTE_SYSTEM = 0x4
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == -1:
                return False
            return bool(attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM))
        except Exception:
            return False
    # POSIX: treat dotfiles as hidden
    base = os.path.basename(path)
    return base.startswith(".")

def get_audio_files(directory: str) -> List[str]:
    """Recursively scan ``directory`` for supported audio files.

    Uses ``os.walk`` so that directory names containing glob meta-characters
    (``[]``, ``?`` …) are handled correctly. Hidden/system entries and empty
    files are skipped to match the previous behaviour.
    """

    directory = os.fspath(directory)
    # normalise extensions once to avoid repeated allocations inside the loop
    exts = {".mp3", ".wav"}
    collected: List[str] = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden/system directories to avoid traversing them entirely
        dirs[:] = [
            d for d in dirs
            if not _is_hidden_or_system(os.path.join(root, d))
        ]

        for name in files:
            path = os.path.normpath(os.path.join(root, name))
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            try:
                if _is_hidden_or_system(path):
                    continue
                if os.path.getsize(path) == 0:
                    continue
            except OSError:
                continue
            collected.append(path)

    # ``set`` removes duplicates, ``sorted`` keeps deterministic ordering
    return sorted(set(collected))