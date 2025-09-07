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
    """
    Recursively scan for MP3/WAV, skip hidden/system/zero-byte files.
    """
    exts = ("mp3", "MP3", "wav", "WAV")
    file_paths = []
    for ext in exts:
        pattern = os.path.join(directory, "**", f"*.{ext}")
        file_paths.extend(glob.glob(pattern, recursive=True))

    # Normalize, dedupe, filter
    cleaned = []
    for p in set(os.path.normpath(fp) for fp in file_paths):
        try:
            if _is_hidden_or_system(p):
                continue
            if os.path.getsize(p) == 0:
                continue
        except OSError:
            continue
        cleaned.append(p)
    return sorted(cleaned)
