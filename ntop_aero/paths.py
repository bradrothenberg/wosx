"""Local paths, kept out of the code. Put them in ntop_aero/local_paths.json (not in git; see local_paths.example.json)
or set the environment variables NTOP_AERO_STUDY, NTOP_AERO_STUDY_PYTHON, NTOP_AERO_FONTS, NTOP_AERO_LOGO.

  study          folder of the trade study whose implicit models are run here (needed for the V.. bodies only)
  study_python   a Python with torch and the field library (fieldlab); it turns a field into triangles and sections
  fonts, logo    optional, for the figures
  busy_process   optional: the name of another compute program (for example a CFD solver). The timing scripts record
                 if it runs at the same time, because their times are then not clean
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_f = HERE / 'local_paths.json'
_cfg = json.loads(_f.read_text()) if _f.exists() else {}


def _get(name, default=''):
    return os.environ.get('NTOP_AERO_' + name.upper(), _cfg.get(name, default))


STUDY = Path(_get('study', 'study'))
STUDY_PY = _get('study_python', sys.executable)
FONTS = Path(_get('fonts', 'fonts'))
LOGO = _get('logo', '')
BUSY_PROCESS = _get('busy_process', '').lower()


def other_job_running():
    """True if the program named by 'busy_process' is running now (Windows task list). None: not set, or not known."""
    if not BUSY_PROCESS:
        return None
    try:
        return BUSY_PROCESS in subprocess.run(['tasklist'], capture_output=True, text=True, timeout=20).stdout.lower()
    except Exception:                                    # noqa: BLE001
        return None
