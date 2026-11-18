"""Local paths, kept out of the code. Put them in ntop_aero/local_paths.json (not in git; see local_paths.example.json)
or set the environment variables NTOP_AERO_STUDY, NTOP_AERO_STUDY_PYTHON, NTOP_AERO_FONTS, NTOP_AERO_LOGO.

  study          folder of the trade study whose implicit models are run here (needed for the V.. bodies only)
  study_python   a Python with torch and the field library (fieldlab); it turns a field into triangles and sections
  fonts, logo    optional, for the figures
"""
import json
import os
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
