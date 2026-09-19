"""Keep validation data out of anything we train on.

Team rule: views recorded during a validation run (api.py's recordings/, or any
copy of them) may be used to evaluate and to inform decisions, never as
training input, not even as unlabelled backgrounds. Every synth script calls
``assert_training_input`` on each input root before reading it, and writes the
checked roots into its manifest.
"""

import re
from pathlib import Path

# The recorder's file names: <arrival>_frame_<frame>_idx_<frame_index>.json/.png
_RECORDER_FILE = re.compile(r'^(\d{5}_)?frame_\d{6}_idx_\d{6}\.(json|png)$')
_FORBIDDEN_PARTS = ('recordings', 'recording_sample', 'val_full')


class ValidationDataError(RuntimeError):
    pass


def assert_training_input(path) -> Path:
    """Raise if ``path`` is, contains, or sits inside recorded validation data."""
    path = Path(path).resolve()
    for part in path.parts:
        if any(part.lower().startswith(bad) for bad in _FORBIDDEN_PARTS):
            raise ValidationDataError(f'{path}: {part!r} looks like recorded validation data; not a training input')
    if path.is_dir():
        for i, child in enumerate(path.rglob('*')):
            if _RECORDER_FILE.match(child.name):
                raise ValidationDataError(f'{path} contains recorder output ({child.name}); not a training input')
            if i > 200000:
                break
    elif _RECORDER_FILE.match(path.name):
        raise ValidationDataError(f'{path} is recorder output; not a training input')
    return path
