"""Resolve current production locations while frozen Git paths keep their identity.

Only source-reading test instruments use this mapping. Production never loads it.
"""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = json.loads(Path(__file__).with_name('main_flow_layout.json').read_text(encoding='utf8'))


def current_source(relative):
    return ROOT / LAYOUT['moved'].get(str(relative), str(relative))


def current_files(legacy_folder):
    """Current files of one frozen family, without absorbing new subfamilies."""
    return sorted(ROOT/new for old,new in LAYOUT['moved'].items()
                  if old.startswith(legacy_folder.rstrip('/')+'/'))


def relocation_text(relative):
    """Undo only verified include-path edits for earlier frozen-body checks."""
    relative=str(relative)
    after=current_source(relative).read_text(encoding='utf-8-sig')
    if relative not in LAYOUT['include_consumers']:return after
    before=subprocess.check_output(['git','show',LAYOUT['baseline']+':'+relative],cwd=ROOT).decode('utf-8-sig').replace('\r\n','\n')
    pattern=re.compile(r'(?m)^(\s*#\s*include\s*["<])([^">\r\n]+)([">])')
    old=list(pattern.finditer(before));new=list(pattern.finditer(after));assert len(old)==len(new),relative
    roots=['include','src','src/core','src/cpu','src/cuda/common','kernels/cuda','third_party/cgbn/include','third_party/cgbn/include/cgbn']
    for a,b in reversed(list(zip(old,new))):
        if a[0]==b[0]:continue
        candidates=[(Path(relative).parent/a[2]).as_posix()]+[(Path(base)/a[2]).as_posix() for base in roots]
        expected=[]
        for candidate in candidates:
            key=os.path.normpath(candidate).replace('\\','/')
            path=current_source(key)
            if path.is_file():expected.append(path.resolve())
        actual=(current_source(relative).parent/b[2]).resolve()
        assert expected and actual==expected[0],('include edge changed',relative,a[2],b[2])
        after=after[:b.start()]+a[0]+after[b.end():]
    return after
