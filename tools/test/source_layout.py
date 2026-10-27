"""Resolve current production locations while frozen Git paths keep their identity.

Only source-reading test instruments use this mapping. Production never loads it.
"""
import json
import os
from pathlib import Path
import re
import subprocess
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = json.loads(Path(__file__).with_name('main_flow_layout.json').read_text(encoding='utf8'))


def current_source(relative):
    relative=str(relative)
    return ROOT / LAYOUT.get('current_overrides',{}).get(relative,LAYOUT['moved'].get(relative,relative))


@lru_cache(None)
def frozen_paths(revision):
    return set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision],
                                       cwd=ROOT).decode('utf8').splitlines())


def frozen_source(revision, relative):
    """Read a frozen identity even when a test now names its relocated target.

    Never substitute current contents or guess between multiple merged sources.
    Existing historical names always take precedence over reverse lookup.
    """
    relative = str(relative)
    available = frozen_paths(revision)
    name = relative
    if name not in available:
        manifest = json.loads((ROOT/'docs/development/ECM_DOMAIN_REFACTOR_MANIFEST.json').read_text(encoding='utf8'))
        # A moved implementation retains its frozen identity when it absorbs an
        # extra fragment. A target composed solely of multiple merges is ambiguous.
        requested = current_source(relative).resolve()
        candidates = {entry['source'] for entry in manifest['files']
                      if entry['action'] in ('move', 'move_split') and
                      requested == current_source(entry['targets'][0]).resolve() and
                      entry['source'] in available}
        if not candidates:
            candidates = {entry['source'] for entry in manifest['files']
                          if any(requested == current_source(target).resolve()
                                 for target in entry['targets']) and entry['source'] in available}
        if not candidates:
            candidates = {old for old, new in {**LAYOUT['moved'], **LAYOUT.get('current_overrides', {})}.items()
                          if current_source(new).resolve() == requested and old in available}
        assert len(candidates) == 1, ('ambiguous/missing frozen identity', revision, relative, sorted(candidates))
        name = candidates.pop()
    return subprocess.check_output(['git', 'show', revision+':'+name], cwd=ROOT).decode('utf-8-sig')


def current_files(legacy_folder):
    """Current files of one frozen family, without absorbing new subfamilies."""
    from instance_refactor import is_retired
    return sorted({current_source(old) for old,new in LAYOUT['moved'].items()
                   if old.startswith(legacy_folder.rstrip('/')+'/') and
                   not is_retired(current_source(old))})


def relocation_text(relative):
    """Undo only verified include-path edits for earlier frozen-body checks."""
    relative=str(relative)
    after=current_source(relative).read_text(encoding='utf-8-sig')
    if relative not in LAYOUT['include_consumers']:return after
    before=subprocess.check_output(['git','show',LAYOUT['baseline']+':'+relative],cwd=ROOT).decode('utf-8-sig').replace('\r\n','\n')
    pattern=re.compile(r'(?m)^(\s*#\s*include\s*["<])([^">\r\n]+)([">])')
    old=list(pattern.finditer(before));new=list(pattern.finditer(after));assert len(old)==len(new),relative
    roots=['include','src','src/core','src/cpu','src/cuda/common','kernels/cuda','kernels/opencl','third_party/cgbn/include','third_party/cgbn/include/cgbn']
    for a,b in reversed(list(zip(old,new))):
        if a[0]==b[0]:continue
        candidates=[(Path(relative).parent/a[2]).as_posix()]+[(Path(base)/a[2]).as_posix() for base in roots]
        expected=[]
        for candidate in candidates:
            key=os.path.normpath(candidate).replace('\\','/')
            path=current_source(key)
            if path.is_file():expected.append(path.resolve())
        active_roots=['include','src','src/ecm/cuda/runtime','src/ecm/cuda',
                      'third_party/cgbn/include','third_party/cgbn/include/cgbn']
        actual=next((p.resolve() for p in [current_source(relative).parent/b[2]]+
                     [ROOT/base/b[2] for base in active_roots] if p.is_file()),None)
        assert expected and actual==expected[0],('include edge changed',relative,a[2],b[2])
        after=after[:b.start()]+a[0]+after[b.end():]
    return after
