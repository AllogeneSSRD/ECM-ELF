"""Audit P0 bytes and the explicitly authorized additive v2 schema exception."""
import argparse, hashlib, json, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parents[3]; baseline=json.loads(a.baseline.read_text())
    checked=[]
    for name,digest in baseline['protected'].items():
        if name=='config/ecm_options.json': continue
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        checked.append(name)
    old=json.loads(subprocess.check_output(['git','show','cb91212:config/ecm_options.json'],cwd=root))
    current=json.loads((root/'config/ecm_options.json').read_text(encoding='utf-8'))
    addition=current.pop('gui_v2_sidecar');assert addition
    # User explicitly authorized GPU clock draft persistence in the main TOML.
    # This one empty-default GUI-only field is additive; every existing option
    # and all protected legacy GUI bytes remain exactly equal to the baseline.
    clocks=[o for o in current['options'] if o['owner']=='gui' and o['key']=='gpu_clock_ranges']
    assert len(clocks)==1 and clocks[0]['field']=='gpu_clock_ranges' and clocks[0]['type']=='string' and clocks[0]['default']=='' and clocks[0]['parser']=='gui'
    current['options'].remove(clocks[0]);assert current==old
    assert hashlib.sha256((root/'build_gui_local/ecm_gui.exe').read_bytes()).hexdigest()==baseline['legacy_sha256']
    result=dict(passed=True,byte_identical=len(checked),legacy_exe=baseline['legacy_sha256'],schema_existing_projection_unchanged=True,
                approved_exception='Add gui_v2_sidecar and empty-default gui.gpu_clock_ranges; user approved TOML clock drafts and shared worker entry ownership separately')
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)); print(json.dumps(result))
if __name__=='__main__': main()
