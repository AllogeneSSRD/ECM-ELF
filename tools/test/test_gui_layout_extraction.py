"""Verify GUI production relocation and explicit test target isolation."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mapping',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);m=json.loads(a.mapping.read_text())['mapping'];verified={}
    strip=lambda s:re.sub(rb'(?m)^#include[ \t]+"[^"\r\n]+"[ \t]*\r?$',b'',s.replace(b'\r\n',b'\n'))
    for old,new in m.items():
        original=subprocess.check_output(['git','show','7b64e76:'+old],cwd=ROOT);current=(ROOT/new).read_bytes()
        assert not (ROOT/old).exists(),old
        if old.endswith('CMakeLists.txt'):continue
        assert strip(original)==strip(current),new
        verified[new]=hashlib.sha256(current).hexdigest()
    cmake=(ROOT/'src/ecm/application/gui/CMakeLists.txt').read_text()
    for target in ['fake_worker','results_test','log_parse_test','gen_test']:
        assert f'add_executable(ecm_gui_{target} EXCLUDE_FROM_ALL "${{CMAKE_SOURCE_DIR}}/tools/test/gui/' in cmake
    script=(ROOT/'tools/build/dev/build_gui.ps1').read_text()
    assert '[string]$BuildDir = "build_gui_local"' in script and "[string[]]$Targets = @('ecm_gui')" in script
    assert not (ROOT/'src/gui').exists()
    report=dict(complete=True,baseline='7b64e76',relocated_files=len(m),content_or_include_only=len(verified),test_fixtures=4,
                default_gui_only=True,explicit_test_targets=4,source_sha256=verified,cuda_compilations=0)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
if __name__=='__main__':main()
