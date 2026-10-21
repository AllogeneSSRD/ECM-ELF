"""Run unchanged legacy contracts and new v2 App contracts; keep raw output."""
import argparse,json,subprocess
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir();old=Path('build_gui_local').resolve();new=a.build.resolve()
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');tests={}
    for name,cmd in [
        ('selftest',[old/'ecm_gui.exe','--selftest','--config',cfg]),
        ('logs',[old/'ecm_gui_log_parse_test.exe']),('generator',[old/'ecm_gui_gen_test.exe']),
        ('results',[old/'ecm_gui_results_test.exe',out/'result-fixture']),('controller',[old/'ecm_gui_controller_test.exe',out/'controller-fixture']),
        ('v2-state',[new/'ecm_gui_v2_state_test.exe',out/'state-fixture']),('v2-app',[new/'ecm_gui_v2_app_test.exe',out/'app-fixture'])]:
        run=subprocess.run(list(map(str,cmd)),cwd=out,capture_output=True,timeout=120);(out/(name+'.log')).write_bytes(run.stdout+run.stderr);assert run.returncode==0,(name,run.stdout[-1500:],run.stderr[-1500:]);tests[name]=run.returncode
    result=dict(passed=True,tests=tests);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
