"""Exercise both directions of an Edwards whole-curve CPU/CUDA worktodo handoff."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from check_opencl_stage2_worker import records, run, sha, stop_after_first_start

# Edwards Stage1 text snapshots for sigma 17/18 on M127 and a known Stage2 hit.
POINTS = """METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA=17; B1=1000; N=0x7fffffffffffffffffffffffffffffff; X=0x43330fd37449fa33aa3a5615347a23b3; STATUS=POINT; CHECKSUM=257868734863184526;
METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA=18; B1=1000; N=0x7fffffffffffffffffffffffffffffff; X=0x47dfccef9193b37d81fb04ea0a8d1586; STATUS=POINT; CHECKSUM=15946409080047046804;
"""
HIT = "METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA=17; B1=20; N=0x138b7fffffffffffffffffffffffffffd8e9; X=0x95a30d0f970f418da1abb62058daa024829; STATUS=POINT; CHECKSUM=9714319387007657894;\n"


def workflow(folder, device):
    folder.mkdir()
    (folder / 'ecm.toml').write_text(
        'config_version = 3\n[shared]\nworktodo_file = "worktodo.txt"\n'
        'finished_file = "finished.txt"\nlog_file = "worker.log"\n'
        f'[stage2]\nsave_dir = "."\ngpu_device = {device}\nd = 210\ncuda.curve_concurrency = "serial"\n'
        'cpu.threads = 2\ncpu.backend = "gwpoly"\ntune.runtime_enabled = false\nresults_file = "results.jsonl"\n', encoding='utf-8')
    (folder / 'points.save').write_text(POINTS, encoding='utf-8')
    (folder / 'hit.save').write_text(HIT, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    cpu, cuda = args.cpu.resolve(), args.cuda.resolve()
    assert cpu.name == cuda.name == 'ecm_stage2.exe'
    assert subprocess.check_output([cpu,'--gui-queue-protocol'], text=True).strip() == 'ECM_QUEUE_OWNER=1'
    summary = {'cpu': sha(cpu), 'cuda': sha(cuda), 'device': args.device}
    for name, first, second in [('cpu_to_cuda',cpu,cuda), ('cuda_to_cpu',cuda,cpu)]:
        case = folder / name
        workflow(case, args.device)
        task = 'ECMSTAGE2=1,2,127,-1,"points.save",40000,0,2'
        queue = case / 'worktodo.txt'
        queue.write_text(task+'\n', encoding='utf-8')
        stop_after_first_start(first, case)
        assert task in queue.read_text() and not (case/'finished.txt').exists()
        checkpoint, = case.rglob('.ecm_ckpt_stage2_*.dat')
        shutil.copyfile(checkpoint, case/'stopped.snapshot')
        assert records(case)[0]['Edwards']['sigma'] == 17
        log = run(second, case, ['--gui-events','--log-level','batches'], label='resume')
        assert 'completed=1 remaining=1' in log and 'curve_start: 2/2 record=2 sigma=18' in log
        assert not checkpoint.exists() and task not in queue.read_text()
        assert (case/'finished.txt').read_text().count(task)==1
        assert [row['Edwards']['sigma'] for row in records(case)] == [17,18]
        public = [json.loads(line) for line in (case/'results.json.txt').read_text().splitlines() if line]
        assert len(public)==1 and public[0]['status']=='NF' and public[0]['curves']==2
        summary[name] = {'checkpoint':sha(case/'stopped.snapshot'), 'curves':2}
    case = folder/'cpu_factor'
    workflow(case,args.device)
    task = 'ECMSTAGE2N=N/A,"hit.save",40000,0,1'
    stage1 = 'ECMSTAGE1=1,2,127,-1,1000,1'
    (case/'worktodo.txt').write_text(task+'\n'+stage1+'\n',encoding='utf-8')
    log = run(cpu,case,['--gui-events','--log-level','batches'])
    assert 'CPU Edwards Stage2 backend=gwpoly threads=2' in log
    assert (case/'worktodo.txt').read_text().strip()==stage1
    factor = json.loads((case/'results.json.txt').read_text())
    assert factor['stage']==2 and factor['factors']==['10007']
    summary['factor']=factor['factors']
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
