"""Offline Stage1 output, batch checkpoint and CPU/GPU endpoint regression gates."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import time
import zlib
from stage1_checkpoint_decode import batch, progressed

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('oracle', ROOT/'tools/test/test_edwards_gpu_pipeline.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)




def verify_result(row):
    assert row['program']['name'] == 'ECM-ELF'
    assert re.fullmatch(r'\d+\.\d+\.\d+', row['program']['version'])
    known = ','.join(row.get('known-factors', []))
    if 'exponent' in row:
        target = str(row['exponent'])
    elif 'k' in row:
        n = (int(row['k'])*row['b']**row['n']+row['c'])//math.prod(map(int, row.get('known-factors', [])))
        target = 'N='+str(n)
    else:
        target = 'N='+str(row['N'])
    sigma = 'E'+str(row['Edwards']['sigma']) if 'Edwards' in row else str(row['sigma'])
    canonical = ';'.join([target, 'ECM', ','.join(row.get('factors', [])), known, str(row['b1']),
                          str(row['b2']) if row.get('b2', 0) > row['b1'] else '', sigma, str(row['fft-length']), 'invariant:0',
                          'ECM-ELF', row['program']['version'], '', '', 'Windows', 'x86_64', row['timestamp']])
    assert row['checksum'] == dict(version=1, checksum=f'{zlib.crc32(canonical.encode()):08X}')
    assert row['stage1-engine']['coverage'] == 'B1'


def identity_case(exe, out):
    work=out/'private_identity';work.mkdir(parents=True)
    user,computer,aid=(secrets.token_hex(12),secrets.token_hex(12),secrets.token_hex(16))
    task=f'ECM2={aid},1,2,130,-1,20,10000,2,20260922,"3"\n'
    queue=work/'worktodo.txt';queue.write_text('[Worker #2]\n'+task)
    ini=work/'ecm.ini';ini.write_text(f'curve=ed\ntarget=cpu\nimpl=scalar\nstage1_threads=1\n'
        f'worktodo={queue}\nfinished={work/"finished.txt"}\ntmp_dir={work}\nlog_file=\n'
        f'user={secrets.token_hex(12)}\ncomputer={secrets.token_hex(12)}\n[Worker #2]\nuser={user}\ncomputer={computer}\n')
    p=subprocess.run([str(exe),'-ini',str(ini),'--worker','2'],cwd=work,capture_output=True,timeout=120)
    (work/'run.log').write_bytes(p.stdout+p.stderr)
    assert p.returncode==0, 'identity case failed; inspect private evidence log'
    inside=[json.loads(v) for v in (work/'stage1_results_2.jsonl').read_text().splitlines()]
    outside=[json.loads(v) for v in (work/'results.json_2.txt').read_text().splitlines()]
    assert len(inside)==2 and sum(v['curves'] for v in outside)==sum(v['status']=='F' for v in inside)
    assert all(v['status']=='F' and v['stage']==1 for v in outside)
    for row in inside+outside:
        assert row['exponent']==130 and row['known-factors']==['3']
        assert row['user']==user and row['computer']==computer and row['aid']==aid
    for row in inside:verify_result(row)
    assert '.e_s1_m130_20.ckpt' in p.stdout.decode('utf-8','replace')
    assert task not in queue.read_text()
    return dict(complete=True,curves=2,worker=2,original_exponent=130,cofactor_bits=129)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--cpu-stage2', type=Path, help='also verify the shared submission ledger across stages')
    parser.add_argument('--gpu-algorithm',choices=('ladder','resident','prac'),default='ladder')
    parser.add_argument('--gpu0-only',action='store_true')
    args = parser.parse_args()
    exe, out = args.stage1.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('use a fresh evidence directory')
    report = dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), gpu_algorithm=args.gpu_algorithm, checks=[], complete=False)
    env = dict(os.environ)
    env.pop('ECM_EDWARDS_STAGE1_ALGO', None)
    env.pop('ECM_EDWARDS_DBL_ALGO', None)
    env['ECM_GPU_STAGE1_ALGO']=args.gpu_algorithm

    def passed(name):
        report['checks'].append(name)
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print('PASS', name, flush=True)

    def call(name, cmd, cwd, text=None, good=True, extra_env=None):
        p = subprocess.run(list(map(str, cmd)), input=text, text=True, encoding='utf-8', errors='replace',
                           capture_output=True, cwd=cwd, env=env | (extra_env or {}), timeout=240)
        log = p.stdout+p.stderr
        (out/(name+'.log')).write_text(log, encoding='utf-8')
        assert (p.returncode == 0) == good, (name, p.returncode, log[-2500:])
        return log

    def run(name, model='ed', target='cpu', impl='scalar', param=0, n=(1 << 347)-1,
            expr=None, b1=1000, count=4, extra_env=None):
        work = out/name
        work.mkdir()
        ini = work/'ecm.ini'
        ini.write_text('user=\ncomputer=\n')
        cmd = [exe, '--curve', model, '--target', target,
               '-d', args.device, '-sigma', 20260922, '-gpucurves', count,
               '--tmp-dir', work, '-ini', ini, '-save', work/'points.save', b1, b1]
        if target == 'gpu' and model == 'mont':
            cmd[1:1] = ['--gpu-param', param]
        if target == 'cpu':
            cmd[1:1] = ['--impl', impl, '--stage1-threads', 2]
        log = call(name, cmd, work, (expr or str(n))+'\n', extra_env=extra_env)
        internal = [json.loads(v) for v in (work/'stage1_results_1.jsonl').read_text().splitlines()] if (work/'stage1_results_1.jsonl').exists() else [json.loads(v) for v in (work/'stage1_results.jsonl').read_text().splitlines()]
        public_path = work/'results_1.json.txt' if (work/'results_1.json.txt').exists() else work/'results.json.txt'
        public = [json.loads(v) for v in public_path.read_text().splitlines()]
        assert len(internal) == count
        for row in internal:
            verify_result(row)
        assert sum(v['curves'] for v in public) == sum(v['status']=='F' for v in internal)
        assert len(list(Path(str(public_path)+'.ecm-ledger').glob('*.curve'))) == len(public)
        assert not list(Path(str(public_path)+'.ecm-ledger').glob('*.batch'))
        for row in public:
            assert row['fft-length'] > 0 and 'checksum' not in row and 'security-code' not in row
            assert row['status'] == 'F' and row['stage'] == 1 and 'b2' not in row
            assert ('Edwards' in row) == (model == 'ed')
        assert '100.0%' in log
        assert not list(work.glob('*.ckpt'))
        assert not list(work.glob('*.tmp'))
        assert (work/'points.save').exists() and len((work/'points.save').read_text().splitlines())==count
        exponent = int(re.search(r'\^(\d+)',expr)[1]) if expr else (n.bit_length() if n & (n+1) == 0 else None)
        label='m'+str(exponent) if exponent else 'bit'+str(n.bit_length())
        assert f'.{"e" if model=="ed" else "m"+str(param)}_s1_{label}_{b1}.ckpt' in log
        if model == 'ed':
            oracle.records(work/'points.save')
        passed(name)
        return work, cmd, internal

    for name, model, target, impl, param in [
            ('ed_scalar', 'ed', 'cpu', 'scalar', 0), ('ed_simd', 'ed', 'cpu', 'simd', 0),
            ('ed_gpu', 'ed', 'gpu', 'auto', 0), ('mont_scalar', 'mont', 'cpu', 'scalar', 0),
            ('mont_simd', 'mont', 'cpu', 'simd', 0), ('mont_gpu0', 'mont', 'gpu', 'auto', 0),
            ('mont_gpu2', 'mont', 'gpu', 'auto', 2), ('mont_gpu3', 'mont', 'gpu', 'auto', 3)]:
        if args.gpu0_only and (model,target,param)!=('mont','gpu',0):continue
        run(name, model, target, impl, param, n=(1 << (991 if param >= 2 else 347))-1,
            count=16 if impl == 'simd' else 4)
    if not args.gpu0_only:
        run('ed_binary', target='gpu', extra_env={'ECM_EDWARDS_STAGE1_ALGO': 'binary'})
        run('ed_factor', n=1009*((1 << 127)-1))
        run('generic_integer', n=(1 << 347)-5)
        run('wide_generic_integer', n=(1 << 8191)-19, b1=20, count=1)
        run('kbnc_expression', n=3*(1 << 127)+1, expr='3*2^127+1')

        if args.cpu_stage2:
            stage2=args.cpu_stage2.resolve()
            for name in ('ed_scalar','ed_factor'):
                work=out/name;submission=work/'results.json.txt'
                before=submission.read_bytes()
                cmd=[stage2,'--save',work/'points.save','--b2',40000,'--curves',4,
                     '--results',work/'stage2_results.jsonl','--submission',submission,'--ini',work/'ecm.ini']
                call(name+'_stage2',cmd,work)
                rows=[json.loads(v) for v in submission.read_text().splitlines()]
                if name=='ed_scalar':
                    assert not before and sum(v['curves'] for v in rows)==4
                    assert all(v['status']=='NF' and v['b2']==40000 for v in rows)
                else:
                    assert len(rows)==4 and submission.read_bytes()==before
                completed=submission.read_bytes()
                replay=[exe,'--curve','ed','--target','cpu','--impl','scalar','--stage1-threads',2,
                        '-sigma',20260922,'-gpucurves',4,'--tmp-dir',work,'-ini',work/'ecm.ini',
                        '-save',work/'points.save',1000,1000]
                n=(1<<347)-1 if name=='ed_scalar' else 1009*((1<<127)-1)
                call(name+'_stage1_replay',replay,work,str(n)+'\n')
                assert submission.read_bytes()==completed
                call(name+'_stage2_replay',cmd,work)
                assert submission.read_bytes()==completed
                passed(name+'_cross_stage_submission')
                if name=='ed_scalar':
                    # Simulate interrupted Stage2 publication on a working copy,
                    # preserving the completed source ledger and its export.
                    pending=work/'pending.json.txt';ledger=Path(str(pending)+'.ecm-ledger')
                    shutil.copytree(Path(str(submission)+'.ecm-ledger'),ledger)
                    for sealed in ledger.glob('*.batch'):sealed.unlink()
                    ini=work/'pending.ini';ini.write_text('user=\ncomputer=\nstage1_submission_file='+str(pending)+'\n')
                    replay[replay.index('-ini')+1]=ini
                    call('stage1_with_pending_stage2_nf',replay,work,str(n)+'\n')
                    assert pending.read_bytes()==b'' and not list(ledger.glob('*.batch'))
                    assert len(list(ledger.glob('*.curve')))==4
                    call('pending_stage2_export',[stage2,'--export-results','--submission',pending,'--ini',ini],work)
                    assert sum(json.loads(v)['curves'] for v in pending.read_text().splitlines())==4
                    passed('stage1_preserves_pending_stage2_nf')

        for model in ('ed', 'mont'):
            name='publication_retry_'+model
            work=out/name;work.mkdir()
            queue=work/'worktodo.txt';task='ECM2=1,2,347,-1,1000,100000,4,20260922\n';queue.write_text(task)
            blocked=work/'blocked';blocked.write_text('preserved publication failure fixture\n')
            ini=work/'ecm.ini';ini.write_text(f'curve={model}\ntarget=cpu\nimpl=scalar\nstage1_threads=2\nworktodo={queue}\n'
                f'finished={work/"finished.txt"}\ntmp_dir={work}\nlog_file=\nstage1_submission_file={blocked/"results.json.txt"}\n')
            call(name+'_failed',[exe,'-ini',ini],work,good=False)
            assert queue.read_text()==task
            primary=work/f'.{"e" if model=="ed" else "m0"}_s1_m347_1000.ckpt'
            _,saved=batch(primary)
            assert len(saved)==4 and all(payload for _,payload in saved)
            if model=='ed':
                assert all(payload.startswith(b'ECM_EDWARDS_DONE_V1') for _,payload in saved)
            blocked.rename(work/'blocked_evidence')
            call(name+'_retry',[exe,'-ini',ini],work)
            assert not queue.read_text().strip() and not primary.exists()
            assert len((work/'stage1_results.jsonl').read_text().splitlines())==4
            assert (blocked/'results.json.txt').read_bytes()==b''
            save=next(p for p in work.glob('*.save') if '.bak' not in p.name)
            assert len(save.read_text().splitlines())==4, 'save replay duplicated curves'
            passed(name)

    if not args.gpu0_only:
        identity_case(exe,out)
        passed('private_identity_known_factors_worker2')

    class Security(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', wintypes.LPVOID), ('inherit', wintypes.BOOL)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(Security), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    for model, target, param, impl in [('ed', 'cpu', 0, 'scalar'), ('ed', 'cpu', 0, 'simd'),
                                     ('ed', 'gpu', 0, 'auto'), ('mont', 'cpu', 0, 'scalar'),
                                     ('mont', 'cpu', 0, 'simd'), ('mont', 'gpu', 0, 'auto'),
                                     ('mont', 'gpu', 2, 'auto'), ('mont', 'gpu', 3, 'auto')]:
        if args.gpu0_only and (model,target,param)!=('mont','gpu',0):continue
        name = f'restart_{model}_{target}{param}_{impl}'
        work = out/name
        work.mkdir()
        bits = 991 if param >= 2 else 347
        task = f'ECM2=1,2,{bits},-1,100000,1000000,8,20260922\n'
        queue = work/'worktodo.txt'
        queue.write_text(task)
        ini = work/'ecm.ini'
        ini.write_text(f'curve={model}\ntarget={target}\nimpl={impl}\ngpu_param={param}\ndevice={args.device}\n'
                       f'ckpt_seconds=0.001\nckpt_backups=2\nstage1_threads=1\nworktodo={queue}\nfinished={work/"finished.txt"}\n'
                       f'tmp_dir={work}\nlog_file=\np95_worktodo_path={work/"absent_prime95"/"worktodo.txt"}\n')
        # Only Edwards and param2 ignore transfer. Do not hand off other cases.
        if model != 'ed' and param != 2:
            ini.write_text(ini.read_text().replace(f'p95_worktodo_path={work/"absent_prime95"/"worktodo.txt"}', 'p95_worktodo_path='))
        sa = Security(ctypes.sizeof(Security), None, True)
        event = kernel.CreateEventW(ctypes.byref(sa), True, False, None)
        cmd = [exe, '-ini', ini]
        with (out/(name+'_stop.log')).open('wb') as log:
            p = subprocess.Popen(list(map(str, cmd+['--stage1-stop-handle', event])), stdout=log, stderr=log,
                                 cwd=work, env=env, close_fds=False, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic()+90
                while time.monotonic() < deadline:
                    assert p.poll() is None, (name, 'exited before stop')
                    candidates = [v for v in work.glob('*.ckpt') if '.bak' not in v.name]
                    try:
                        if candidates and progressed(batch(candidates[0])[1]):
                            break
                    except (PermissionError, FileNotFoundError):
                        pass  # Atomic replacement can briefly deny a concurrent reader.
                    time.sleep(.05)
                else:
                    raise AssertionError('no arithmetic checkpoint')
                competing=call(name+'_exclusive',cmd,work,good=False)
                assert 'another worker owns' in competing
                assert kernel.SetEvent(event)
                assert p.wait(timeout=120) != 0
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
                kernel.CloseHandle(event)
        assert queue.read_text() == task and not (work/'absent_prime95').exists()
        primary = work/f'.{"e" if model == "ed" else "m"+str(param)}_s1_m{bits}_100000.ckpt'
        _, rows = batch(primary)
        assert len(rows) == 8 and all(not sigma or sigma == 20260922+i for i, (sigma, _) in enumerate(rows))
        original = primary.read_bytes()
        (out/(name+'_checkpoint_before_fault.bin')).write_bytes(original)
        # A colliding short filename must never permit a different batch count.
        queue.write_text(task.replace(',8,20260922', ',7,20260922'))
        call(name+'_reject_count', cmd, work, good=False)
        assert primary.read_bytes() == original
        queue.write_text(task.replace('20260922','20260923'))
        call(name+'_reject_sigma',cmd,work,good=False)
        assert primary.read_bytes()==original
        queue.write_text(task.replace('ECM2=1,2,','ECM2=3,2,',1))
        call(name+'_reject_target',cmd,work,good=False)
        assert primary.read_bytes()==original
        queue.write_text(task)
        ini.write_text(ini.read_text().replace('ckpt_seconds=0.001', 'ckpt_seconds=0'))
        backup = primary.with_name(primary.stem+'.bak'+primary.suffix)
        assert backup.exists()
        primary.write_bytes(original[:-4]+b'BAD\n')
        (out/(name+'_checkpoint_fault.bin')).write_bytes(primary.read_bytes())
        log = call(name+'_resume', cmd, work)
        assert not queue.read_text().strip() and not list(work.glob('*.ckpt'))
        assert not (work/'absent_prime95').exists()
        if model == 'ed':
            oracle.records(next(work.glob('e_*.save')))
            assert 'ignored' in log
        passed(name)
    report['complete'] = True
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
