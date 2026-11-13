"""Exercise real queue/checkpoint/publication code with seeded completed NF work.

The seed helper writes real storage formats; only remaining curves run on GPU.
No production save, queue, executable or result file is modified.
"""
import argparse,ctypes,hashlib,json,re,subprocess,time
import shutil
from ctypes import wintypes
from pathlib import Path
from test_result_output_consistency import crc_valid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('exe','fixture','save','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--skip-long-continue',action='store_true',help='run changed recovery cases without repeating the 150-curve GPU gate')
    a=p.parse_args()
    exe,fixture,save,out=(getattr(a,key).resolve() for key in ('exe','fixture','save','output'))
    out.mkdir(parents=True,exist_ok=False);checks=[]
    def seed(name,schema='v3',done=150):
        work=out/name
        result=subprocess.run(list(map(str,[fixture,work,save,schema,done,960,10000])),capture_output=True,text=True,check=True)
        metadata=json.loads(result.stdout)
        ini=work/'ecm.toml';ini.write_text('[shared]\nuser=""\ncomputer=""\nb2_ratio=0\n[stage2]\n'
            'stage2_save_dir="."\nstage2_finished="finished.txt"\nstage2_progress_dir="progress"\n',encoding='utf-8')
        return work,metadata
    def task(work,count,b2=20000):
        line=f'ECMSTAGE2=1,2,521,-1,"e_m521_1000.save",{b2},0,{count}\n'
        (work/'worktodo.txt').write_text(line);return line
    def command(work):
        return list(map(str,[exe,'--config',work/'ecm.toml','--worktodo',work/'worktodo.txt','--once',
            '--device',a.device,'--results',work/'stage2_results.jsonl','--submission',work/'results.json.txt',
            '--short-calibration',0,'--tune-runtime',0,'--auto-carrier',0,'--log-level',1]))
    def execute(work,label='run',good=True):
        result=subprocess.run(command(work),cwd=work,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        text=result.stdout+result.stderr;(work/(label+'.log')).write_text(text,encoding='utf-8')
        assert (result.returncode==0)==good,(work,result.returncode,text[-3000:])
        return text
    def finish(work,done,new_curves,last_b2):
        assert not (work/'worktodo.txt').read_text().strip()
        assert not list((work/'progress').glob('*.dat'))
        rows=[json.loads(line) for line in (work/'stage2_results.jsonl').read_text().splitlines()]
        reports=[json.loads(line) for line in (work/'results.json.txt').read_text().splitlines()]
        assert len(rows)==done and len(reports)==1 and reports[0]['curves']==done and reports[0]['b2']==last_b2
        assert all(map(crc_valid,rows+reports))
        assert sum(row.get('stage2-engine',{}).get('seeded_resume_fixture',False) for row in rows)==done-new_curves
        assert all(row['b2']==20000 for row in rows[-new_curves:]) if new_curves else all(row['b2']==10000 for row in rows)
        # V2 publication ownership must survive the new checkpoint identity.
        assert 'ECMSTAGE2=' in (work/'finished.txt').read_text()
    for schema in ('v3','v2','v1'):
        work,_=seed(schema+'_shrink',schema);task(work,100)
        text=execute(work);assert 'already completed=150' in text and 'curve_start:' not in text and 'remaining=0' in text
        finish(work,150,0,10000);checks.append(schema+'_150_of_960_to_100')
    work,_=seed('pending_ack','pending',149);task(work,100)
    text=execute(work);assert 'legacy_archived=1' in text and 'already completed=150' in text
    finish(work,150,0,10000);checks.append('pending_receipt_before_shrink')
    work,_=seed('unconfirmed','unconfirmed');task(work,100)
    text=execute(work);assert 'legacy_archived=1' in text and 'curve_start:' not in text
    finish(work,150,0,10000);checks.append('uncommitted_attempt_abandoned_before_shrink')
    work,_=seed('torn_internal','torn_internal');task(work,100)
    text=execute(work);assert 'already completed=151' in text and 'curve_start:' not in text
    report,=map(json.loads,(work/'results.json.txt').read_text().splitlines())
    assert report['curves']==151 and crc_valid(report)
    checks.append('ledger_recovers_torn_jsonl')
    work,_=seed('missing_evidence','v3');line=task(work,100)
    shutil.move(work/'data',work/'evidence')
    before={path:path.read_bytes() for path in (work/'progress').glob('*.dat')}
    text=execute(work,good=False)
    assert 'cannot verify legacy Stage2 outcome' in text and (work/'worktodo.txt').read_text()==line
    assert {path:path.read_bytes() for path in (work/'progress').glob('*.dat')}==before
    checks.append('missing_evidence_retains_original')
    if not a.skip_long_continue:
        work,_=seed('v3_continue');task(work,300)
        started=time.monotonic();text=execute(work);finish(work,300,150,20000)
        assert 'record=151 ' in text and 'record=300 ' in text
        checks.append('v3_150_of_960_to_300_with_new_B2')
        print('PASS continued 150 GPU curves; seconds='+str(round(time.monotonic()-started,3)),flush=True)
    # Stop after one curve, then change the policy again. This verifies the V2
    # import is persisted as schema 5 before resuming arithmetic.
    work,metadata=seed('v2_resume_stop','v2');task(work,300)
    class Security(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('descriptor',wintypes.LPVOID),('inherit',wintypes.BOOL)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateEventW.argtypes=[ctypes.POINTER(Security),wintypes.BOOL,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateEventW.restype=wintypes.HANDLE;kernel.SetEvent.argtypes=[wintypes.HANDLE];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    security=Security(ctypes.sizeof(Security),None,True);event=kernel.CreateEventW(ctypes.byref(security),True,False,None)
    assert event
    log=work/'stop.log'
    with log.open('wb') as stream:
        process=subprocess.Popen(command(work)+['--stage2-stop-handle',str(event)],cwd=work,stdout=stream,stderr=stream,
            close_fds=False,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+180
            while 'curve_start:' not in log.read_text(encoding='utf-8',errors='replace'):
                assert process.poll() is None,log.read_text(errors='replace')[-3000:]
                if time.monotonic()>deadline:raise TimeoutError('no resumed curve start')
                time.sleep(.01)
            assert kernel.SetEvent(event);assert process.wait(timeout=180)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
            kernel.CloseHandle(event)
    checkpoints=list((work/'progress').glob('*.dat'));assert len(checkpoints)==1
    state=checkpoints[0].read_text();assert state.startswith('5 151 300 ') and metadata['publication_task'] in state
    assert checkpoints[0].name=='.ecm_ckpt_stage2_521_1000.dat'
    old=Path(metadata['checkpoint'])
    assert not old.exists() and Path(str(old)+'.migrated').exists()
    shutil.move(work/'data',work/'retired-ledger')
    (work/'stage2_results.jsonl').unlink()
    task(work,100,30000);text=execute(work,'shrink_again')
    assert 'already completed=151' in text and 'curve_start:' not in text
    report,=map(json.loads,(work/'results.json.txt').read_text().splitlines())
    assert report['curves']==151 and report['b2']==20000 and crc_valid(report)
    assert not checkpoints[0].exists()
    checks.append('v2_import_stop_then_remove_legacy_evidence')
    work,_=seed('v4_gap','v4',done=2);task(work,4)
    text=execute(work)
    assert re.findall(r'^curve_start: .* record=(\d+) ',text,re.M)==['3']
    report,=map(json.loads,(work/'results.json.txt').read_text().splitlines())
    assert report['curves']==4 and crc_valid(report)
    checks.append('v4_out_of_order_gap')
    work,meta=seed('v5_old_name','v5',done=0);task(work,1)
    text=execute(work)
    assert '.ecm_ckpt_stage2_521_1000.dat' in text
    assert not Path(meta['checkpoint']).exists() and Path(meta['checkpoint']+'.migrated').exists()
    checks.append('v5_old_sha_filename')
    work,meta=seed('sealed_history','sealed',done=4);task(work,4,10000)
    published=(work/'results.json.txt').read_bytes()
    execute(work)
    assert (work/'results.json.txt').read_bytes()==published
    assert not Path(meta['checkpoint']).exists()
    checks.append('sealed_nf_not_republished')
    work,meta=seed('migration_retry','v3',done=4);task(work,4,10000)
    archive=Path(meta['checkpoint']+'.migrated')
    archive.mkdir()
    text=execute(work,'archive_blocked',good=False)
    assert 'cannot archive imported Stage2 checkpoint' in text
    assert Path(meta['checkpoint']).exists()
    assert (work/'progress/.ecm_ckpt_stage2_521_1000.dat').read_text().startswith('5 4 ')
    archive.rmdir()
    shutil.move(work/'data',work/'retired-ledger')
    (work/'stage2_results.jsonl').unlink()
    execute(work)
    row,=map(json.loads,(work/'results.json.txt').read_text().splitlines())
    assert row['curves']==4 and crc_valid(row)
    assert archive.is_file() and not Path(meta['checkpoint']).exists()
    checks.append('resume_interrupted_migration_from_new_checkpoint')
    work,meta=seed('pending_without_ledger','pending',done=0);task(work,1,10000)
    row_path=work/'stage2_results.jsonl'
    row=json.loads(row_path.read_text())
    point=(work/'e_m521_1000.save').read_bytes().split(b'\n',1)[0]
    record_hash=14695981039346656037
    for byte in point:
        record_hash=((record_hash^byte)*1099511628211) % (1 << 64)
    row['stage2-engine'].update(record=1,record_hash=str(record_hash))
    row_path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    shutil.move(work/'data',work/'retired-ledger')
    text=execute(work,good=False)
    assert 'cannot verify legacy Stage2 outcome' in text and Path(meta['checkpoint']).exists()
    assert 'curve_start:' not in text
    checks.append('confirmed_pending_with_missing_ledger_preserved')
    report={'complete':True,'checks':checks,'gpu_curves':3 if a.skip_long_continue else 153,'default_seeded_prior_nf':150,
        'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
