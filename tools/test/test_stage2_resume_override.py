"""Exercise real queue/checkpoint/publication code with seeded completed NF work.

The seed helper writes real storage formats; only remaining curves run on GPU.
No production save, queue, executable or result file is modified.
"""
import argparse,hashlib,json,subprocess,time
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
        ini=work/'ecm.toml';ini.write_text('config_version=4\n[shared]\nuser=""\ncomputer=""\nfinished_file="finished.txt"\n'
            '[stage2]\ndevice="gpu"\nb2_ratio=0\nstage2_save_dir="."\ncheckpoint_dir="progress"\n',encoding='utf-8')
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
    for schema in ("v1", "v2", "v3", "v4", "pending"):
        work, meta = seed(schema + "_with_progress", schema, done=4)
        line = task(work, 4, 10000)
        checkpoint = Path(meta["checkpoint"])
        before = checkpoint.read_bytes()
        text = execute(work, good=False)
        assert "predates self-contained outcomes" in text and "curve_start:" not in text
        assert checkpoint.read_bytes() == before and (work / "worktodo.txt").read_text() == line
        assert not (work / "results.json.txt").exists()
        checks.append(schema + "_retains_checkpoint")

    work, meta = seed("v5_shrink", "v5")
    task(work, 100)
    text = execute(work)
    assert "already completed=150" in text and "curve_start:" not in text
    finish(work, 150, 0, 10000)
    old = Path(meta["checkpoint"])
    assert not old.exists() and Path(str(old) + ".migrated").is_file()
    checks.append("v5_150_of_960_to_100")

    work, meta = seed("v5_adopt_nf", "sealed", done=4)
    task(work, 4, 10000)
    path = work / "results.json.txt"
    old = path.read_bytes()
    execute(work)
    assert path.read_bytes() == old and not Path(meta["checkpoint"]).exists()
    checks.append("v5_adopts_existing_nf")

    work, meta = seed("retry_migration", "v5", done=4)
    task(work, 4, 10000)
    old = Path(meta["checkpoint"])
    archive = Path(str(old) + ".migrated")
    archive.mkdir()
    text = execute(work, "archive_blocked", good=False)
    assert "cannot archive imported Stage2 checkpoint" in text and old.exists()
    assert (work / "progress/.ecm_ckpt_stage2_521_1000.dat").read_text().startswith("6 4 ")
    archive.rmdir()
    execute(work)
    assert archive.is_file() and not old.exists()
    finish(work, 4, 0, 10000)
    checks.append("retry_checkpoint_migration")

    if not a.skip_long_continue:
        work, _ = seed("v5_continue", "v5")
        task(work, 300)
        started = time.monotonic()
        text = execute(work)
        finish(work, 300, 150, 20000)
        assert "record=151 " in text and "record=300 " in text
        checks.append("v5_150_of_960_to_300_with_new_B2")
        print("PASS continued 150 GPU curves; seconds=" + str(round(time.monotonic() - started, 3)), flush=True)
    report = {"complete": True, "checks": checks,
              "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__=='__main__':main()
