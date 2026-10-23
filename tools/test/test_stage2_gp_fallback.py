"""Stage2 optional decomposition faults preserve real saved-X hits and receipts.

Saved-X fixtures exercise the CPU shortcut, not GPU Stage2 arithmetic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('exe','fixture','gp','output'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);passed=[]
    for mode in ('real','missing','timeout','protocol','exit','flood','factor_invalid','probable','partial','blocked_log'):
        work=root/mode;work.mkdir();env=os.environ.copy();env['ECM_TEST_GP_MODE']=mode
        if mode=='probable':exponent=254;raw=(1<<127)-1
        elif mode=='partial':exponent=251;raw=178230287214063289511*61676882198695257501367
        else:exponent=21;raw=7**2*127
        n=(1<<exponent)-1;assert n%raw==0
        save=work/'fixture.save'
        save.write_text(f'METHOD=ECM; PARAM=0; SIGMA=6; B1=2; N=(2^{exponent}-1); X=0x{raw:x}; CHECKSUM={2*6*n*raw%4294967291};\n',encoding='utf-8')
        gp=a.gp.resolve() if mode in ('real','blocked_log') else work/'missing_gp.exe' if mode in ('missing','probable','partial') else a.fixture.resolve()
        ini=work/'ecm.toml';ini.write_text(f'[shared]\nuser=""\ncomputer=""\n[stage2]\nstage2_factorize_hits=true\nstage2_gp={json.dumps(str(gp))}\nstage2_factor_timeout=1\n',encoding='utf-8')
        if mode=='blocked_log':(work/'factor_details').write_text('Deliberate non-directory diagnostic path',encoding='utf-8')
        results=work/'result.jsonl';log=work/'curve.log'
        queue=work/'worktodo.txt';queue.write_text(f'ECMSTAGE2=1,2,{exponent},-1,"fixture.save",3,0,1\n',encoding='utf-8')
        cmd=[str(a.exe.resolve()),'--config',str(ini),'--worktodo',str(queue),'--once','--device','1',
             '--short-calibration','0','--tune-runtime','0','--d','210','--carrier-exponent','0',
             '--results',str(results),'--log',str(log)]
        # Watch the actual GP child boundary: the raw submission must already exist.
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=work,env=env)
        if mode=='timeout':
            deadline=time.monotonic()+10;durable_before_done=False
            while time.monotonic()<deadline and process.poll() is None:
                submission=work/'results.json.txt'
                if submission.exists() and str(raw) in submission.read_text(encoding='utf-8'):
                    durable_before_done=True;break
                time.sleep(.01)
            assert durable_before_done,'No durable factor while optional GP was waiting'
        stdout,stderr=process.communicate(timeout=25);(work/'driver.log').write_bytes(stdout+stderr)
        assert process.returncode==0,(mode,stdout,stderr)
        records=[json.loads(line) for line in results.read_text(encoding='utf-8').splitlines()]
        assert len(records)==1,records
        row=records[0];assert row['factors']==[str(raw)],row
        details=row.get('stage2-engine',row);assert details['status']=='factor_in_saved_X',row
        analysis=details['factor_analysis'][0]
        assert analysis['raw']==str(raw)
        reconstructed=int(analysis['remaining_cofactor'])
        for q in analysis['prime_powers']:reconstructed*=int(q['factor'])**q['multiplicity']
        assert reconstructed==raw,analysis
        if mode in ('real','blocked_log'):assert analysis['backend']=='gp' and details['factorization_complete'],analysis
        elif mode=='factor_invalid':assert analysis['backend']=='gp' and not details['factorization_complete'],analysis
        elif mode=='probable':
            assert not details['factorization_complete'] and details['prime_factors']==[],details
            assert analysis['status']=='partial' and analysis['reason']=='primality_unproven',analysis
            assert len(analysis['prime_powers'])==1 and not analysis['prime_powers'][0]['proven'],analysis
        elif mode=='partial':assert not details['factorization_complete'] and int(analysis['remaining_cofactor'])>1,analysis
        else:
            assert analysis['backend']=='native' and details['factorization_complete'],details
            assert {int(q['factor']):q['multiplicity'] for q in analysis['prime_powers']}=={7:2,127:1}
        assert all(q['proven'] for q in analysis['prime_powers'] if q['factor'] in details['prime_factors'])
        # Replay uses the existing receipt/checkpoint; diagnosis must not duplicate hits.
        before=results.read_bytes();repeat=subprocess.run(cmd,capture_output=True,cwd=work,env=env,timeout=25)
        assert repeat.returncode==0 and results.read_bytes()==before,(mode,repeat.stdout,repeat.stderr)
        assert str(raw) in (work/'results.json.txt').read_text(encoding='utf-8')
        passed.append(dict(case=mode,returncode=0,receipt_preserved=True,analysis=analysis));print('passed',mode,flush=True)
    summary=dict(passed=len(passed),failed=0,checks=passed,scope='real Stage2 frontend saved-X shortcut; no GPU arithmetic',
                 exe_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=len(passed),failed=0)))


if __name__=='__main__':main()
