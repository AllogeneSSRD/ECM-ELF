"""Real Stage1 V5/V1/Edwards legacy resume and automatic save-name regression.

Consumes frozen mid-arithmetic checkpoints from test_stage1_usability.py. Uses
fresh directories; never changes the production save/checkpoint evidence.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
from stage1_checkpoint_decode import batch


def fnv(data, seed=1469598103934665603):
    for x in data:seed=((seed^x)*1099511628211)&((1<<64)-1)
    return seed


def v1(identity, rows):
    body=b'ECM_STAGE1_BATCH_V1\n'+identity.encode()+b'\n'+str(len(rows)).encode()+b'\n'
    for sigma,payload in rows:body+=f'{sigma} {len(payload)}\n'.encode()+payload+b'\n'
    return body+f'CHECKSUM {fnv(body)}\n'.encode()


def points(save):
    rows=[]
    for line in save.read_text().splitlines():
        fields=dict(re.findall(r'(\w+)\s*=\s*([^;]+)',line))
        rows.append((fields.get('METHOD'),fields.get('PARAM'),fields['SIGMA'],fields['B1'],fields['X'],fields.get('STATUS')))
    return sorted(rows)


def legacy_edwards(payload, b1):
    assert payload[:8]==b'ECMEDB3\0'
    version,w,d,count,done,flags=struct.unpack_from('<IIIIII',payload,8)
    assert version==3 and flags==0
    at=56
    def value():
        nonlocal at
        size=struct.unpack_from('<I',payload,at)[0];at+=4
        n=int.from_bytes(payload[at:at+size],'little');at+=size
        return n
    n=value()
    sieve=bytearray(b'\1')*(b1+1);sieve[:2]=b'\0\0'
    for p in range(2,math.isqrt(b1)+1):
        if sieve[p]:sieve[p*p::p]=b'\0'*((b1-p*p)//p+1)
    scalar=48
    for p in range(2,b1+1):
        if sieve[p]:
            power=p
            while power<=b1//p:power*=p
            scalar*=power
    text=f'ECM_GPU_EDWARDS_NAF_V2\n{w} {d}\n{n:x}\n{scalar:x}\n{b1} {count} {done}\n'
    for _ in range(count):
        sigma,hit=struct.unpack_from('<QI',payload,at);at+=12
        text+=f'{sigma} {hit} '+' '.join(f'{value():x}' for _ in range(10))+'\n'
    assert at==len(payload)-8
    body=text.encode();return body+f'CHECKSUM {fnv(body,14695981039346656037)}\n'.encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    args=p.parse_args();exe=args.stage1.resolve();fixtures=args.fixtures.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ);env['ECM_GPU_STAGE1_ALGO']='ladder'
    for key in ('ECM_EDWARDS_STAGE1_ALGO','ECM_EDWARDS_DBL_ALGO'):env.pop(key,None)
    report=dict(complete=False,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),checks=[])

    def setup(name, model='mont', target='gpu', param=0, pattern='', task=None):
        work=out/name;work.mkdir();ini=work/'ecm.toml'
        ini.write_text(f'[shared]\nworktodo={json.dumps(str(work/"worktodo.txt"))}\nuser=""\ncomputer=""\n'
            f'[stage1]\ncurve="{model}"\ntarget="{target}"\nimpl="{"scalar" if target=="cpu" else "auto"}"\n'
            f'gpu_param={param}\ndevice={args.device}\nckpt_seconds=0\nckpt_backups=2\nexp_cache="off"\n'
            f'finished={json.dumps(str(work/"finished.txt"))}\ntmp_dir={json.dumps(str(work))}\n'
            'log_file=""\n'+(f'save_name_pattern={json.dumps(pattern)}\n' if pattern else ''))
        (work/'worktodo.txt').write_text(task or 'ECM2=1,2,347,-1,100000,1000000,8\n')
        return work,ini

    def run(name,work,ini,local_exe=None,good=True,empty=True):
        result=subprocess.run([str(local_exe or exe),'--config',str(ini)],cwd=work,env=env,capture_output=True,timeout=180)
        text=(result.stdout+result.stderr).decode('utf-8','replace');(out/(name+'.log')).write_text(text)
        assert (result.returncode==0)==good,(name,text[-2500:])
        if good and empty:assert not (work/'worktodo.txt').read_text().strip()
        report['checks'].append(name)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        return text

    mont_source=fixtures/'restart_mont_gpu0_auto_checkpoint_before_fault.bin'
    identity,rows=batch(mont_source);payload=rows[0][1];assert payload[:8]==struct.pack('<II',0x45555047,5)
    nh=identity.split('|')[0];n=int(nh,16)
    legacy_name=f'.ecm_ckpt_{n.bit_length()}_{nh[:8]}_{nh[-8:]}.dat'
    expected=points(fixtures/'restart_mont_gpu0_auto/m0_m347_1e5.save')
    for name,seed,backup in [('mont_v5',False,False),('mont_v5_seed_only',True,False),('mont_v5_backup',False,True),('mont_v1',False,False)]:
        work,ini=setup(name);old=work/legacy_name
        if name=='mont_v1':
            (work/'.m0_s1_m347_100000.ckpt').write_bytes(v1(identity,rows))
        else:
            old.write_bytes(payload)
            if seed:(work/'.m0_s1_m347_100000.ckpt').write_bytes(v1(identity,[(sigma,b'') for sigma,_ in rows]))
            if backup:
                old.with_name(old.stem+'.bak.dat').write_bytes(payload)
                damaged=bytearray(payload);damaged[-10]^=1;old.write_bytes(damaged)
        before=old.read_bytes() if old.exists() else None
        text=run(name,work,ini)
        if seed:
            # A validated seed-only batch keeps its own curve identities. The
            # unchanged legacy importer does not replace a pending container.
            assert 'Stage1 batch resume:' in text and 'legacy GPU checkpoint imported:' not in text
            assert old.read_bytes()==before
        else:
            assert 'Resuming from checkpoint:' in text
            if name!='mont_v1':assert 'legacy GPU checkpoint imported:' in text and old.read_bytes()==before
        assert points(work/'m0_m347_1e5.save')==expected,'legacy Mont endpoint changed'
        if name=='mont_v5':
            (work/'worktodo.txt').write_text('ECM2=1,2,347,-1,100000,1000000,8\n')
            fresh=run(name+'_fresh',work,ini)
            assert 'legacy GPU checkpoint imported:' not in fresh,'completed legacy curves replayed'
            assert len(points(work/'m0_m347_1e5.save'))==16

    ed_identity,ed_rows=batch(fixtures/'restart_ed_gpu0_auto_checkpoint_before_fault.bin')
    ed_payload=legacy_edwards(ed_rows[0][1],100000)
    expected=points(fixtures/'restart_ed_gpu0_auto/e_m347_1e5.save')
    for name in ('ed_v1_text_payload','ed_raw_text_payload'):
        work,ini=setup(name,model='ed')
        if name=='ed_v1_text_payload':
            records=[(sigma,ed_payload if i==0 else b'') for i,(sigma,_) in enumerate(ed_rows)]
            (work/'.e_s1_m347_100000.ckpt').write_bytes(v1(ed_identity,records))
        else:(work/'ed_m347_1e5.save.gpu.ckpt').write_bytes(ed_payload)
        text=run(name,work,ini);assert 'GPU Edwards resume:' in text
        if name=='ed_raw_text_payload':
            assert 'legacy Edwards checkpoint imported:' in text
            assert (work/'ed_m347_1e5.save.gpu.ckpt').read_bytes()==ed_payload
        assert points(work/'e_m347_1e5.save')==expected,'legacy Edwards endpoint changed'
        if name=='ed_raw_text_payload':
            (work/'worktodo.txt').write_text('ECM2=1,2,347,-1,100000,1000000,8\n')
            fresh=run(name+'_fresh',work,ini)
            assert 'legacy Edwards checkpoint imported:' not in fresh,'completed Edwards curves replayed'
            assert len(points(work/'e_m347_1e5.save'))==16

    # Reuse the validated M991 arithmetic input while keeping original n=1982.
    # It also selects the 1024-bit tier in the restricted test build for PARAM2.
    task=f'ECM2=1,2,1982,-1,1000,1000,2,20260922,"{(1<<991)+1}"\n'
    for model,target,param in [('mont','cpu',0),('mont','gpu',0),('mont','gpu',2),('mont','gpu',3),('ed','cpu',0),('ed','gpu',0)]:
        for pattern in ('','custom_{n}_{b1}.save'):
            name=f'name_{model}_{target}{param}_{"custom" if pattern else "default"}'
            work,ini=setup(name,model,target,param,pattern,task);run(name,work,ini)
            prefix='e' if model=='ed' else 'm'+str(param)
            filename=f'{prefix}_{"custom_1982" if pattern else "m1982"}_1e3.save'
            assert (work/filename).exists(),(name,filename)
            assert len(points(work/filename))==2

    # ECMSTAGE1 runs Stage1. Keep its original START text,
    # but publish the actual prefixed save token to finished and Prime95.
    aid='ABCDEF0123456789ABCDEF0123456789'
    baseline=out/'name_mont_gpu0_default/m0_m1982_1e3.save'
    for name,model,history,absolute in [('queue_mont_history','mont',True,False),
                                       ('queue_ed_empty','ed',False,False),
                                       ('queue_csv_absolute','mont',False,True)]:
        work,ini=setup(name,model=model)
        local_exe=work/'ecm_cuda.exe';shutil.copyfile(exe,local_exe)
        shutil.copyfile(exe.parent/'gmp-10.dll',work/'gmp-10.dll')
        source=work/('comma, name_1e3.save' if absolute else 'm1982_1e3.save')
        source.write_bytes(baseline.read_bytes() if history else b'')
        source_bytes=source.read_bytes();prefix='e' if model=='ed' else 'm0'
        target=source.with_name(prefix+'_'+source.name)
        token=str(source) if absolute else source.name
        new_token=str(target) if absolute else target.name
        task=f'ECMSTAGE1={aid},1,2,1982,-1,"{token}",260000000,{2 if history else 0},2,"{(1<<991)+1}"\n'
        (work/'worktodo.txt').write_text(task)
        prime=work/'p95';prime.mkdir();(prime/'worktodo.txt').write_text('[Worker #1]\n')
        with ini.open('a') as file:file.write(f'sigma=20260924\np95_worktodo_path={json.dumps(str(prime/"worktodo.txt"))}\np95_sync_dir={json.dumps(str(prime/"saves"))}\np95_add_workers="1"\nsave_sync_dir_1={json.dumps(str(prime))}\n')
        text=run(name,work,ini,local_exe)
        expected=task.strip().replace('"'+token+'"','"'+str(target)+'"')
        assert 'START: '+task.strip() in text
        assert (work/'finished.txt').read_text().strip()==expected
        assert source.read_bytes()==source_bytes
        assert len(points(target))==(4 if history else 2)
        if history:assert set(points(baseline)).issubset(set(points(target)))
        if model=='mont':
            delivered=expected.replace('ECMSTAGE1=','ECMSTAGE2=',1).replace('"'+str(target)+'"','"'+str(prime/'saves'/target.name)+'"')
            assert delivered in (prime/'worktodo.add').read_text().splitlines()
            assert (prime/'saves'/target.name).exists()
            assert (prime/'worktodo.txt').read_text()=='[Worker #1]\n'
        else:assert not (prime/'worktodo.add').exists()

    work,ini=setup('queue_reject_mixed_history',model='ed')
    source=work/'legacy_1e3.save';source.write_bytes(baseline.read_bytes());before=source.read_bytes()
    task=f'ECMSTAGE1=1,2,1982,-1,"{source}",260000000,2,2,"{(1<<991)+1}"\n'
    (work/'worktodo.txt').write_text(task)
    text=run('queue_reject_mixed_history',work,ini,good=False)
    assert 'legacy save belongs to another curve/parametrization' in text
    assert source.read_bytes()==before and (work/'worktodo.txt').read_text()==task

    # Each typed queue save overrides the INI construction, without leaking into
    # the next line. B1 belongs to the filename even if an unrelated key is present.
    for prefix,model,target,param in [('e','mont','gpu',3),('m0','ed','gpu',2),
                                      ('m2','ed','gpu',0),('m3','ed','gpu',0),
                                      ('e','mont','cpu',0),('m0','ed','cpu',0)]:
        name=f'prefix_{prefix}_{target}';work,ini=setup(name,model,target,param)
        save=work/f'{prefix}_m1982_1e3.save'
        task=f'ECMSTAGE1=1,2,1982,-1,"{save}",260000000,0,2,"{(1<<991)+1}"\n'
        (work/'worktodo.txt').write_text(task)
        with ini.open('a') as file:
            file.write('sigma=20260922\n'+('lcm="12"\n' if prefix=='e' else ''))
        text=run(name,work,ini)
        actual=points(save);assert len(actual)==2
        assert all(row[3]=='1000' for row in actual)
        if prefix=='e':assert all(row[0]=='ECM-EDWARDS-AM' and row[1] is None for row in actual)
        # GMP-ECM omits PARAM for Suyama PARAM0; an absent PARAM means zero.
        else:assert all(row[0]=='ECM' and (row[1] or '0')==prefix[1:] for row in actual)
        assert (work/'finished.txt').read_text()==task

    name='prefix_mixed_queue';work,ini=setup(name,param=2)
    local_exe=work/'ecm_cuda.exe';shutil.copyfile(exe,local_exe)
    shutil.copyfile(exe.parent/'gmp-10.dll',work/'gmp-10.dll')
    prime=work/'p95';prime.mkdir();(prime/'worktodo.txt').write_text('[Worker #1]\n')
    with ini.open('a') as file:file.write(f'sigma=20260922\np95_worktodo_path={json.dumps(str(prime/"worktodo.txt"))}\np95_sync_dir={json.dumps(str(prime/"saves"))}\np95_add_workers="1"\n')
    tasks=[]
    for prefix in ('e','m0','m3','legacy'):
        save=work/f'{prefix}_m1982_1e3.save'
        tasks.append(f'ECMSTAGE1=1,2,1982,-1,"{save}",260000000,0,2,"{(1<<991)+1}"')
    (work/'worktodo.txt').write_text('\n'.join(tasks)+'\n')
    mixed_log=run(name,work,ini,local_exe)
    finished=(work/'finished.txt').read_text().splitlines()
    assert finished[:3]==tasks[:3]
    assert 'm2_legacy_' in finished[3]
    delivered=(prime/'worktodo.add').read_text().splitlines()
    assert [line for line in delivered if line.startswith('ECMSTAGE2=')]==[
        task.replace('ECMSTAGE1=','ECMSTAGE2=',1).replace(str(work),str(prime/'saves'))
        for task in tasks[1:3] if 'task='+task not in mixed_log]
    assert all(row[:2]==('ECM','2') for row in points(work/'m2_legacy_m1982_1e3.save'))
    for prefix in ('m2','m3'):
        name=f'prefix_reject_{prefix}_cpu';work,ini=setup(name,target='cpu')
        task=f'ECMSTAGE1=1,2,1982,-1,"{prefix}_m1982_1e3.save",0,0,2,"{(1<<991)+1}"\n'
        (work/'worktodo.txt').write_text(task)
        text=run(name,work,ini,empty=False)
        assert 'requires target=gpu' in text and '# ERROR' in (work/'worktodo.txt').read_text()
        assert not list(work.glob('*.save'))
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,checks=len(report['checks']))))


if __name__=='__main__':main()
