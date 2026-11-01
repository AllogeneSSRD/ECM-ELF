"""Real-worker tri-state precedence and parent/child forwarding on a tiny save."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True,help='Independent sigma26/B1=20 Montgomery M127 point')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=a.exe.resolve();identity=hashlib.sha256(exe.read_bytes()).hexdigest()
    saved=out/'prime127.save';shutil.copy2(a.save,saved)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    cases=[('default','',[],0),('auto','"auto"',[],0),('true','true',[],1),('false','false',[],0),
        ('cli_auto','true',['--groot-layout','auto'],0),('cli_true','false',['--groot-layout','true'],1),
        ('cli_false','true',['--groot-layout','false'],0),('cli_one','false',['--groot-layout','1'],1),
        ('cli_zero','true',['--groot-layout','0'],0),('toml_one','1',[],1),('toml_zero','0',[],0),
        ('toml_two','2',[],None),('toml_float','1.0',[],None),
        ('bad_cli','"auto"',['--groot-layout','2'],None)]
    report=dict(complete=False,exe_sha256=identity,cases=[])

    def invoke(name,command):
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==identity,'Binary changed'
        r=subprocess.run(command,cwd=out,env=env,capture_output=True,timeout=45)
        text=(r.stdout+r.stderr).decode('utf-8',errors='replace')
        (out/(name+'.log')).write_text(text,encoding='utf-8')
        (out/(name+'.command.json')).write_text(json.dumps(command,indent=2),encoding='utf-8')
        return r.returncode,text

    for name,value,extra,expected in cases:
        config=out/(name+'.toml')
        setting=('stage2_groot_balanced='+value+'\n') if value else ''
        config.write_text('[stage2]\n'+setting+'stage2_device='+str(a.device)+'\n'
            'stage2_tune_runtime=0\nstage2_short_calibration=0\n',encoding='utf-8')
        command=[str(exe),'--config',str(config),'--save',str(saved),'--b2','13230','--d','210',
                 '--device',str(a.device),'--plan-only',*extra]
        code,text=invoke(name,command)
        if expected is None:assert code!=0 and ('groot' in text),text
        else:
            assert code==0,text
            layouts=re.findall(r'"groot_balanced"\s*:\s*([01])',text)
            assert layouts and set(layouts)=={str(expected)},(name,layouts,text[-3000:])
        report['cases'].append(dict(name=name,returncode=code,layout=expected))

    # An explicit auto in the queue parent must reach its curve child, otherwise
    # the child silently reapplies this forced-true TOML setting.
    queue=out/'worktodo.txt';queue.write_text('ECMSTAGE2=1,2,127,-1,"prime127.save",13230,0,1\n',encoding='utf-8')
    config=out/'queue.toml';config.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n'
        '[stage2]\nstage2_groot_balanced=1\nstage2_tune_runtime=0\nstage2_short_calibration=0\n',encoding='utf-8')
    results=out/'queue_results.jsonl';debug=out/'queue_debug.log';log=out/'queue_engine.log'
    code,text=invoke('queue_auto',[str(exe),'--config',str(config),'--device',str(a.device),
        '--worker','1','--once','--d','210','--groot-layout','auto','--results',str(results),
        '--submission',str(out/'submission.txt'),'--log',str(log),'--debug-log-file',str(debug)])
    assert code==0,text
    raw=[json.loads(line) for line in results.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    assert len(raw)==1 and raw[0]['status']=='NF',raw
    trace=text+'\n'+debug.read_text(encoding='utf-8')+'\n'+log.read_text(encoding='utf-8')
    layouts=re.findall(r'groot_layout: balanced=([01])',trace)
    assert layouts and set(layouts)=={'0'},layouts
    report['cases'].append(dict(name='queue_auto',returncode=code,layout=0))
    report['complete']=True
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(cases)} public CLI/TOML cases and a real queue-child auto override')


if __name__=='__main__':main()
