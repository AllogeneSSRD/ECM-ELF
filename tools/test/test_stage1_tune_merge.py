"""Offline Stage1 merge checks; synthetic failures never alter source profiles."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2',type=Path,required=True)
    p.add_argument('--input',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    tool=ROOT/'tools/bench/merge_stage1_tune_profiles.py'
    spec=importlib.util.spec_from_file_location('merge_tool',tool)
    merge_tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(merge_tool)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    original={str(path.resolve()):sha(path) for path in a.input}
    profiles=[tomllib.loads(path.read_text(encoding='utf-8-sig')) for path in a.input]
    assert len(profiles)>=2
    cli_rejected=0
    def run(label,paths,destination,replace=False,success=True):
        args=[sys.executable,str(tool),'--stage2',str(a.stage2.resolve()),'--output',str(destination),
              '--evidence',str(out/(label+'_evidence'))]
        args += [x for path in paths for x in ('--input',str(path.resolve()))]
        if replace:args+=['--replace-scopes']
        proc=subprocess.run(args,capture_output=True,text=True,errors='replace',timeout=60)
        (out/(label+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert (proc.returncode==0)==success,(label,proc.stdout,proc.stderr)
        return proc
    combined=out/'combined.toml';run('merged',a.input,combined)
    result=tomllib.loads(combined.read_text())
    expected={tuple(s[k] for k in merge_tool.SCOPE):s for p in profiles for s in p['stage1'].values()}
    assert {tuple(s[k] for k in merge_tool.SCOPE):s for s in result['stage1'].values()}==expected
    assert result['summary']['measured']==len(expected)
    assert not any(key in combined.read_text() for key in ('binary_sha256','manifest','source_path'))
    duplicate=out/'duplicate.toml';run('identical',a.input+[a.input[0]],duplicate)
    assert tomllib.loads(duplicate.read_text())==result
    before=sha(combined)
    # A complete real profile with changed paired timing arrays is synthetic,
    # used only to verify explicit conflict replacement, never as measurements.
    conflict=copy.deepcopy(profiles[0])
    for sample in conflict['stage1'].values():
        for key in ('seconds','gpu_seconds'):
            if key in sample:sample[key]=[x*2 for x in sample[key]]
        for key in ('median_seconds','mad_seconds','median_gpu_seconds'):
            if key in sample:sample[key]*=2
    text,_=merge_tool.merge([conflict]);conflict_file=out/'synthetic_conflict.toml';conflict_file.write_text(text)
    run('conflict_rejected',[a.input[0],conflict_file],combined,success=False);cli_rejected+=1
    assert sha(combined)==before
    replaced=out/'synthetic_replaced.toml'
    run('explicit_replace',[a.input[0],conflict_file],replaced,replace=True)
    assert list(tomllib.loads(replaced.read_text())['stage1'].values())==list(conflict['stage1'].values())
    rejected=0
    for section,key,value in [('profile','unit','other'),('profile','repeats',4),
                              ('device','uuid_hex','0'*32),('device','cuda_driver',1),
                              ('policy','algorithm','prac'),('policy','exp_cache','on')]:
        changed=copy.deepcopy(profiles[0]);changed[section][key]=value
        try:merge_tool.merge([profiles[0],changed])
        except ValueError:rejected+=1
        else:raise AssertionError((section,key))
    invalid=out/'synthetic_incomplete.toml'
    invalid.write_text(a.input[0].read_text(encoding='utf-8-sig').replace('complete = 1','complete = 0'))
    run('incomplete_rejected',[invalid],combined,success=False);cli_rejected+=1
    assert sha(combined)==before
    # Native readers accept unknown keys; merging must still reject identities
    # rather than propagate them into the published performance TOML.
    identity=out/'synthetic_identity.toml'
    identity.write_text(a.input[0].read_text(encoding='utf-8-sig').replace(
        '[profile]', '[profile]\nsource_path = "private/input"'))
    run('identity_rejected',[identity],combined,success=False);cli_rejected+=1
    assert sha(combined)==before
    run('output_input_collision',[a.input[0]],a.input[0],success=False);cli_rejected+=1
    assert all(sha(Path(path))==h for path,h in original.items())
    report=dict(complete=True,merged_scopes=len(expected),duplicate_idempotent=True,
        conflict_requires_explicit_replace=True,policy_rejections=rejected,cli_rejections=cli_rejected,
        failed_publish_preserves_destination=True,original_profiles_unchanged=True,gpu_queries=0,curves=0)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    main()
