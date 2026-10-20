"""Exercise release SASS and PTX images on the available sm89 device.

Native sm89, compatible sm86 SASS, and explicitly forced PTX JIT are separate
coverage classes. Unavailable native hardware remains explicitly untested.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from test_cuda_p3_runtime import stable
from result_format import normalize_result


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('static-gate', 'native-gate', 'stage1-packages', 'stage2-packages', 'native-stage1',
                 'native-stage2', 'small-save', 'fixtures', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    static = json.loads(a.static_gate.read_text())
    assert static['complete'] and len(static['images']) == 12
    release_images = {(row['stage'], row['architecture']):row['exe_sha256'] for row in static['images']}
    assert set(release_images) == {(stage, arch) for stage in ('stage1','stage2') for arch in (60,70,75,86,89,120)}, 'incomplete release architecture matrix'
    native = json.loads(a.native_gate.read_text())
    assert native['complete'], 'native gate incomplete'
    inputs = {str(path):sha(path) for path in (a.native_stage1, a.native_stage2, a.small_save, a.fixtures)}
    assert all(native['inputs'].get(path) == digest for path,digest in inputs.items()), 'native reference identity changed'
    for (stage,arch),digest in release_images.items():
        folder = a.stage1_packages if stage == 'stage1' else a.stage2_packages
        exe = folder/f'cuda-{stage}-sm{arch}'/('ecm_cuda.exe' if stage == 'stage1' else 'ecm_cuda_stage2.exe')
        inputs[str(exe)] = sha(exe)
        assert inputs[str(exe)] == digest, 'release image identity changed'
    out = a.output; out.mkdir(parents=True, exist_ok=False)
    info = subprocess.check_output(['nvidia-smi','-i',str(a.device),'--query-gpu=compute_cap,name,uuid','--format=csv']).decode()
    assert info.splitlines()[1].split(',')[0].strip() == '8.9'
    (out/'device.csv').write_text(info)
    result = dict(complete=False, device=a.device, collector_sha256=sha(__file__),
                  static_gate_sha256=sha(a.static_gate), native_gate_sha256=sha(a.native_gate),
                  inputs=inputs, cases=[], native_architectures=[89],
                  untested_native_architectures=[60,70,75,86,120],
                  unsupported_images=[dict(stage='stage1',arch=v,reason='SASS major differs; package has no PTX') for v in (60,70,120)] +
                    [dict(stage='stage2',arch=120,reason='target newer than available hardware')])
    config = out/'fixture.toml'
    config.write_text('[stage1]\nauto_stage2=0\nedwards_stage2_transfer=false\n'
                      '[stage2]\nstage2_short_calibration=false\nstage2_tune_runtime=false\n')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('ECM_', 'NTT_', 'CUDA_FORCE_', 'CUDA_DISABLE_'))}
    signatures = {}
    identities = {}
    def persist(): (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def invoke(stage, arch, mode, case, exe, args, input=None):
        name = f'{stage}_{arch}_{mode}_{case}'
        folder = out/name; folder.mkdir()
        command = [str(exe),'--config',str(config),*[str(v).replace('$RUN',str(folder)) for v in args]]
        image_env = env | {'NTT_TARGET_LEAF_HASH':'1'} | (
            {'CUDA_FORCE_PTX_JIT':'1'} if mode == 'ptx_jit' else {'CUDA_DISABLE_PTX_JIT':'1'})
        identity = sha(exe); identities[str(exe)] = identity
        if mode != 'reference': assert identity == release_images[(stage,arch)], 'release image identity changed'
        (folder/'invocation.json').write_text(json.dumps(dict(command=command, mode=mode,
            environment={k:v for k,v in image_env.items() if k.startswith(('CUDA_', 'NTT_', 'ECM_'))}, input=input),indent=2)+'\n')
        proc = subprocess.run(command, cwd=folder, env=image_env, input=input.encode() if input else None,
                              capture_output=True, timeout=900)
        text = (proc.stdout+proc.stderr).decode('utf-8','replace')
        (folder/'stdout.log').write_text(text)
        for name in ('debug.log','engine.log'):
            if (folder/name).exists(): text += '\n'+(folder/name).read_text(encoding='utf-8',errors='replace')
        assert proc.returncode == 0,(case,arch,mode,proc.returncode,text[-2000:])
        if stage == 'stage1':
            if case.startswith('mont'):
                times=re.findall(r'GPU stage1 returned: \d+ gputime=([\d.eE+-]+) ms',text)
                assert len(times)==1 and float(times[0])>0
                assert f'GPU: will use device {a.device}:' in text
            else:
                assert f'GPU Edwards Stage1: device={a.device}' in text
                assert 'GPU Edwards arithmetic: algorithm='+('binary' if case=='ed_binary' else 'naf') in text
            rows = []
            for line in (folder/'point.save').read_text(encoding='utf-8-sig').splitlines():
                if 'SIGMA=' not in line: continue
                fields = dict(v.strip().split('=',1) for v in line.split(';') if '=' in v)
                fields.pop('WHO',None); fields.pop('TIME',None); rows.append(fields)
            assert len(rows) == 4
            signature = rows
        else:
            rows = [stable(normalize_result(json.loads(line))) for line in (folder/'result.jsonl').read_text().splitlines()]
            assert len(rows) == 1 and not rows[0]['bad_factors']
            for token in ('mont_selftest: cases=2048 mismatches=0','gmp_check_bad=0','pending=0'):
                assert token in text,(arch,case,token)
            leaves = re.findall(r'^target_descent_values: (.*)$',text,re.M)
            assert leaves
            signature = dict(result=rows, leaves=leaves[-1])
        key = (stage,case)
        if key in signatures: assert signatures[key] == signature,(stage,arch,mode,case,'output changed')
        else: signatures[key] = signature
        assert sha(exe) == identity
        result['cases'].append(dict(stage=stage,arch=arch,mode=mode,case=case,
            binary_sha256=identity, signature_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()))
        persist(); print('PASS image',stage,arch,mode,case,flush=True)
    persist()
    stage1_cases = [('mont0','mont',0),('mont2','mont',2),('mont3','mont',3),
                    ('ed_naf','ed',None),('ed_binary','ed',None)]
    for arch,mode,exe in [(89,'reference',a.native_stage1),
        *[(v,'ptx_jit' if v==75 else 'native_sass' if v==89 else 'compatible_sass',
           a.stage1_packages/f'cuda-stage1-sm{v}/ecm_cuda.exe') for v in (75,86,89)]]:
        for name,curve,param in stage1_cases:
            if name == 'ed_binary': env['ECM_EDWARDS_STAGE1_ALGO'] = 'binary'
            else: env.pop('ECM_EDWARDS_STAGE1_ALGO',None)
            args = ['--curve',curve,'--target','gpu','-d',a.device,'-sigma',26,'-gpucurves',4,
                    '--ckpt',0,'--exp-cache','off','--tmp-dir','$RUN/scratch','-save','$RUN/point.save']
            if param is not None: args += ['--gpu-param',param]
            invoke('stage1',arch,mode,name,exe,[*args,20,0],'(2^127-1)\n')
    env.pop('ECM_EDWARDS_STAGE1_ALGO',None)
    fixtures = json.loads(a.fixtures.read_text())['fixtures']
    wide = next(v for v in fixtures if v['name']=='generic8193')
    assert sha(wide['save']) == wide['save_sha256']
    for arch,mode,exe in [(89,'reference',a.native_stage2),
        *[(v,'ptx_jit' if v<80 else 'native_sass' if v==89 else 'compatible_sass',
           a.stage2_packages/f'cuda-stage2-sm{v}/ecm_cuda_stage2.exe') for v in (60,70,75,86,89)]]:
        for name,save,carrier in [('ordinary',a.small_save,0),('carrier',a.small_save,254),('generic8193',wide['save'],0)]:
            args = ['--save',save,'--b2',13230,'--d',210,'--device',a.device,'--curves',1,
                '--carrier-exponent',carrier,'--short-calibration',0,'--tune-runtime',0,'--factor-only',
                '--results','$RUN/result.jsonl','--log','$RUN/engine.log','--debug-log-file','$RUN/debug.log',
                '--log-level','debug']
            invoke('stage2',arch,mode,name,exe,args)
    assert identities == {path:sha(path) for path in identities}
    assert inputs == {path:sha(path) for path in inputs}
    result['complete'] = True; persist()


if __name__ == '__main__': main()
