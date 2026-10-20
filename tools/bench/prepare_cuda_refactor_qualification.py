"""Prepare fixed full-refactor performance inputs after functional/build gates.

Every Stage2 input is a complete PARAM0 Stage1 point independently checked by
the integer ladder. Preparation is outside all measured boundaries. This does
not change user configuration, queues, GPU clocks, or power settings.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from bench_edwards_gpu_stage1 import reference, validate, integer_expression
from qualify_cuda_refactor import sha, load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'stage1', 'stage2', 'gmp-ecm', 'fixtures', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gate', type=Path, action='append', required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--inputs-only', action='store_true',
                   help='Functional preparation only; the emitted plan cannot start acceptance sampling')
    a = p.parse_args()
    assert a.device == 1
    assert all(load(path)['complete'] for path in a.gate)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    old = a.baseline.resolve()
    exes = dict(stage1=dict(old=old/'ecm_cuda.exe', new=a.stage1.resolve()),
                stage2=dict(old=old/'ecm_cuda_stage2.exe', new=a.stage2.resolve()))
    cases = []
    def binary_fields(kind):
        return {key:value for side,path in exes[kind].items()
                for key,value in ((side+'_exe',str(path)), (side+'_sha256',sha(path)))}
    for bits, curves in ((127,128), (607,64), (2203,16), (9689,4)):
        models = ['ed', 'mont0'] + (['ed-5', 'ed-binary', 'mont2', 'mont3'] if bits == 2203 else [])
        for model in models:
            name = f'stage1_m{bits}_{model}'
            ed = model.startswith('ed')
            sigma = 1000000 if model == 'mont2' else 26
            env = dict(ECM_GPU_STAGE1_ALGO='ladder')
            args = ['--config', '$CONFIG', '--curve', 'ed' if ed else 'mont', '--target', 'gpu',
                '-d', a.device, '-sigma', sigma, '-gpucurves', curves, '--ckpt', 0, '--exp-cache', 'off',
                '--tmp-dir', '$RUN/scratch', '-save', '$RUN/point.save']
            if ed:
                env['ECM_EDWARDS_STAGE1_ALGO'] = 'binary' if model == 'ed-binary' else 'naf'
                if model != 'ed-binary':
                    args += ['--naf-w', 6]
                    env['ECM_EDWARDS_DBL_ALGO'] = '5' if model == 'ed-5' else '4'
            else: args += ['--gpu-param', int(model[-1])]
            args += [10000, 0]
            cases.append(dict(name=name, kind='stage1', **binary_fields('stage1'),
                args=args, environment=env, input=f'(2^{bits}-1)\n', N_hex=format((1 << bits)-1,'x'),
                b1=10000, curves=curves, sigma=sigma, model=model))
            if model == 'mont2':
                oracle = {}
                for index in (0, curves-1):
                    folder = out/'param2_oracle'/str(index); folder.mkdir(parents=True)
                    save = folder/'point.save'
                    command = [str(a.gmp_ecm.resolve()), '-param', '2', '-sigma', str(sigma+index),
                               '-c', '1', '-save', str(save), '10000', '0']
                    (folder/'invocation.json').write_text(json.dumps(command, indent=2)+'\n')
                    proc = subprocess.run(command, cwd=folder, input=(str((1 << bits)-1)+'\n').encode(),
                                          capture_output=True, timeout=300)
                    (folder/'stdout.log').write_bytes(proc.stdout+proc.stderr)
                    assert proc.returncode == 0, ('GMP-ECM PARAM2', index)
                    rows = [line for line in save.read_text().splitlines() if 'SIGMA=' in line]
                    assert len(rows) == 1
                    fields = dict(re.findall(r'(?:^|;\s*)([A-Z0-9]+)=([^;]+)', rows[0]))
                    assert int(fields['SIGMA']) == sigma+index and int(fields['PARAM']) == 2
                    oracle[str(index)] = ['POINT', int(fields['X'], 0)]
                cases[-1]['oracle'] = oracle
                cases[-1]['oracle_kind'] = 'independent GMP-ECM PARAM2 saved affine X'
    native_gates = [path for path in a.gate if any(g.get('name') == 'edwards' for g in load(path).get('gates', []))]
    assert len(native_gates) == 1, 'need the complete native width gate'
    native = native_gates[0].resolve().parent
    source = native/'edwards/saves/n17407_b20_w6_a4.save'
    ed_gate = next(g for g in load(native_gates[0])['gates'] if g['name']=='edwards')
    assert ed_gate['outputs'][str(source.relative_to(native))] == sha(source)
    line = next(line for line in source.read_text().splitlines() if 'SIGMA=' in line)
    fields = dict(re.findall(r'(?:^|;\s*)([A-Z0-9]+)=([^;]+)', line))
    ed_n = integer_expression(fields['N'])
    mont_source = native/'montgomery/evidence/w16384_param0_new/input.txt'
    mont_gate = next(g for g in load(native_gates[0])['gates'] if g['name']=='montgomery')
    assert mont_gate['outputs'][str(mont_source.relative_to(native))] == sha(mont_source)
    mont_n = int(mont_source.read_text().strip(), 0)
    assert ed_n.bit_length() == 17407 and mont_n.bit_length() == 16378
    for model,n,sigma,curves,origin in [
        *[(model,ed_n,int(fields['SIGMA']),1,source) for model in ('ed','ed-5','ed-binary')],
        ('mont0',mont_n,26,2,mont_source)]:
        ed = model.startswith('ed')
        env = dict(ECM_GPU_STAGE1_ALGO='ladder')
        args = ['--config','$CONFIG','--curve','ed' if ed else 'mont','--target','gpu','-d',a.device,
                '-sigma',sigma,'-gpucurves',curves,'--ckpt',0,'--exp-cache','off',
                '--tmp-dir','$RUN/scratch','-save','$RUN/point.save']
        if ed:
            env['ECM_EDWARDS_STAGE1_ALGO'] = 'binary' if model=='ed-binary' else 'naf'
            if model != 'ed-binary':
                args += ['--naf-w',6]
                env['ECM_EDWARDS_DBL_ALGO'] = '5' if model=='ed-5' else '4'
        else: args += ['--gpu-param',0]
        cases.append(dict(name=f'stage1_boundary{n.bit_length()}_{model}',kind='stage1',
            **binary_fields('stage1'),args=[*args,20,0],environment=env,
            input='0x'+format(n,'x')+'\n',N_hex=format(n,'x'),b1=20,curves=curves,sigma=sigma,model=model,
            boundary_input=dict(path=str(origin),sha256=sha(origin))))
    config = out/'fixture.toml'
    config.write_text('[stage1]\nauto_stage2 = 0\nedwards_stage2_transfer = false\n')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('ECM_', 'NTT_', 'CUDA_FORCE_', 'CUDA_DISABLE_'))}
    shapes = [('m127', (1 << 127)-1),
              ('p384', (1 << 384)-(1 << 128)-(1 << 96)+(1 << 32)-1),
              *[(f'm{b}', (1 << b)-1) for b in (521,1279,2203,4423,16381)]]
    inputs = []
    wide = next(c for c in load(a.fixtures)['fixtures'] if c['name'] == 'm16381')
    for name,n in shapes:
        folder = out/'inputs'/name; folder.mkdir(parents=True)
        save = folder/'point.save'
        if name == 'm16381':
            # Stage2 supports actual W=256. Stage1 GPU Montgomery needs six
            # spare bits and stops at 16378 target bits; use the existing
            # independently GMP-verified CPU fixture for this wider target.
            assert sha(wide['save']) == wide['save_sha256'] and wide['stage1_cpu_gmp_equal']
            assert int(wide['N_hex'], 16) == n
            shutil.copy2(wide['save'], save)
            (folder/'origin.json').write_text(json.dumps(wide, indent=2)+'\n')
        else:
            command = [str(exes['stage1']['new']), '--config', str(config), '--curve', 'mont', '--target', 'gpu',
                '-d', str(a.device), '--gpu-param', '0', '-sigma', '26', '-gpucurves', '1', '--ckpt', '0',
                '--exp-cache', 'off', '--tmp-dir', str(folder/'scratch'), '-save', str(save), '20', '0']
            value = '0x'+format(n,'x')+'\n'
            (folder/'invocation.json').write_text(json.dumps(dict(command=command, input=value), indent=2)+'\n')
            proc = subprocess.run(command, cwd=folder, env=env, input=value.encode(), capture_output=True, timeout=300)
            (folder/'stdout.log').write_bytes(proc.stdout+proc.stderr)
            assert proc.returncode == 0, name
        rows = validate(save, n, 20, 1, 26, 'mont0')
        assert rows[0][1:] == reference(n, 26, 20, 'mont0')
        assert rows[0][1] == 'POINT', (name, 'need complete unit point')
        record = dict(name=name, N_hex=format(n,'x'), bits=n.bit_length(), save=str(save),
                      save_sha256=sha(save), independent_integer_ladder=True)
        inputs.append(record)
        variants = [('ordinary',0,2310)]
        if name == 'm127': variants += [('large_tree',0,30030), ('carrier',254,2310)]
        if name == 'm4423': variants += [('carrier',8846,2310)]
        for variant,carrier,d in variants:
            b2 = 26_000_000 if variant == 'large_tree' else 2_600_000
            args = ['--config', '$CONFIG', '--save', str(save), '--b2', b2, '--d', d,
                '--device', a.device, '--curves', 1, '--carrier-exponent', carrier,
                '--short-calibration', 0, '--tune-runtime', 0, '--factor-only',
                '--results', '$RUN/result.jsonl', '--log', '$RUN/engine.log',
                '--debug-log-file', '$RUN/debug.log', '--log-level', 'debug']
            cases.append(dict(name=f'stage2_{name}_{variant}', kind='stage2', **binary_fields('stage2'),
                args=args, environment={}, input='', **{k:v for k,v in record.items() if k!='name'},
                b1=20, b2=b2, d=d, carrier_exponent=carrier))
        print('PASS prepared independent point', name, flush=True)
    plan = dict(schema=1, threshold=1.05, device=a.device, acceptance_ready=not a.inputs_only,
                prerequisites=[str(path.resolve()) for path in a.gate],
                gmp_ecm=dict(path=str(a.gmp_ecm.resolve()), sha256=sha(a.gmp_ecm)),
                preparation_collector_sha256=sha(__file__), inputs=inputs, cases=cases)
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print('Prepared fixed qualification plan:', len(cases), 'cases')


if __name__ == '__main__': main()
