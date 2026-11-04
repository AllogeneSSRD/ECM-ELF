"""Check GPU backend routing through the public workers in a co-installed Windows bundle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    with tempfile.TemporaryDirectory(prefix='ECM backend ') as directory:
        config = Path(directory) / 'workflow config.toml'
        config.write_text('[stage1]\nstage1_backend="cuda"\n[stage2]\nstage2_backend="cuda"\n'
                          '[workers.2.stage1]\nstage1_backend="opencl"\n'
                          '[workers.2.stage2]\nstage2_backend="opencl"\n', encoding='utf-8')
        for stage, cuda, opencl in [(1,'ecm_cuda.exe','ecm.exe'),
                                     (2,'ecm_cuda_stage2.exe','ecm_opencl_stage2.exe')]:
            query = ['--gpu-info','-d'] if stage == 1 else ['--cost-device-info','--device']
            for exe, worker, override, masked, expected in [
                    (cuda,2,None,False,'opencl'), (opencl,1,None,False,'cuda'),
                    (cuda,2,'cuda',False,'cuda'), (opencl,2,'auto',False,'cuda'),
                    (cuda,1,'auto',True,'opencl')]:
                env = os.environ.copy()
                device = args.device
                if masked:
                    env['CUDA_VISIBLE_DEVICES'] = '-1'
                    # NVIDIA's OpenCL ICD also honors the mask. Query the surviving ICDs.
                    if stage == 2:
                        import re
                        listing = subprocess.check_output([str(bundle/opencl),'--list-devices'],env=env,text=True)
                        match = re.search(r'index=(\d+).*? supported=1',listing)
                        if not match:
                            print('No OpenCL device survives the CUDA mask')
                            continue
                        device = int(match[1])
                cmd = [str(bundle/exe),'--config',str(config),'--worker',str(worker),
                       *query,str(device)]
                if override:
                    cmd.extend(['--gpu-backend',override])
                output = subprocess.check_output(cmd, env=env, cwd=directory, text=True,
                                                 errors='replace', timeout=45)
                if stage == 1:
                    actual = next(line.split('=',1)[1].lower() for line in output.splitlines()
                                  if line.startswith('backend='))
                else:
                    info = json.loads(output.strip())
                    actual = 'opencl' if info['fixed_mode'] == -2 else 'cuda'
                if expected not in actual:
                    raise ValueError(f'{cmd}: expected {expected}, got {output}')
                print(f'Stage{stage} {exe} worker={worker} CLI={override} hidden={masked}: {actual}')

        # A GUI stop event must reach the actual queue owner across the handoff.
        from check_opencl_stage2_worker import config as write_config, records, save_line, stop_after_first_start
        queue = Path(directory) / 'queue'
        ini = write_config(queue,args.device)
        with ini.open('a',encoding='utf-8') as out:
            out.write('stage2_backend="opencl"\nstage2_d=30\n')
        (queue/'input.save').write_text(save_line((1<<127)-1)*3,encoding='utf-8')
        (queue/'worktodo.txt').write_text('ECMSTAGE2=1,2,127,-1,"input.save",300,0,3\n',encoding='utf-8')
        stop_after_first_start(bundle/'ecm_cuda_stage2.exe',queue)
        rows = records(queue)
        assert len(rows)==1 and rows[0]['stage2-engine']['name']=='opencl-ntt'
        print('Forwarded GUI stop: one committed OpenCL curve')


if __name__ == '__main__':
    main()
