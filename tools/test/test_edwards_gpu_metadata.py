"""Replay live GPU Edwards/Montgomery worker output through GUI factor persistence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from test_edwards_gpu_pipeline import ROOT, mont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args(); out = a.output.resolve(); s1 = a.stage1.resolve(); s2 = a.stage2.resolve(); probe = a.probe.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh evidence directory')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(identity=dict(stage1=sha(s1), stage2=sha(s2), probe=sha(probe), test=sha(__file__),
                  reference=sha(ROOT/'tools/stat/suyama_mont_ref.py')),
                  device=a.device, checks=[], complete=False, formal_performance_samples=0)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    def persist(): (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    def call(name, command, input=None):
        r = subprocess.run([str(x) for x in command], input=input, capture_output=True, cwd=out, env=env, timeout=300)
        raw = r.stdout+r.stderr; path = out/(name+'.log'); path.write_bytes(raw)
        assert r.returncode == 0, (name, r.returncode, raw[-2000:])
        report['checks'].append(dict(name=name, exit=r.returncode, log_sha256=sha(path)))
        persist(); print('PASS', name, flush=True)
        return raw.decode('utf-8', 'replace')
    def replay(name, stage, log, model):
        result = out/(name+'.jsonl')
        call(name, [probe, stage, log, result, model])
        rows = [json.loads(line) for line in result.read_text().splitlines()]
        assert rows and all(row['execution_target'] == 'gpu' for row in rows)
        for row in rows:
            if row['curve_model'] == 'ed': assert 'param' not in row
            else: assert row['curve_model'] == 'mont' and row['param'] == 0
        return rows
    persist()
    n = 1009*((1 << 127)-1)
    text = call('stage1_cli', [s1, '--curve', 'ed', '--target', 'gpu', '-d', a.device, '-sigma', 17,
        '-gpucurves', 1, '--tmp-dir', out/'cli_tmp', '-save', out/'cli.save', 1000, 1000], str(n).encode())
    assert 'curve_model=ed execution_target=gpu' in text
    rows = replay('gui_stage1_cli', 'stage1', out/'stage1_cli.log', 'ed')
    assert rows[0]['sigma'] == 17 and rows[0]['factors'] == ['1009']
    queue = out/'worktodo.txt'; finished = out/'finished.txt'; ini1 = out/'stage1.ini'
    queue.write_text('ECM2=1009,2,127,-1009,1000,0,1,17\n')
    ini1.write_text(f'curve=ed\ntarget=gpu\ndevice={a.device}\nworktodo={queue}\nfinished={finished}\ntmp_dir={out/"queue_tmp"}\nlog_file=\n')
    text = call('stage1_queue', [s1, '-ini', ini1])
    assert 'Edwards Atkin-Morain' in text and 'param3' not in text
    replay('gui_stage1_queue', 'stage1', out/'stage1_queue.log', 'ed')
    save = out/'ed_point.save'; n = 10007*((1 << 127)-1)
    call('stage1_point', [s1, '--curve', 'ed', '--target', 'gpu', '-d', a.device, '-sigma', 17,
        '-gpucurves', 1, '--tmp-dir', out/'point_tmp', '-save', save, 20, 20], str(n).encode())
    ini2 = out/'stage2.ini'; ini2.write_text(f'device={a.device}\n')
    def stage2(name, source, b2, extra=None):
        return call(name, [s2, '--ini', ini2, '--save', source, '--b2', b2, '--d', 210,
            '--device', a.device, '--results', out/(name+'_engine.jsonl'), '--log', out/(name+'_engine.log'),
            '--short-calibration', 0, '--tune-runtime', 0, '--factor-only', *(extra or [])])
    ed = stage2('stage2_ed', save, 40000)
    rows = replay('gui_stage2_ed', 'stage2', out/'stage2_ed.log', 'ed')
    assert rows[0]['factors'] == ['10007'] and rows[0]['sigma'] == 17
    n = (1 << 128)+1; point = mont.stage1(26, 1000, n); mont_save = out/'mont_point.save'
    mont_save.write_text(f'METHOD=ECM; SIGMA=26; B1=1000; N=0x{n:x}; X=0x{point["x"]:x};\n')
    legacy = stage2('stage2_mont', mont_save, 1000000)
    rows = replay('gui_stage2_mont', 'stage2', out/'stage2_mont.log', 'mont')
    assert rows[0]['factors'] == ['59649589127497217'] and rows[0]['sigma'] == 26
    quiet = stage2('stage2_ed_quiet', save, 40000, ['--log-level', 'quiet', '--gui-events'])
    result_lines = [line for line in quiet.splitlines() if line.startswith('stage2_result:')]
    assert len(result_lines) == 1 and 'curve_model=ed' in result_lines[0] and 'sigma=17' in result_lines[0]
    quiet_log = out/'quiet_result_only.log'; quiet_log.write_text(result_lines[0]+'\n')
    replay('gui_stage2_quiet', 'stage2', quiet_log, 'ed')
    mixed = out/'mixed_pipe.log'; mixed.write_text(ed+'\n'+legacy+'\n'+result_lines[0]+'\n')
    rows = replay('gui_mixed_models', 'stage2', mixed, 'mixed')
    assert [row['curve_model'] for row in rows] == ['ed', 'mont', 'ed']
    report['complete'] = True; persist(); print('All live worker/GUI identity gates passed', flush=True)


if __name__ == '__main__': main()
