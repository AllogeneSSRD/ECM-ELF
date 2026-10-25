"""GPU failure/publication gates, replaying a successful event-wait benchmark row."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exe', type=Path, help='Replay with the final build after configuration-only changes')
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads(a.measurements.read_text(encoding='utf-8'))
    command = data['runs'][0]['command']
    if a.exe:
        command[0] = str(a.exe.resolve())
    env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_HOST_EVENT_WAIT='1', NTT_HOST_READBACK_WAIT='1', NTT_S4_TIMING_FIFO='1', NTT_NO_PROGRESS='1')
    cases = [('invalid_wait', {'NTT_HOST_EVENT_WAIT':'2'}, 'NTT_HOST_EVENT_WAIT must be 0 or 1'),
             ('invalid_fifo', {'NTT_S4_TIMING_FIFO':'-1'}, 'NTT_S4_TIMING_FIFO must be 0 or 1'),
             ('invalid_readback', {'NTT_HOST_READBACK_WAIT':'2'}, 'NTT_HOST_READBACK_WAIT must be 0 or 1'),
             ('nonnumeric_readback', {'NTT_HOST_READBACK_WAIT':'yes'}, 'nonnumeric policy: NTT_HOST_READBACK_WAIT'),
             ('oracle_final_drain', {'NTT_S4_ORACLE_TEST_BAD':'1'}, 'the device reduction disagrees with GMP')]
    report = []
    for name, changes, expected in cases:
        cmd = list(command)
        result,debug,log = [out/(name+s) for s in ('.jsonl','.debug.log','.log')]
        cmd[cmd.index('--results')+1] = str(result)
        cmd[cmd.index('--debug-log-file')+1] = str(debug)
        with log.open('wb') as stream:
            run = subprocess.run(cmd, cwd=out, env=env|changes, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        text = log.read_text(encoding='utf-8', errors='replace')
        if debug.exists():
            text += debug.read_text(encoding='utf-8', errors='replace')
        if not run.returncode or expected not in text or (result.exists() and result.stat().st_size):
            raise AssertionError(f'{name}: failure not detected or result published; see {log}')
        report.append(dict(name=name, exit_code=run.returncode, published=False, detected=True))
        print(name, 'passed', flush=True)
    (out/'summary.json').write_text(json.dumps(dict(passed=True, cases=report), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
