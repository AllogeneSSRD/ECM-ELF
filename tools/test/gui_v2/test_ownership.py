"""Exercise process-lifetime queue exclusion, partitioning, aliases and release."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); exe = args.probe.resolve(); out = args.out.resolve(); out.mkdir()
    queue = out / 'worktodo.txt'; queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n')
    startup = subprocess.STARTUPINFO(); startup.dwFlags = subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow = 0
    cases = []
    for owner, other, expected in [(1,1,23),(1,2,0),(0,1,23),(1,0,23),(0,0,23)]:
        process = subprocess.Popen([str(exe),str(queue),str(owner),'900'], stdout=subprocess.PIPE, startupinfo=startup)
        assert process.stdout.readline().strip() == b'ACQUIRED'
        alias = queue.parent / '..' / queue.parent.name / queue.name
        run = subprocess.run([str(exe),str(alias),str(other),'0'], capture_output=True, startupinfo=startup, timeout=5)
        assert run.returncode == expected, run.stdout
        process.wait(timeout=5); assert process.returncode == 0
        released = subprocess.run([str(exe),str(queue),str(other),'0'], capture_output=True, startupinfo=startup, timeout=5)
        assert released.returncode == 0, released.stdout
        cases.append(dict(owner=owner,contender=other,exit=expected,released=True))
    # A killed owner must release its kernel locks without deleting/replacing the lock file.
    process = subprocess.Popen([str(exe),str(queue),'1','30000'],stdout=subprocess.PIPE,startupinfo=startup)
    assert process.stdout.readline().strip() == b'ACQUIRED'
    process.kill(); process.wait()
    run = subprocess.run([str(exe),str(queue),'1','0'],capture_output=True,startupinfo=startup)
    assert run.returncode == 0
    result = dict(passed=True,cases=cases,killed_owner_released=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
