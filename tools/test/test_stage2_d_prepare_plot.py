"""Check preparation-plot audit rejection using read-only input overrides; no GPU."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import plot_stage2_d_prepare as plot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();study=a.study.resolve();source=study/'report.json';original=source.read_bytes()
    before=json.loads(original);verified=plot.audit(study)
    mutations=[('incomplete',lambda r:r.update(complete=False)),('learning',lambda r:r.update(frozen_control=False)),
        ('failed',lambda r:r.update(failures=['failed'])),
        ('profile changed',lambda r:r['runs'][0].update(profile_unchanged=False)),
        ('wrong parent',lambda r:r['runs'][0]['curves'][0].update(parent_seconds=100000)),
        ('wrong shape',lambda r:r['runs'][0]['curves'][0].update(D=1)),
        ('wrong sum',lambda r:r['runs'][0].update(engine_sum=1)),
        ('nonfinite curve',lambda r:r['runs'][0]['curves'][0].update(engine_seconds=float('nan'))),
        ('nonfinite sum',lambda r:r['runs'][0].update(curve_wall_sum=float('nan'))),
        ('changing geometry',lambda r:r['runs'][0]['selected_shapes'].append([1,0]))]
    read=Path.read_text
    for name,mutate in mutations:
        altered=copy.deepcopy(before);mutate(altered)
        def override(path,*args,**kwargs):
            return json.dumps(altered) if path.resolve()==source else read(path,*args,**kwargs)
        with patch.object(Path,'read_text',override):
            try:plot.audit(study)
            except ValueError:pass
            else:raise AssertionError('Bad evidence accepted: '+name)
    assert source.read_bytes()==original
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(complete=True,checks=1+len(mutations),CUDA_calls=0,read_only_inputs=True,
                source_sha256=plot.sha(source),plot_script_sha256=plot.sha(Path(plot.__file__)),input=verified['input'])
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
