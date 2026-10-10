"""Small production-engine checks for portable tune, startup calibration and Auto B2.

Raw commands/results and failed cases are retained in a new ignored directory.
Does not change GPU clocks, power, driver settings or other GPU processes.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time
import tomllib

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage2',type=Path,required=True)
    ap.add_argument('--device',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();exe=a.stage2.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    before=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,clocks.gr,power.draw',
                                    '--format=csv,noheader'],text=True)
    (out/'gpu_before.csv').write_text(before,encoding='utf-8')
    selected=next((r for r in before.splitlines() if r.split(',')[0].strip()==str(a.device)),None)
    if selected is None or '4060 Laptop' not in selected:
        raise ValueError('This agreed validation requires the RTX 4060 Laptop device')
    ini=out/'test.ini';ini.write_text('stage2_log_level=quiet\nstage2_log_file=\nstage2_tune_profile=\n',encoding='utf-8')
    report=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),device=selected,cases=[])
    def run(name,args,timeout=180,expected_success=True):
        command=[str(exe),'--ini',str(ini),'--device',str(a.device),*map(str,args)]
        (out/(name+'.command.json')).write_text(json.dumps(command,indent=2),encoding='utf-8')
        start=time.monotonic();p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=timeout)
        (out/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
        receipt=dict(name=name,exit_code=p.returncode,expected_success=expected_success,wall_seconds=time.monotonic()-start)
        report['cases'].append(receipt)
        (out/'report.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if (p.returncode==0)!=expected_success:raise RuntimeError(name+' unexpected status; inspect '+str(out/(name+'.log')))
        return p.stdout+p.stderr
    profile=out/'resident.toml'
    common=['--tune','ecm','--tune-file',profile,'--tune-exponents','512,4423','--tune-d','210',
            '--tune-b2','60000','--tune-repeats',2,'--arena-mb',512,'--owner-budget-mb',640]
    first=run('tune_resident',common+['--tune-budget-seconds',30])
    table=tomllib.loads(profile.read_text(encoding='utf-8'))
    assert table['profile']['format']==5
    assert {512,4423} <= {s['target_bits'] for s in table['sample'].values() if s['source']=='measured'}
    assert all(not isinstance(v,list) for group in table.values() if isinstance(group,dict)
               for entry in group.values() if isinstance(entry,dict) for v in entry.values())
    original={tuple(s[k] for k in ['condition','target_bits','b1','b2','d','carrier_exponent','execution_path']):s['median_seconds']
              for s in table['sample'].values() if s['source']=='measured'}
    run('tune_incremental',common+['--tune-budget-seconds',1])
    again=tomllib.loads(profile.read_text(encoding='utf-8'))
    for s in again['sample'].values():
        key=tuple(s[k] for k in ['condition','target_bits','b1','b2','d','carrier_exponent','execution_path'])
        if key in original and s['source']=='measured':assert s['median_seconds']<=original[key]
    evidence=Path(re.search(r'evidence=(.+)',first).group(1).strip())
    spec=importlib.util.spec_from_file_location('reference',ROOT/'tools/stat/suyama_mont_ref.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    for bits in [512,4423]:
        base=(evidence/f'p{bits}.save').read_text(encoding='utf-8')
        n=int(re.search(r'N=0x([0-9a-f]+)',base).group(1),16)
        point=ref.stage1(26,1000,n)
        save=out/f'n{bits}_b1000.save'
        save.write_text(f'METHOD=ECM; PARAM=0; SIGMA=26; B1=1000; N=0x{n:x}; X=0x{point["x"]:x}; Z=1;\n',encoding='utf-8')
        selected=run(f'cross_b1_{bits}',['--save',save,'--b2',60000,'--d',210,'--plan-only','--tune-profile',profile,
            '--short-calibration',0,'--arena-mb',512,'--owner-budget-mb',640])
        assert '"selected":true' in selected
        if bits==512:
            short=out/'short.toml'
            args=['--save',save,'--b2',60000,'--d',210,'--plan-only','--tune-profile',short,'--arena-mb',512]
            initial=run('short_missing_profile',args)
            assert 'stage2_short_calibration:' in initial and short.is_file()
            resumed=run('short_reuse',args)
            assert 'stage2_short_calibration:' not in resumed
            absent=out/'disabled_missing.toml'
            disabled=run('short_disabled_missing_profile',['--save',save,'--b2',60000,'--d',210,
                '--plan-only','--tune-profile',absent,'--short-calibration',0,'--arena-mb',512])
            assert '"selected":false' in disabled and not absent.exists()
            auto=run('auto_b2_csv',['--save',save,'--auto-b2','--tune-profile',profile,'--short-calibration',0,
                '--stage1-cost-csv',ROOT/'config/stage1_cost_4060_2000mhz.csv','--auto-min-b2',60000,
                '--auto-max-b2',200000000,'--d',210,'--plan-only','--arena-mb',512])
            assert '"type":"auto_b2"' in auto and '"T1_source":"csv_' in auto
    fallback=out/'fallback.toml'
    run('tune_fallback',['--tune','ecm','--tune-file',fallback,'--tune-exponents','512,4423','--tune-d',210,
        '--tune-b2',60000,'--tune-repeats',2,'--tune-budget-seconds',30,'--arena-mb',512,'--owner-budget-mb',0])
    f=tomllib.loads(fallback.read_text(encoding='utf-8'))
    assert {512,4423} <= {s['target_bits'] for s in f['sample'].values() if s['source']=='measured' and s['fold_resident']==0}
    ntt=out/'ntt.toml'
    run('ntt_final_summary',['--tune','ntt','--tune-file',ntt,'--length-log2','10:11',
        '--tune-slices','1,4','--tune-repeats',2,'--tune-budget-seconds',10])
    nt=tomllib.loads(ntt.read_text(encoding='utf-8'))
    assert nt['profile']['format']==5 and len(nt['ntt'])==4
    assert all('seconds' not in s and 'outer_radix_bits' not in s for s in nt['ntt'].values())
    merged=out/'merged.toml'
    run('merge_summaries',['--tune','ecm','--tune-merge',profile,'--tune-merge',fallback,
        '--tune-ntt-profile',ntt,'--tune-file',merged])
    mt=tomllib.loads(merged.read_text(encoding='utf-8'))
    assert len(mt['condition'])>=2 and len(mt['ntt'])>=4
    ntt_before=ntt.read_bytes()
    rejected=run('protect_ntt_input',['--tune','ecm','--tune-merge',profile,
        '--tune-ntt-profile',ntt,'--tune-file',ntt],expected_success=False)
    assert 'tune-file must differ from NTT inputs' in rejected and ntt.read_bytes()==ntt_before
    csv_input=out/'cost_input.toml'
    csv_input.write_bytes((ROOT/'config/stage1_cost_4060_2000mhz.csv').read_bytes());csv_before=csv_input.read_bytes()
    run('protect_csv_input',['--tune','ntt','--tune-file',csv_input,
        '--stage1-cost-csv',csv_input],expected_success=False)
    assert csv_input.read_bytes()==csv_before
    legacy_ntt=ROOT/'data/experiments/ntt_phase_tune_20261010/supplement/ntt_0.toml'
    if legacy_ntt.is_file():
        imported=out/'legacy_import.toml'
        run('merge_legacy_ntt',['--tune','ecm','--tune-merge',profile,
            '--tune-ntt-profile',legacy_ntt,'--tune-file',imported])
        lt=tomllib.loads(imported.read_text(encoding='utf-8'))
        assert any(s['length']==2048 and s['batch']==451 for s in lt['ntt'].values())
    run('execute_4423',['--save',out/'n4423_b1000.save','--b2',60000,'--d',210,'--tune-profile',profile,
        '--short-calibration',0,'--curves',1,'--results',out/'executed.jsonl','--log',out/'executed_engine.log','--arena-mb',512])
    # The first task lies outside calibrated widths, but a later valid task is
    # covered. Partial queue coverage must not trigger width calibration.
    n=((1<<4423)-1)*((1<<521)-1)
    point=ref.stage1(26,20,n);wide=out/'n4944_b20.save'
    wide.write_text(f'METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N=0x{n:x}; X=0x{point["x"]:x}; Z=1;\n',encoding='utf-8')
    small_save=out/'n512_b1000.save'
    small_n=int(re.search(r'N=0x([0-9a-f]+)',small_save.read_text(encoding='utf-8')).group(1),16)
    partial=out/'worktodo_partial.txt'
    partial.write_text(f'ECMSTAGE2={(n-1)//2},2,1,1,"{wide}",60000,0,1\n'
                       f'ECMSTAGE2={(small_n-1)//2},2,1,1,"{small_save}",60000,0,1\n',encoding='utf-8')
    queue_before=partial.read_bytes()
    ranged=run('partially_covered_queue',['--worktodo',partial,'--once','--plan-only',
        '--tune-profile',profile,'--d',210,'--arena-mb',512])
    assert 'stage2_short_calibration:' not in ranged and '"selected":true' in ranged
    assert partial.read_bytes()==queue_before
    saved=ROOT/'data/experiments/ecm_tune_n318_prod_20261010/stage1/n318_generic_b10000000_c8_r0/stage1.save'
    if saved.is_file():
        text=saved.read_text(encoding='utf-8');n=int(re.search(r'N=([0-9]+)',text).group(1));m=(1<<503)-1
        assert m%n==0
        point=ref.stage1(26,20,n);save=out/'m503_20.save'
        save.write_text(f'METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N=0x{n:x}; X=0x{point["x"]:x}; Z=1;\n',encoding='utf-8')
        worktodo=out/'worktodo_carrier.txt'
        worktodo.write_text(f'ECMSTAGE2=1,2,503,-1,"{save}",2600000000,0,1,"{m//n}"\n',encoding='utf-8')
        carrier=out/'carrier.toml'
        args=['--worktodo',worktodo,'--once','--plan-only','--tune-profile',carrier,'--d',30030,'--arena-mb',512]
        automatic=run('worktodo_carrier_inference',args)
        ct=tomllib.loads(carrier.read_text(encoding='utf-8'))
        assert any(s['carrier_exponent']==503 for s in ct['sample'].values())
        locked=run('worktodo_carrier_lock',args+['--carrier-exponent',503])
        assert '"selected":true' in locked and '"carrier_exponent":503' in locked
    report['passed']=True
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
