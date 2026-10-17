"""Offline regression checks for legacy NF conversion; never contacts GIMPS."""
import argparse
import importlib.util
import json
from pathlib import Path
import secrets

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('rewrite',ROOT/'tools/ecm_report/rewrite_stage2_results.py')
rewrite=importlib.util.module_from_spec(spec);spec.loader.exec_module(rewrite)


def bitwise_crc(text):
    value=0xffffffff
    for byte in text.encode('utf-8'):
        value^=byte
        for _ in range(8):value=(value>>1)^(0xedb88320 if value&1 else 0)
    return f'{value^0xffffffff:08X}'


def validate_crc(row):
    marker='E' if 'Edwards' in row else ''
    expected=(f'{row["exponent"]};ECM;;'+','.join(row['known-factors'])+
              f';{row["b1"]};{row["b2"]};{marker};{row["fft-length"]};invariant:0;'
              f'ECM-ELF;{row["program"]["version"]};;;Windows;x86_64;{row["timestamp"]}')
    assert row['checksum']==dict(version=1,checksum=bitwise_crc(expected))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actual-export',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);checks=[]
    assert bitwise_crc('123456789')=='CBF43926';checks.append('independent_crc_reference')
    catalog=a.output/'targets.json';catalog.write_text(json.dumps([dict(exponent=29,N_hex=f'{(1<<29)-1:x}',known_factors=[])]))
    targets=rewrite.target_catalog(catalog);version=rewrite.public_version()
    def receipt(sigma,record,b2):
        return dict(status='stage2_completed',N_hex=f'{(1<<29)-1:x}',sigma=sigma,record=record,B1=1000,B2=b2,
                    param=0,worker=1,save='unit.save',bad_factors=0,hits=0,factors=[],factorization_complete=True,
                    requested_D=30,timestamp_ms=1792000000000+record*1000)
    rows=[receipt(31,1,1000000),receipt(32,2,1000000),receipt(31,1,1000000),
          receipt(33,3,2000000),receipt(34,4,2000000),receipt(33,3,3000000)]
    output,audit=rewrite.convert(list(enumerate(rows,1)),targets,version,dict(user=secrets.token_hex(12)))
    assert [r['curves'] for r in output]==[2,1,1] and audit['unique_curves']==4 and len(audit['dropped'])==2
    assert all('Edwards' not in r and 'security-code' not in r and 'd' not in r for r in output)
    for r in output:validate_crc(r)
    checks.extend(['unique_maximum_B2','adjacent_batch_boundaries','montgomery_marker_omitted','requested_D_is_not_actual_D','numeric_version_and_crc'])
    ed=receipt(40,1,1000000);ed.update(curve='ed',param=None)
    result,_=rewrite.convert([(1,ed)],targets,version,{})
    assert result[0]['Edwards']=={};validate_crc(result[0]);checks.append('edwards_empty_NF_marker')
    log=a.output/'geometry.log'
    log.write_text('stage2_plan: save=unit.save requested=1\ncurve_start: 1/1 record=1 sigma=31 B1=1000\n'
                   'stage2_real: device=1 N_bits=29 sigma=31 B1=1000 B2=1000000 requested_D=0\n'
                   'real_shape: D=30 P=phi(D)/2=4 baby_points=4 B1=1000 B2=1000000 S_bits=29\n'
                   'stage2_result: record=1 seconds=1.0\n')
    result,_=rewrite.convert([(1,rows[0])],targets,version,{},rewrite.geometry_logs([log]))
    assert result[0]['d']==30 and result[0]['poly-size']==4;validate_crc(result[0]);checks.append('completed_geometry_recovery')
    for name,patch in [('failed',dict(status='error')),('bad_factor',dict(bad_factors=1)),('factor',dict(factors=['233'])),
                       ('wrong_target',dict(N_hex='ff')),('wrong_program',dict(program=dict(name='Prime95',version='31.4')))]:
        try:rewrite.convert([(1,rows[0]|patch)],targets,version,{})
        except ValueError:checks.append(name+'_rejected')
        else:raise AssertionError(name)
    for name,target in [('prime',dict(exponent=127,N_hex=f'{(1<<127)-1:x}',known_factors=[])),
                        ('wrong_known_factors',dict(exponent=29,N_hex=f'{(1<<29)-1:x}',known_factors=['233']))]:
        path=a.output/(name+'.json');path.write_text(json.dumps([target]))
        try:rewrite.target_catalog(path)
        except ValueError:checks.append(name+'_rejected')
        else:raise AssertionError(name)
    ini=a.output/'identity.ini';token=secrets.token_hex(12)
    ini.write_text(f'user=\ncomputer=\n[Worker #2]\nuser={token}\n')
    assert rewrite.ini_identity(ini,2)['user']==token and rewrite.ini_identity(ini,1)['user']==''
    checks.append('ini_worker_identity')
    if a.actual_export:
        actual=[json.loads(s) for s in a.actual_export.read_text(encoding='utf-8').splitlines()]
        for r in actual:
            assert r['status']=='NF' and r['program']['version']==version and 'security-code' not in r
            assert r['fft-length']>0 and 'Edwards' not in r;validate_crc(r)
        checks.append('actual_export_independent_crc')
    (a.output/'summary.json').write_text(json.dumps(dict(complete=True,checks=checks,actual_batches=len(actual) if a.actual_export else None),indent=2)+'\n')
    print('PASS',len(checks),'offline conversion checks')


if __name__=='__main__':main()
