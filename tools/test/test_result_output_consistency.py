"""Exercise real result ledger export order, batching and final-record CRC offline."""
import argparse,json,shutil,subprocess,zlib,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def crc_valid(row):
    ed=row.get('Edwards');sigma=('E' if ed is not None else '')+str((ed or row.get('GMP-ECM-param3') or {}).get('sigma',row.get('sigma','')))
    bound=row.get('b2',0)
    target=str(row['exponent']) if 'exponent' in row else 'N='+str(row.get('N', (int(row.get('k',1))*int(row.get('b',2))**int(row.get('n',0))+int(row.get('c',0)))//math.prod(map(int,row.get('known-factors',[])))))
    slots=[target,'ECM',','.join(row.get('factors',[])),','.join(row.get('known-factors',[])),
        str(row['b1']),str(bound) if bound>row['b1'] else '',sigma,str(row['fft-length']),'invariant:0',
        row['program']['name'],row['program']['version'],'','',row.get('os',{}).get('os','Windows'),
        row.get('os',{}).get('architecture','x86_64'),row['timestamp']]
    return row.get('checksum')==dict(version=1,checksum=f'{zlib.crc32(";".join(slots).encode()):08X}')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    gmp=ROOT/'third_party/gmp-zen3/dist';exe=out/'fixture.exe'
    script=out/'compile.cmd';vc=r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
    script.write_text('@echo off\ncall "'+vc+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 /I"'+str(gmp/'include')+'" "'+str(ROOT/'tools/test/result_output_fixture.cpp')+
        '" /Fe:"'+str(exe)+'" /Fo:"'+str(out/'fixture.obj')+'" /link /LIBPATH:"'+str(gmp/'lib')+'" gmp.lib\n',encoding='utf-8')
    r=subprocess.run(['cmd','/c',str(script)],capture_output=True);(out/'compile.log').write_bytes(r.stdout+r.stderr);assert not r.returncode,'inspect compile.log'
    shutil.copy2(gmp/'bin/gmp-10.dll',out/'gmp-10.dll')
    r=subprocess.run([str(exe),str(out/'run')],capture_output=True);(out/'run.log').write_bytes(r.stdout+r.stderr);assert not r.returncode,'inspect run.log'
    read=lambda name:[json.loads(s) for s in (out/'run'/name).read_text(encoding='utf-8').splitlines()]
    internal=read('internal.jsonl');first=read('stage1.json.txt');second=read('stage2.json.txt');geometry=read('geometry.json.txt');mixed=read('mixed.json.txt')
    checks=dict(internal_save_order=[r['Edwards']['sigma'] for r in internal]==[900,70,1000],
        stage1_batch=len(first)==1 and first[0]['curves']==3 and first[0]['b2']==first[0]['b1'],
        stage1_crc=all(map(crc_valid,first)),stage2_crc=all(map(crc_valid,second)),
        stage2_batch=len(second)==1 and second[0]['curves']==3,
        stage2_varying_geometry=len(geometry)==1 and geometry[0]['curves']==3 and geometry[0]['d']==30034 and geometry[0]['poly-size']==160 and geometry[0]['timestamp']=='2026-10-20 12:00:02' and all(map(crc_valid,geometry)),
        immediate_factor=read('factor.json.txt')[0]['curves']==1 and all(map(crc_valid,read('factor.json.txt'))),
        mixed_factor_and_nf=len(mixed)==2 and {r['status'] for r in mixed}=={'F','NF'} and all(r['curves']==1 for r in mixed) and all(map(crc_valid,mixed)),
        factor_before_nf=[r['status'] for r in mixed]==['F','NF'],
        interleaved_publication=[r['status'] for r in read('alternating.json.txt')]==['NF','F','NF'],
        stage1_task_append=[r['timestamp'] for r in read('append1.json.txt')]==['2026-10-21 12:00:02','2026-10-21 12:00:01','2026-10-21 12:00:00'],
        stage2_task_append=[r['timestamp'] for r in read('append2.json.txt')]==['2026-10-21 12:00:02','2026-10-21 12:00:01','2026-10-21 12:00:00'],
        ledgers_under_data=all((out/'run/data'/(name+'.ecm-ledger')).is_dir() for name in ('internal.jsonl','stage1.json.txt','stage2.json.txt')))
    for name in ('append1.json.txt','append2.json.txt','alternating.json.txt'):
        checks[name+'_crc']=all(map(crc_valid,read(name)))
    def copied_case(name):
        work=out/name;work.mkdir();path=work/'results.json.txt'
        source=out/'run/alternating.json.txt';shutil.copy2(source,path)
        ledger=work/'data/results.json.txt.ecm-ledger'
        shutil.copytree(source.parent/'data/alternating.json.txt.ecm-ledger',ledger)
        return path,ledger
    def export(path,good=True):
        result=subprocess.run([str(exe),'--export-submission',str(path)],capture_output=True)
        assert (result.returncode==0)==good,(path,result.stderr)
    path,ledger=copied_case('partial_append');expected=path.read_bytes()
    path.write_bytes(expected[:expected.index(b'\n')+21]);export(path)
    checks['partial_append_recovery']=path.read_bytes()==expected
    export(path);checks['recovery_no_duplicates']=path.read_bytes()==expected
    path,ledger=copied_case('missing_output');expected=path.read_bytes();path.unlink();export(path)
    checks['missing_output_keeps_publication_order']=path.read_bytes()==expected
    path,ledger=copied_case('existing_output');(ledger/'publication.order').unlink()
    rows=path.read_bytes().splitlines();prefix=rows[1]+b'\r\n'+rows[0]
    path.write_bytes(prefix);export(path)
    checks['adopt_existing_bytes_and_order']=path.read_bytes()==prefix+b'\n'+rows[2]+b'\n'
    before=path.read_bytes();before_time=path.stat().st_mtime_ns;export(path)
    checks['unchanged_export_does_not_write']=path.read_bytes()==before and path.stat().st_mtime_ns==before_time
    path,ledger=copied_case('changed_output');damaged=b'changed\n'+path.read_bytes();path.write_bytes(damaged)
    export(path,good=False);checks['changed_output_refused_without_rewrite']=path.read_bytes()==damaged
    path,ledger=copied_case('damaged_order');expected=path.read_bytes()
    order=ledger/'publication.order';order.write_bytes(order.read_bytes().replace(b'publication-order',b'publication-error'))
    export(path,good=False);checks['damaged_history_refused_without_rewrite']=path.read_bytes()==expected
    (out/'result.json').write_text(json.dumps(dict(complete=all(checks.values()),checks=checks),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks));assert all(checks.values()),'result output regression'
if __name__=='__main__':main()
