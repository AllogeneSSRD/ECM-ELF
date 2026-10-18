"""Cross-language column order, legacy format-8 migration and row provenance."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from stage2_tune_matrix import COLUMN_ORDER, ROW_ORDER, aggregate, encode_ids, literal, utc_updated_at, write_profile
from stage2_tune_tables import summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=a.fixture_exe.resolve()
    def run(*args):return subprocess.check_output([str(exe),*map(str,args)],text=True)
    assert json.loads(run('--columns'))==list(COLUMN_ORDER)
    condition=dict(uuid_hex='0123456789abcdef0123456789abcdef',sm_major=8,sm_minor=9,
        cuda_runtime=13030,cuda_driver=13030,gl_fixed_mode=3,outer_unroll_u=0,add_sub_mask=1)
    def sample(token,time,version,ns=None):
        s=dict(condition=0,target_bits=4916,arithmetic_bits=4916,carrier_exponent=0,modulus_kind='generic',
            b1=260000000,b2=1300000000000,d=690690,fold_resident=1,frontier_resident=1,
            execution_path='resident',giant_chunk_points=253440,giant_chain_min=32768,giant_force_ladder=0,
            source='measured',benchmark_kind='runtime_save',repeats=1,verified=1,mean_seconds=time,
            stddev_seconds=0,measurement_ids=encode_ids({token}),program_version=version)
        if ns is not None:s['updated_at']=utc_updated_at(ns)
        return s
    def put(name,conditions,rows):
        file=out/name;file.write_text(write_profile(conditions,rows),encoding='utf-8');return file
    def compare(doc,rows):
        got=summaries(doc);assert len(got)==len(rows)
        for x,y in zip(got,rows):
            for k in ('repeats','mean_seconds','stddev_seconds'):
                assert math.isclose(x[k],y[k],rel_tol=1e-12,abs_tol=1e-14),(k,x,y)
            for k in ('program_version','updated_at','measurement_ids'):
                assert x.get(k,'')==y.get(k,''),(k,x,y)
        return got
    early=sample('early',10,'0.1.0',1700000000000000001)
    late=sample('late',12,'0.2.0',1700000000000000002)
    old=put('old.toml',[condition],[early]);new=put('new.toml',[condition],[late])
    merged=run('--update',old,new);doc=tomllib.loads(merged)
    expected=aggregate([early,late]);compare(doc,expected)
    assert doc['width']['4916']['matrix'][0]==tomllib.loads(write_profile([condition],[early,late]))['width']['4916']['matrix'][0]
    assert all('program_version' not in c for c in doc['condition'].values())
    path=out/'merged.toml';path.write_text(merged,encoding='utf-8')
    assert run('--roundtrip',path)==merged
    assert run('--update',path,new)==merged,'duplicate input refreshed provenance or counts'
    compare(tomllib.loads(run('--update',new,old)),expected)
    # An older numbered executable can run later; time determines provenance.
    last=sample('last',14,'0.0.1',1700000000000000003)
    latest=put('latest.toml',[condition],[last])
    compare(tomllib.loads(run('--update',path,latest)),aggregate([early,late,last]))
    # Real old-format-8 layout: version in condition, no row timestamp.
    legacy_rows=[]
    for i,s in enumerate((early,late)):
        row={k:v for k,v in s.items() if k not in ('target_bits','arithmetic_bits','execution_path','program_version','updated_at')}
        row['condition']=i;legacy_rows.append(row)
    lines=['[profile]','format = 8','unit = "full_stage2"']
    for i,version in enumerate(('0.1.0','0.2.0')):
        lines += [f'[condition.{i}]']+[f'{k} = {literal(v)}' for k,v in (condition|{'program_version':version}).items()]
    columns=sorted(legacy_rows[0]);lines += ['[width.4916]','matrix = [',
        '['+', '.join(map(literal,columns))+'],']
    lines += ['['+', '.join(literal(s[k]) for k in columns)+'],' for s in legacy_rows];lines += [']']
    legacy=out/'legacy_conditions.toml';legacy.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    migrated=tomllib.loads(run('--roundtrip',legacy));values=summaries(migrated)
    assert len(migrated['condition'])==1 and len(values)==1 and values[0]['repeats']==2
    assert math.isclose(values[0]['mean_seconds'],11) and math.isclose(values[0]['stddev_seconds'],math.sqrt(2))
    assert values[0]['program_version']=='0.1.0' and not values[0].get('updated_at'),'unknown old time was invented'
    canonical=copy.deepcopy(migrated['condition']['0']);different=canonical|{'uuid_hex':'1123456789abcdef0123456789abcdef'}
    separate=put('different_device.toml',[different],[late])
    isolated=tomllib.loads(run('--update',old,separate))
    assert len(isolated['condition'])==2 and len(summaries(isolated))==2
    # Conflict between D/B2 and carrier order, plus source and stable ties.
    assert [name for name,_ in ROW_ORDER['ecm']]==['condition','b2','d','carrier_exponent','b1','source']
    changes=[dict(condition=1),dict(b2=1400000000000,d=210),dict(b1=260000001),
        dict(carrier_exponent=5003),dict(d=210,carrier_exponent=5003),
        dict(d=210,source='model',fold_resident=0),dict(d=210),
        dict(d=210,frontier_resident=0),dict(d=210,b1=260000001)]
    order_rows=[]
    for i,change in enumerate(changes):
        row=sample('order-'+str(i),20-i,'0.0.'+str(i))
        row.update(benchmark_kind='synthetic_order',**change)
        if row['carrier_exponent']:row.update(arithmetic_bits=5003,modulus_kind='mersenne')
        order_rows.append(row)
    ordered=put('row_order.toml',[condition,different],order_rows)
    expected_versions=['0.0.'+str(i) for i in (6,7,5,8,4,2,3,1,0)]
    document=tomllib.loads(ordered.read_text(encoding='utf-8'))
    assert [s['program_version'] for s in summaries(document)]==expected_versions
    # Feed native code unsorted input rather than relying on Python's output order.
    header,*values=document['width']['4916']['matrix']
    values.sort(key=lambda row:int(row[header.index('program_version')].rsplit('.',1)[1]))
    prefix=ordered.read_text(encoding='utf-8').split('matrix = [',1)[0]
    unsorted=out/'row_unsorted.toml'
    unsorted.write_text(prefix+'matrix = [\n'+''.join(
        '['+', '.join(map(literal,row))+'],\n' for row in [header,*values])+']\n',encoding='utf-8')
    native=tomllib.loads(run('--roundtrip',unsorted))
    assert summaries(native)==summaries(document),'native/Python row order or data differs'
    # Metadata errors fail explicitly; the aggregate must not acquire an invalid clock.
    damaged=out/'bad_time.toml';damaged.write_text(old.read_text().replace('2023-11-14T22:13:20.000000001Z','not-a-time'),encoding='utf-8')
    assert subprocess.run([str(exe),'--roundtrip',str(damaged)],capture_output=True).returncode!=0
    report=dict(complete=True,shared_columns=True,native_python_order_equal=True,cross_version_statistics=True,
        latest_source_by_time=True,duplicate_no_refresh=True,legacy_condition_migration=True,
        unknown_time_preserved=True,hardware_isolation=True,invalid_time_rejected=True,
        shared_row_order=True,native_python_row_order=True,stable_ties=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
