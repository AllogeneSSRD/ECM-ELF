"""Export legacy completed Stage2 receipts without submitting or altering inputs.

Target metadata must exactly reconstruct N. Duplicate/covered curves are removed
before adjacent runs are merged. Private payloads belong only in ignored data.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import zlib

import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from result_format import normalize_result

ROOT=Path(__file__).resolve().parents[2]


def public_version():
    text=(ROOT/'src/core/ecm_program_version.h').read_text(encoding='utf-8')
    return re.search(r'version\[\]\s*=\s*"([0-9.]+)"',text)[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def composite_witness(n):
    if n<4:return None
    if n%2==0:return 2
    d=n-1;s=0
    while d%2==0:d//=2;s+=1
    for a in (2,3,5,7,11,13,17):
        if a>=n:continue
        x=pow(a,d,n)
        if x in (1,n-1):continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1:break
        else:return a
    return None


def target_catalog(path):
    targets={}
    for r in json.loads(Path(path).read_text(encoding='utf-8-sig')):
        n=int(r['N_hex'],16);p=int(r['exponent'])
        if not 2<=p<=16384 or n<4:raise ValueError('Target outside current converter scope')
        m=(1<<p)-1;known=list(map(str,r['known_factors']))
        for text in known:
            if not text.isdecimal() or int(text)<=1 or m%int(text):raise ValueError('Invalid known factor')
            m//=int(text)
        if m!=n:raise ValueError('Target metadata does not reconstruct N')
        witness=composite_witness(n)
        if witness is None:raise ValueError('No compositeness witness; review target before export')
        entry=dict(exponent=p,known=known,witness=witness)
        if n in targets and targets[n]!=entry:raise ValueError('Ambiguous target catalog')
        targets[n]=entry
    return targets


def ini_identity(path,worker):
    if not path:return {}
    scope=0;global_values={};worker_values={}
    for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line=raw.strip()
        if not line or line.startswith(('#',';')):continue
        m=re.fullmatch(r'\[Worker\s*#(\d+)\]',line,re.I)
        if m:scope=int(m[1]);continue
        if line.startswith('[') or '=' not in line:continue
        key,value=map(str.strip,line.split('=',1));key=key.lower()
        if key in ('user','computer'):
            if scope==0:global_values[key]=value
            elif scope==worker:worker_values[key]=value
    return global_values|worker_values


def reference_fft(p):
    table=(ROOT/'src/core/generated/p95_fft_reference.h').read_text(encoding='utf-8')
    for maximum,length in re.findall(r'\{(\d+)ull,(\d+)u\}',table):
        if p<=int(maximum):return int(length)
    raise ValueError('No FFT capacity reference')


def checksum(row):
    ed=row.get('Edwards');sigma=('E' if ed is not None else '')+str((ed or {}).get('sigma',row.get('sigma','')))
    fields=[str(row['exponent']),'ECM',','.join(row.get('factors',[])),','.join(row.get('known-factors',[])),
            str(row['b1']),str(row['b2']) if row.get('b2',0)>row['b1'] else '',sigma,str(row['fft-length']),'invariant:0',
            row['program']['name'],row['program']['version'],'','',row['os']['os'],row['os']['architecture'],row['timestamp']]
    return dict(version=1,checksum=f'{zlib.crc32(";".join(fields).encode("utf-8")):08X}')


def geometry_logs(paths):
    found={}
    for path in paths:
        current=None;save=None;curve=None;real=None
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            m=re.search(r'stage2_plan: save=(.*?) requested=',line)
            if m:save=m[1].strip()
            m=re.search(r'curve_start: .* record=(\d+) sigma=(\d+) B1=(\d+)',line)
            if m:curve=tuple(map(int,m.groups()));current=None;real=None
            m=re.search(r'stage2_real: .* N_bits=(\d+) sigma=(\d+) B1=(\d+) B2=(\d+)',line)
            if m:real=tuple(map(int,m.groups()))
            m=re.search(r'real_shape: D=(\d+) P=phi\(D\)/2=(\d+).* B1=(\d+) B2=(\d+) S_bits=(\d+)',line)
            if m:current=tuple(map(int,m.groups()))
            m=re.search(r'stage2_result: record=(\d+).* sigma=(\d+) B1=(\d+).* save=("(?:[^"\\]|\\.)*")',line)
            if m and current and int(m[3])==current[2]:
                save=json.loads(m[4]);key=(Path(save).name.lower(),int(m[1]),int(m[2]),int(m[3]),current[3])
                found.setdefault(key,set()).add((current[0],current[1],current[4]))
                current=None
            else:
                m=re.search(r'stage2_result: record=(\d+)',line)
                if m and current and save and curve and real and int(m[1])==curve[0] and curve[1:]==real[1:3] and current[2:4]==real[2:4]:
                    key=(Path(save).name.lower(),curve[0],curve[1],curve[2],real[3])
                    found.setdefault(key,set()).add((current[0],current[1],current[4]));current=None
    return found


def convert(rows,targets,legacy_version,identity,geometry=None):
    if not re.fullmatch(r'\d+(?:\.\d+)+',legacy_version):raise ValueError('Numeric dotted version required')
    receipts=[];best={};migration=0
    for line,r in rows:
        if r['status']!='stage2_completed' or r.get('bad_factors') or r.get('hits') or r.get('factors'):
            raise ValueError('Only complete, error-free NF receipts are supported')
        if r.get('factorization_complete') is False:raise ValueError('Incomplete receipt')
        n=int(r['N_hex'],16)
        if n not in targets:raise ValueError('No exact target metadata')
        model='ed' if r.get('curve')=='ed' else 'mont'
        param=None if model=='ed' else r.get('param')
        if model=='mont' and param not in (0,3):raise ValueError('Unknown Montgomery parameterization')
        b1,b2,sigma=map(int,(r['B1'],r['B2'],r['sigma']))
        if not 2<=b1<b2 or not 0<sigma<1<<64:raise ValueError('Invalid bounds/sigma')
        if r.get('curves',1)!=1:raise ValueError('Individual curve receipt required')
        if r.get('program',{}).get('name','ECM-ELF')!='ECM-ELF':raise ValueError('Unexpected source program identity')
        version=r.get('program',{}).get('version','')
        if not re.fullmatch(r'\d+(?:\.\d+)+',version):version=legacy_version;migration+=1
        key=(n,sigma,b1,model,param)
        receipt=dict(line=line,raw=r,n=n,b1=b1,b2=b2,sigma=sigma,model=model,param=param,version=version,key=key)
        receipts.append(receipt)
        if key not in best or b2>best[key]['b2']:best[key]=receipt
    selected={r['line'] for r in best.values()};batches=[];dropped=[]
    for item in receipts:
        r=item['raw']
        if item['line'] not in selected:
            reason='covered-by-larger-B2' if item['b2']<best[item['key']]['b2'] else 'duplicate'
            dropped.append(dict(line=item['line'],reason=reason,retained_line=best[item['key']]['line']));continue
        group=(item['n'],item['b1'],item['b2'],item['model'],item['param'],item['version'],r['worker'])
        if batches:
            previous=batches[-1][-1]
            adjacent=item['line']==previous['line']+1 and r['record']==previous['raw']['record']+1
        else:adjacent=False
        if batches and batches[-1][0]['group']==group and adjacent:batches[-1].append(item)
        else:item['group']=group;batches.append([item])
        # Keep the group on each member for subsequent adjacency checks.
        item['group']=group
    output=[];audit_batches=[]
    for batch in batches:
        first,last=batch[0],batch[-1];target=targets[first['n']]
        timestamp=datetime.fromtimestamp(last['raw']['timestamp_ms']/1000,timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        row=dict(status='NF',exponent=target['exponent'],worktype='ECM',b1=first['b1'],b2=first['b2'])
        plans=[]
        for item in batch:
            raw=item['raw'];key=(Path(raw['save']).name.lower(),raw['record'],item['sigma'],item['b1'],item['b2'])
            options=(geometry or {}).get(key,set())
            allowed_bits=(item['n'].bit_length(),targets[item['n']]['exponent'])
            options={(d,p) for d,p,bits in options if bits in allowed_bits}
            # New receipts carry the completed engine geometry directly. A
            # requested_D remains insufficient, and conflicting evidence is omitted.
            d,p=raw.get('d'),raw.get('poly-size')
            if d is not None or p is not None:
                if type(d) is not int or type(p) is not int or d<6 or d%2 or p<1:
                    raise ValueError('Invalid completed Stage2 geometry')
                options.add((d,p))
            plans.append(next(iter(options)) if len(options)==1 else None)
        if plans and plans[0] is not None and all(x==plans[0] for x in plans):
            row['d'],row['poly-size']=plans[0]
        row['curves']=len(batch)
        if first['model']=='ed':row['Edwards']={}
        row.update({'fft-length':reference_fft(target['exponent']),
                    'program':dict(name='ECM-ELF',version=first['version'],port=4),'timestamp':timestamp,
                    'known-factors':target['known'],'errors':dict(invariant=0),'os':dict(os='Windows',architecture='x86_64')})
        row.update({k:v for k,v in identity.items() if k in ('user','computer') and v})
        row['checksum']=checksum(row)
        if len(json.dumps(row,separators=(',',':'),ensure_ascii=False).encode('utf-8'))>=2000:raise ValueError('Submission line exceeds capacity')
        output.append(row);audit_batches.append(dict(source_lines=[x['line'] for x in batch],exponent=row['exponent'],
            b1=row['b1'],b2=row['b2'],curves=row['curves'],model=first['model'],param=first['param'],
            geometry='verified_constant' if 'd' in row else 'mixed' if None not in plans else 'incomplete_or_ambiguous'))
    return output,dict(input_receipts=len(receipts),unique_curves=len(best),output_batches=len(output),
        exact_unique=len({(r['key'],r['b2']) for r in receipts}),legacy_version=legacy_version,version_migration_rows=migration,
        version_note='Migration label for missing/non-numeric versions; exact original per-receipt build is not reconstructed',
        dropped=dropped,batches=audit_batches,
        target_witnesses=[dict(exponent=r['exponent'],miller_rabin_witness=r['witness']) for r in targets.values()])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--legacy-version',default=public_version());p.add_argument('--ini',type=Path)
    p.add_argument('--geometry-logs',type=Path,nargs='*',default=[])
    p.add_argument('--worker',type=int,default=1)
    a=p.parse_args()
    paths=[x.resolve() for x in (a.input,a.targets,a.output,a.audit)]
    if len(set(paths))!=4 or a.output.exists() or a.audit.exists():raise ValueError('Use fresh, distinct output paths')
    raw=a.input.read_bytes();rows=[(i,normalize_result(json.loads(s))) for i,s in enumerate(raw.decode('utf-8-sig').splitlines(),1) if s.strip()]
    if not rows:raise ValueError('No receipts')
    if any(b[1]['timestamp_ms']<a[1]['timestamp_ms'] for a,b in zip(rows,rows[1:])):raise ValueError('Input receipt times are not ordered')
    output,audit=convert(rows,target_catalog(a.targets),a.legacy_version,ini_identity(a.ini,a.worker),geometry_logs(a.geometry_logs))
    audit.update(input_sha256=hashlib.sha256(raw).hexdigest(),targets_sha256=digest(a.targets),script_sha256=digest(__file__),
                 fft_reference_sha256=digest(ROOT/'src/core/generated/p95_fft_reference.h'),
                 identity_source=str(a.ini.resolve()) if a.ini else None,
                 geometry_logs={str(path.resolve()):digest(path) for path in a.geometry_logs},
                 geometry_batches=sum('d' in r for r in output),
                 omitted_stage2_fft='GPU NTT pipeline has multiple transform sizes; no Prime95 Stage2 FFT length is claimed',complete=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.audit.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    audit['output_sha256']=digest(a.output);a.audit.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print('Converted',len(rows),'receipts to',len(output),'batches;',sum(r['curves'] for r in output),'unique curves. No submission performed.')


if __name__=='__main__':main()
