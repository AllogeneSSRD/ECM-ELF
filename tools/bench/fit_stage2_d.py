"""Fit positive per-phase D ranking rates; predictions are empirical seconds, not cycles."""
import argparse
import csv
import json
from pathlib import Path
from calibrate_stage2_d import features,parse,read_log

FEATURES={'baby':'baby','affine':'affine','ftree':'ftree','giant':'giant','gtrees':'gtrees',
          'fold':'fold','descent':'descent','inv':'inverse','accum':'accum','residual':'G'}

def fit(rows):
    return {k:max(0,sum(r['features'][f]*r['phases'][k] for r in rows)/
                  sum(r['features'][f]**2 for r in rows)) for k,f in FEATURES.items()}

def predict(f,rates):
    p={k:f[feat]*rates[k] for k,feat in FEATURES.items()}
    p['init']=p['baby']+p['affine']+p['ftree']
    p['main']=sum(p[k] for k in ('giant','gtrees','fold','descent','inv','accum','residual'))
    p['full']=p['init']+p['main']
    return p

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--anchor-csv',type=Path,nargs='*',default=[])
    p.add_argument('--anchor-measurements',type=Path,nargs='*',default=[],help='Verified same-binary phase measurements with identical controls')
    p.add_argument('--anchor-mode',choices=('6_mont','shape_outer','short_fold'),default='6_mont')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.measurements.read_text())
    rows=[r.copy() for r in m['runs'] if r['features']['G']>1]
    if not rows:raise ValueError('Resident fold fit needs measurements with G >= 2')
    profile=4 if m['env'].get('NTT_BABY_DEVICE')=='1' else 3 if m['env'].get('NTT_GL_SHORT_REDUCE')=='1' else 2 if m['env'].get('NTT_FUSE_COOP_OUTER')=='2' else 0
    if profile in (3,4) and m['env'].get('NTT_FUSE_COOP_OUTER')!='2':
        raise ValueError('Short-reducer fit requires shape policy mode 2')
    if profile==4 and (m['env'].get('NTT_GL_SHORT_REDUCE')!='1' or a.anchor_csv):
        raise ValueError('GPU baby fit requires short reducer; CPU baby CSV anchors are forbidden')
    if profile==2 and a.anchor_csv and a.anchor_mode!='shape_outer':
        raise ValueError('Original-NTT xADD anchors cannot enter a shape-policy fit')
    if profile==3 and a.anchor_csv and a.anchor_mode!='short_fold':
        raise ValueError('Short-reducer fit needs short_fold anchors')
    for row in rows:
        cmd=row['command'];b2=int(cmd[cmd.index('--b2')+1]);bits=int(cmd[cmd.index('--n-hex')+1],16).bit_length()
        row['features']=features(row['features']['D'],b2,bits,profile)
        if profile==4:
            text=read_log(row['log'])
            for token in ('baby_device: requested=1 enabled=1','gmp_selftest_bad=0',
                          'gmp_check_bad=0','pending=0','clean=1',m['Q_line']):
                if token not in text:raise ValueError('Invalid GPU baby measurement: '+token)
            if parse(text)!=row['phases']:raise ValueError('Measurement phases differ from raw log')
    for path in a.anchor_measurements:
        anchor=json.loads(path.read_text())
        clean=lambda env:{k:str(v) for k,v in env.items() if v is not None}
        if (anchor['sha256'].lower()!=m['sha256'].lower() or anchor['device']!=m['device'] or
            anchor['Q_line']!=m['Q_line'] or clean(anchor['env'])!=clean(m['env']) or
            anchor.get('sources')!=m.get('sources')):
            raise ValueError('Measurement anchors differ in binary/device/Q/controls/sources')
        for row in anchor['runs']:
            cmd=row['command'];d=int(cmd[cmd.index('--d')+1]);b2=int(cmd[cmd.index('--b2')+1])
            bits=int(cmd[cmd.index('--n-hex')+1],16).bit_length();f=features(d,b2,bits,profile)
            if f['G']<2:raise ValueError('Anchor is outside resident fold scope')
            text=read_log(row['log'])
            for token in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1',m['Q_line']):
                if token not in text:raise ValueError('Invalid measurement anchor: '+token)
            if profile==4 and 'baby_device: requested=1 enabled=1' not in text:
                raise ValueError('CPU baby or fallback anchor cannot enter GPU baby fit')
            phases=parse(text)
            if phases!=row['phases']:raise ValueError('Anchor phase data differs from raw log')
            rows.append(dict(name=path.parent.name+'/'+row['name'],log=row['log'],command=cmd,features=f,phases=phases))
    for path in a.anchor_csv:
        prov=json.loads((path.parent/'provenance.json').read_text(encoding='utf-8-sig'))
        if profile in (2,3) and prov['sha256'].lower()!=m['sha256'].lower():
            raise ValueError('Shape anchors must use the same measured binary')
        controls=next(r for r in prov['mode_controls'] if r['mode']==a.anchor_mode)
        if profile in (2,3) and controls.get('NTT_FUSE_COOP_OUTER')!='2':
            raise ValueError('Shape anchors must enable the same NTT profile')
        if profile==3 and controls.get('NTT_GL_SHORT_REDUCE')!='1':
            raise ValueError('Short anchors must enable the short reducer')
        argv=prov['args'];d=int(argv[argv.index('--d')+1]);b2=int(argv[argv.index('--b2')+1])
        bits=int(argv[argv.index('--n-hex')+1],16).bit_length()
        if profile==3:
            anchor_env={k:str(v) for k,v in prov['env'].items() if v is not None}
            anchor_env.update({k:str(v) for k,v in controls.items() if k!='mode'})
            for key,value in m['env'].items():
                if value is not None and anchor_env.get(key)!=str(value):
                    raise ValueError('Anchor configuration differs: '+key)
            if int(argv[argv.index('--device')+1])!=m['device']:
                raise ValueError('Anchor device differs')
        for row in csv.DictReader(path.open(encoding='utf-8-sig')):
            if row['mode']!=a.anchor_mode:continue
            text=read_log(row['log'])
            if profile==3:
                for token in (f"ntt_gl_reduce_mode: device={m['device']} short=1",'gmp_selftest_bad=0',
                              'gmp_check_bad=0','pending=0','point_arithmetic: xadd6=1'):
                    if token not in text:raise ValueError('Invalid short anchor: '+token)
                if m['Q_line'] not in text:raise ValueError('Anchor Stage1 Q differs')
            rows.append(dict(name=str(path.parent.name)+'/'+row['run'],log=row['log'],
                             features=features(d,b2,bits,profile),phases=parse(text)))
    rates=fit(rows)
    result={'exe':m['exe'],'sha256':m['sha256'],'device':m['device'],'env':m['env'],'rates':rates,'feature_profile':profile,
            'scope':'RTX4060 Laptop sm89, exact M4423, resident pipeline, xADD6, warp tail, batch64/chain64',
            'equations':FEATURES,'runs':[]}
    for r in rows:
        pred=predict(r['features'],rates)
        held=[x for x in rows if x['features']['D']!=r['features']['D']]
        cross=predict(r['features'],fit(held)) if held else None
        result['runs'].append(dict(name=r['name'],features=r['features'],actual=r['phases'],prediction=pred,
                                  full_error_percent=100*(pred['full']/r['phases']['full']-1),
                                  leave_D_out_error_percent=100*(cross['full']/r['phases']['full']-1) if cross else None))
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'rates':rates,'full_errors':[r['full_error_percent'] for r in result['runs']],
                      'leave_D_out_errors':[r['leave_D_out_error_percent'] for r in result['runs']]},indent=2))

if __name__=='__main__':main()
