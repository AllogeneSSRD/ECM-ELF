"""Verify scoped B2 estimates, holdout gates and non-extrapolation natively."""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib


def literal(value):
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return '['+', '.join(map(literal, value))+']'
    return repr(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--valid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    template = tomllib.loads(args.valid.read_text(encoding='utf-8'))
    counts = {'accepted': 0, 'ineligible': 0, 'invalid_profiles': 0}

    def invoke(name, bounds, query, time=lambda i: .1+.00001*i, mutate=None, expected=True):
        data = json.loads(json.dumps(template))
        data['profile']['prediction_model'] = 'linear_giant_points_v1'
        samples = []
        for b2 in bounds:
            sample = data['ecm']['sample_0'].copy()
            sample['b2'] = b2
            sample['giant_points'] = b2//sample['d']+2
            seconds = time(sample['giant_points'])
            sample.update(seconds=[seconds-.001, seconds, seconds+.001],
                          median_seconds=seconds, mad_seconds=.001)
            for key,fraction in {'init_seconds':.2,'main_seconds':.8,'giant_seconds':.12,
                                 'gtrees_seconds':.4,'fold_seconds':.12,'descent_seconds':.05,
                                 'inverse_seconds':.05,'accum_seconds':.01}.items():
                sample[key] = seconds*fraction
            samples.append(sample)
        data['summary']['measured'] = len(samples)
        if mutate:
            mutate(data, samples)
        tables = [('profile', data['profile']), ('device', data['device']),
                  ('policy', {k:v for k,v in data['policy'].items() if k!='environment'}),
                  ('policy.environment', data['policy']['environment'])]
        tables += [('ecm.sample_'+str(i), s) for i,s in enumerate(samples)]
        tables += [('summary', data['summary'])]
        path = out/(name+'.toml')
        path.write_text(''.join('\n['+section+']\n'+''.join(k+' = '+literal(v)+'\n' for k,v in fields.items())
                                for section,fields in tables), encoding='utf-8')
        proc = subprocess.run([str(args.fixture.resolve()), '--predict', str(path), str(query)],
                              capture_output=True, text=True, errors='replace', timeout=30)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr, encoding='utf-8')
        if expected is None:
            assert proc.returncode != 0, name
            counts['invalid_profiles'] += 1
            return
        assert proc.returncode == 0, (name,proc.stderr)
        prediction = json.loads(proc.stdout)
        assert prediction['eligible'] == expected, (name,prediction)
        counts['accepted' if expected else 'ineligible'] += 1
        return prediction

    bounds = [26000000000, 52000000000, 104000000000]
    query = 78000000000
    prediction = invoke('linear', bounds, query)
    assert abs(prediction['seconds']-(.1+.00001*(query//180180+2))) < 1e-12
    assert prediction['max_relative_error'] < 1e-12
    assert abs(prediction['mad_seconds']-.001) < 1e-12
    assert prediction['samples'] == 3
    invoke('insufficient', bounds[:2], query, expected=False)
    for name, query in [('below',1),('above',104000000001),('low_endpoint',bounds[0]),('high_endpoint',bounds[-1])]:
        invoke(name,bounds,query,expected=False)
    query = 78000000000
    invoke('negative_fixed',bounds,query,time=lambda i:-10+.0001*i,expected=False)
    invoke('negative_rate',bounds,query,time=lambda i:10-.00001*i,expected=False)
    def outlier(data,samples):
        s=samples[1];s['median_seconds']*=1.2
        s['seconds']=[s['median_seconds']-.001,s['median_seconds'],s['median_seconds']+.001]
    invoke('holdout_error',bounds,query,mutate=outlier,expected=False)
    invoke('nonresident',bounds,query,mutate=lambda d,s:s[1].update(fold_resident=0),expected=False)
    invoke('different_width',bounds,query,mutate=lambda d,s:s[1].update(target_bits=319),expected=False)
    invoke('missing_optin',bounds,query,mutate=lambda d,s:d['profile'].pop('prediction_model'),expected=False)
    invoke('unsupported_model',bounds,query,mutate=lambda d,s:d['profile'].update(prediction_model='unknown'),expected=None)
    invoke('insufficient_span',[26000000000,30000000000,34000000000],31000000000,expected=False)
    # Distinct B2 values can encode exactly the same integer giant count.
    invoke('duplicate_giant_count',[26000000000,26000000001,104000000000],query,expected=False)
    report = dict(counts, model='linear_giant_points_v1', leave_one_out=True,
                  endpoint_extrapolation_refused=True, negative_costs_refused=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
