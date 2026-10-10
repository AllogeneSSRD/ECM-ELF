"""Verify native ECM tune merging without running GPU work."""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--valid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    text = args.valid.read_text(encoding='utf-8')
    original = tomllib.loads(text)
    base = out/'base.toml'
    base.write_text(text, encoding='utf-8')

    def merge(name, texts, success=True):
        paths = [base]
        for index, value in enumerate(texts):
            path = out/(name+str(index)+'.toml')
            path.write_text(value, encoding='utf-8')
            paths.append(path)
        proc = subprocess.run([str(args.fixture.resolve()), '--merge', *map(str, paths)],
                              capture_output=True, text=True, errors='replace', timeout=30)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr, encoding='utf-8')
        assert (proc.returncode == 0) == success, name
        if not success:
            return None
        profile = tomllib.loads(proc.stdout)
        destination = out/(name+'_merged.toml')
        destination.write_text(proc.stdout, encoding='utf-8')
        assert subprocess.run([str(args.fixture.resolve()), '--load', str(destination)],
                              capture_output=True).returncode == 0
        return profile

    single = merge('single', [])
    assert single['ecm'] == original['ecm']
    no_skip_count = merge('optional_skip_count', [text.replace('skipped = 0\n', '')])
    assert no_skip_count['summary']['skipped'] == 0
    duplicated = merge('duplicate', [text])
    assert duplicated['summary']['measured'] == 1
    assert duplicated['summary']['replaced_scopes'] == 1
    later = text.replace('seconds = [2.0, 3.0, 4.0]', 'seconds = [4.0, 6.0, 8.0]')
    later = later.replace('median_seconds = 3.0', 'median_seconds = 6.0')
    later = later.replace('mad_seconds = 1.0', 'mad_seconds = 2.0')
    replacement = merge('replacement', [later])
    assert replacement['ecm']['sample_0']['seconds'] == [4.0, 6.0, 8.0]
    wider = text.replace('target_bits = 318', 'target_bits = 319')
    combined = merge('additional_scope', [wider])
    assert combined['summary']['measured'] == 2
    assert {x['target_bits'] for x in combined['ecm'].values()} == {318, 319}
    legacy = text.replace('format = 3', 'format = 2').replace('max_batches = 64\n', '')
    legacy = legacy.replace('[policy.environment]\nxadd6 = 1', 'environment = "NTT_XADD6=1;"')
    compatible = merge('legacy', [legacy])
    assert compatible['profile']['format'] == 3
    assert compatible['policy']['environment'] == {'xadd6': 1}
    assert compatible['profile']['max_batches'] == 0
    for name, old, new in [
        ('device', 'sm_minor = 9', 'sm_minor = 6'),
        ('policy', 'batch_mb = 256', 'batch_mb = 64'),
        ('environment', 'xadd6 = 1', 'xadd6 = 0'),
    ]:
        merge('reject_'+name, [text.replace(old, new)], False)
    different_repeats = text.replace('repeats = 3', 'repeats = 2')
    different_repeats = different_repeats.replace('seconds = [2.0, 3.0, 4.0]', 'seconds = [2.0, 4.0]')
    merge('reject_repeats', [different_repeats], False)
    merge('reject_bad_statistics', [text.replace('median_seconds = 3.0', 'median_seconds = 9.0')], False)
    report = {'valid_merges': 6, 'rejected_merges': 5, 'native_reader_roundtrips': 6,
              'duplicate_scope_last_wins': True, 'format2_to_named_format3': True}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
