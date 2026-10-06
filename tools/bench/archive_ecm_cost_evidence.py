"""Archive completed Auto B2 calibration/validation, including failed scopes.

Records and raw sources are kept at their original repository-relative paths.
The executable is identified by SHA but is not embedded. Publication approval
comes from the audit, never from successful archive creation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'profile', 'blind', 'audit', 'stage2', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    study, model, blind, audit = map(read, (args.study, args.profile, args.blind, args.audit))
    if not study.get('complete') or not blind.get('complete'):
        raise ValueError('Both collections must be terminal and complete')
    if (model['source_sha256'] != sha(args.study) or blind['profile_sha256'] != sha(args.profile)
            or audit['study_sha256'] != sha(args.study) or audit['blind_sha256'] != sha(args.blind)
            or audit['profile_sha256'] != sha(args.profile) or not audit['integrity_passed']):
        raise ValueError('Complete integrity audit and unchanged model/evidence are required')
    if sha(args.stage2) != study['identity']['stage2_sha256']:
        raise ValueError('Executable differs from calibration')
    if args.output.suffix != '.zip' or args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Use a fresh .zip destination and manifest')
    files = {}

    def add(path, expected=None):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        digest = sha(path)
        if expected is not None and digest != expected.lower():
            raise ValueError('Input hash changed: ' + relative)
        if path == args.output.resolve():
            raise ValueError('Archive cannot include itself')
        if relative in files and files[relative]['sha256'] != digest:
            raise ValueError('Input changed during enumeration: ' + relative)
        files[relative] = dict(sha256=digest, bytes=path.stat().st_size)

    for path in (args.study, args.profile, args.blind, args.audit,
                 args.blind.parent/'predictions.json', Path(__file__)):
        add(path)
    for row in study['stage2'] + blind['runs']:
        log = Path(row['log']); add(log, row['log_sha256'])
        # Calibration uses a per-curve folder; validation uses a shared folder.
        siblings = (log.parent/'driver.log', log.parent/'result.jsonl') if 'case' not in row else (
            log.with_name(log.stem+'_driver.log'), log.with_suffix('.jsonl'))
        for path in siblings:
            add(path)
    for row in study['stage1']:
        cmd = row['command']; saved = Path(cmd[cmd.index('-save')+1])
        add(saved, row['save_sha256']); add(saved.parent/'driver.log'); add(saved.parent/'ecm.ini')
    for saved in study['saves'].values():
        add(saved['path'], saved['sha256'])
    if 'stage1_provenance' in study:
        add(study['stage1_provenance']['path'], study['stage1_provenance']['sha256'])
    binary_dir = args.stage2.resolve().parent
    # Calibration fingerprints the build receipt; validation fingerprints the
    # raw-source receipt. They are separate files despite the historical key.
    add(binary_dir/'build_manifest.json', study['identity']['build_manifest_sha256'])
    add(binary_dir/'frozen_sources_manifest.json', blind['build_manifest_sha256'])
    build = read(binary_dir/'frozen_sources_manifest.json')
    if (build['binary_sha256'] != sha(args.stage2)
            or {k: v.lower() for k, v in build['sources'].items()}
            != {k: v.lower() for k, v in study['identity']['sources'].items()}):
        raise ValueError('Calibration and validation source receipts differ')
    for name, digest in build['sources'].items():
        add(binary_dir/'sources'/name, digest)
    for name, digest in study['identity']['tools'].items():
        add(args.study.parent/'tools'/name, digest)
    frozen_dir = args.profile.parent/'fit_validation_sources'
    snapshot = read(frozen_dir/'manifest.json'); add(frozen_dir/'manifest.json')
    for name, digest in snapshot.items():
        add(frozen_dir/name, digest)
    for name, digest in blind['tools'].items():
        add(frozen_dir/name, digest)
    manifest = dict(schema=1, kind='completed_auto_b2_evidence_archive',
                    release_passed=audit['passed'], audit_sha256=sha(args.audit),
                    binary_embedded=False, binary_path=str(args.stage2.resolve()),
                    binary_sha256=sha(args.stage2), files=files,
                    note='Raw records retain original paths. Restore the repository-relative files under the original workspace for the current auditor.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrite; check each input again immediately before archiving.
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, info in sorted(files.items()):
            contents = (ROOT/name).read_bytes()
            if hashlib.sha256(contents).hexdigest() != info['sha256']:
                raise ValueError('Input changed during archiving: ' + name)
            archive.writestr(name, contents)
        archive.writestr('evidence_manifest.json', json.dumps(manifest, indent=2))
    manifest['archive_sha256'] = sha(args.output)
    manifest['archive_bytes'] = args.output.stat().st_size
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(files), archive=str(args.output),
                         sha256=manifest['archive_sha256'], release_passed=audit['passed'])))


if __name__ == '__main__':
    main()
