"""Export the two core dataset tables without changing the database."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
from dataset import ROOT, digest

TABLES = ('mersennes', 'factors')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=ROOT/'tools/ecm_dataset/ecm_stage2_dataset.sqlite')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory to preserve previous snapshots')
    args.output.mkdir(parents=True, exist_ok=True)
    sha = digest(args.db)
    db = sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    counts = {}
    for table in TABLES:
        rows = [dict(row) for row in db.execute('SELECT * FROM '+table+' ORDER BY rowid')]
        counts[table] = len(rows)
        write_jsonl(args.output/(table+'.jsonl'), rows)
        if table == 'factors':
            with (args.output/'factors.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[x[1] for x in db.execute('PRAGMA table_info(factors)')])
                writer.writeheader(); writer.writerows(rows)
    db.close()
    assert digest(args.db) == sha, 'Database changed during export'
    manifest = dict(schema=2, database_sha256=sha, counts=counts,
        bound_model='PARAM0_point_order_single_prime_semismooth',
        dataset_b2_zero='Stage1_only_not_native_auto_B2',
        exporter_sha256=digest(__file__), files={p.name:digest(p) for p in sorted(args.output.iterdir())})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(counts=counts, database_sha256=sha), indent=2))


if __name__ == '__main__':
    main()
