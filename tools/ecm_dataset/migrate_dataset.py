"""Migrate the legacy dataset to two core tables, preserving the selected sigma."""
import argparse
import json
from pathlib import Path
import sqlite3
from dataset import DATABASE_BUSY_TIMEOUT_SECONDS, DEFAULT_DB, CORE_COLUMNS, SCHEMA_SQL, digest

LEGACY_TABLES = {'sources','mersennes','factors','analyses','frontier','observations','production_runs'}


def migrate(path):
    path = Path(path).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    db = sqlite3.connect(path, timeout=DATABASE_BUSY_TIMEOUT_SECONDS)
    try:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = tuple(r[1] for r in db.execute('PRAGMA table_info(factors)'))
        if tables == {'mersennes','factors'} and columns == CORE_COLUMNS:
            return dict(status='already_core',database=str(path))
        if tables != LEGACY_TABLES or not set(CORE_COLUMNS).issubset(columns):
            raise ValueError('Unsupported schema; database left unchanged')
        backup = path.with_name(path.name+'-before-core-v2.bak')
        if backup.exists(): raise FileExistsError('Backup exists; refusing to overwrite: '+str(backup))
        # Backup through SQLite rather than copying possibly uncommitted WAL bytes.
        with sqlite3.connect(backup) as target: db.backup(target)
        before = list(db.execute('SELECT '+','.join(CORE_COLUMNS)+' FROM factors ORDER BY exponent,length(value),value'))
        counts = {t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in sorted(tables)}
        db.execute('PRAGMA foreign_keys=OFF')
        try:
            db.execute('BEGIN IMMEDIATE')
            factor_sql = SCHEMA_SQL[SCHEMA_SQL.index('CREATE TABLE IF NOT EXISTS factors'):]
            db.execute(factor_sql.replace('IF NOT EXISTS factors','factors_core'))
            names = ','.join(CORE_COLUMNS)
            db.execute('INSERT INTO factors_core('+names+') SELECT '+names+' FROM factors')
            for table in ('frontier','observations','production_runs','analyses','factors','sources'):
                db.execute('DROP TABLE '+table)
            db.execute('ALTER TABLE factors_core RENAME TO factors')
            db.execute('PRAGMA user_version=2')
            after = list(db.execute('SELECT '+names+' FROM factors ORDER BY exponent,length(value),value'))
            if before != after: raise ValueError('Core values changed; rolling back migration')
            if list(db.execute('PRAGMA foreign_key_check')): raise ValueError('Invalid core foreign keys')
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.execute('PRAGMA foreign_keys=ON')
        db.execute('VACUUM')  # actually reclaim pages occupied by discarded historical data
        return dict(status='migrated',before_counts=counts,
            after_counts={t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ('mersennes','factors')},
            selected_sigmas=db.execute('SELECT count(*) FROM factors WHERE sigma IS NOT NULL').fetchone()[0],
            backup=str(backup),database_bytes=path.stat().st_size,database_sha256=digest(path))
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=DEFAULT_DB)
    args = p.parse_args()
    print(json.dumps(migrate(args.db),indent=2))


if __name__ == '__main__': main()
