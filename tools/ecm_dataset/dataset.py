"""Core Mersenne/factor corpus: only one best PARAM0 sigma per factor."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
GP_SOURCE = Path(__file__).with_name('param0_order.gp')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DEFAULT_DB = ROOT/'tools/ecm_dataset/ecm_stage2_dataset.sqlite'
B2_RATIO = 5000
SCHEMA_VERSION = 2
DATABASE_BUSY_TIMEOUT_SECONDS = 3600
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mersennes(
  exponent INTEGER PRIMARY KEY,expression TEXT NOT NULL,digital INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS factors(
  exponent INTEGER NOT NULL REFERENCES mersennes,value TEXT NOT NULL,digital INTEGER NOT NULL,
  primality_verified INTEGER,sigma TEXT,b1 TEXT,b2 TEXT,group_order TEXT,
  group_factorization TEXT,point_order TEXT,point_factorization TEXT,
  PRIMARY KEY(exponent,value));
"""
CORE_COLUMNS = ('exponent','value','digital','primality_verified','sigma','b1','b2',
                'group_order','group_factorization','point_order','point_factorization')


def connect(path=DEFAULT_DB):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=DATABASE_BUSY_TIMEOUT_SECONDS)
    try:
        db.row_factory = sqlite3.Row
        db.execute(f'PRAGMA busy_timeout={DATABASE_BUSY_TIMEOUT_SECONDS * 1000}')
        db.execute('PRAGMA foreign_keys=ON')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and (tables != {'mersennes','factors'} or
                       tuple(r[1] for r in db.execute('PRAGMA table_info(factors)')) != CORE_COLUMNS):
            raise ValueError('Legacy/unsupported schema; migrate first with migrate_dataset.py --db PATH')
        db.execute('PRAGMA journal_mode=WAL')
        if not tables:
            db.executescript(SCHEMA_SQL)
        if db.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
            db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        return db
    except Exception:
        db.close()
        raise


def ensure_factor(db, n, factor):
    with db:
        db.execute('INSERT OR IGNORE INTO mersennes VALUES(?,?,?)', (n, f'2^{n}-1', len(str((1 << n)-1))))
        db.execute('INSERT OR IGNORE INTO factors(exponent,value,digital) VALUES(?,?,?)',
                   (n, str(factor), len(str(factor))))


def catalog(path):
    # Read literal digit-only link/TD attributes, never execute the HTML.
    current = None; pairs = set()
    for m in re.finditer(r'id="M(\d+)"|href="/factor/(\d+)"', Path(path).read_text(encoding='utf-8')):
        if m[1]: current = int(m[1])
        elif current: pairs.add((current, int(m[2])))
    if not pairs: raise ValueError('No Mersenne factor rows found')
    for n, f in sorted(pairs):
        if not 1 <= n <= 9999 or f < 2 or pow(2, n, f) != 1:
            raise ValueError(f'Invalid catalog divisor M{n}: {f}')
        yield n, f


def import_catalog(db, path):
    pairs = list(catalog(path))
    with db:
        for n, f in pairs:
            db.execute('INSERT OR IGNORE INTO mersennes VALUES(?,?,?)', (n, f'2^{n}-1', len(str((1 << n)-1))))
            db.execute('INSERT OR IGNORE INTO factors(exponent,value,digital) VALUES(?,?,?)',
                       (n, str(f), len(str(f))))
    return {'exponents': len({n for n, _ in pairs}), 'factors': len(pairs)}


def gp_path(explicit=None):
    # Port Stage1 normalize_gp_path/resolve_gp_path: explicit argument first,
    # otherwise locate gp through the PATH environment variable (no GP_PATH key).
    cleaned = str(explicit).strip() if explicit is not None else ''
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace('"','').replace("'",'')
    if cleaned:
        candidate = shutil.which(cleaned)
        if not candidate: raise FileNotFoundError('PARI/GP executable not found: '+cleaned)
    else:
        candidate = shutil.which('gp.exe') or shutil.which('gp')
        if not candidate: raise FileNotFoundError('PARI/GP not found on PATH; pass --gp PATH or add its directory to PATH')
    return Path(candidate).resolve()


def gp_call(gp, statement, timeout=30):
    body = GP_SOURCE.read_text(encoding='utf-8') + '\n' + statement + '\nquit();\n'
    start = time.monotonic()
    run = subprocess.run([str(gp), '-q', '-f'], input=body.encode(), capture_output=True, timeout=timeout)
    out = run.stdout.decode('utf-8', errors='replace').replace('\r', '')
    err = run.stderr.decode('utf-8', errors='replace').replace('\r', '')
    if run.returncode or not re.search(r'^OK$', out, re.M):
        raise ValueError('GP failed: ' + (err + out)[-1600:])
    return out, time.monotonic()-start


def parse_factors(text, tag):
    values = [(int(p), int(e)) for p, e in re.findall(r'^' + tag + r'\|(\d+)\|(\d+)$', text, re.M)]
    if any(p < 2 or e < 1 for p, e in values) or len({p for p, _ in values}) != len(values):
        raise ValueError('Invalid factorization')
    return sorted(values)


def product(parts):
    return math.prod(p**e for p, e in parts)


def bounds(parts, torsion=1):
    """Exact powersmooth lcm bound and nondominated single-prime Stage2 bounds.

    The point order, not #E alone, is authoritative. Stage2 removes one copy of
    one residual prime. Extra 2/3 valuations from choose12 are counted exactly.
    """
    if torsion not in (1, 12): raise ValueError('torsion must be 1 or 12')
    adjusted = []
    for p, e in parts:
        extra = 2 if torsion == 12 and p == 2 else 1 if torsion == 12 and p == 3 else 0
        adjusted.append((p, max(0, e-extra)))
    stage1 = max([2] + [p**e for p, e in adjusted if e])
    candidates = [(stage1, 0)]  # B2=0 in this dataset means Stage1 alone is sufficient.
    for left, eleft in adjusted:
        if not eleft: continue
        b1 = max([2] + [p**(e-(p == left)) for p, e in adjusted if e-(p == left)>0])
        if left > b1: candidates.append((b1, left))
    frontier = [a for a in sorted(set(candidates)) if not any(
        b != a and b[0] <= a[0] and b[1] <= a[1] for b in candidates)]
    return stage1, frontier


def factor_integer(gp, n, timeout=30):
    if n < 2: raise ValueError('factor must exceed 1')
    text, _ = gp_call(gp, f'ecm_factor({n});', timeout)
    parts = parse_factors(text, 'F')
    if product(parts) != n: raise ValueError('GP factor product mismatch')
    return parts


def bound_score(pair):
    """Product score, treating Stage1-only B2=0 as B1 squared."""
    b1, b2 = pair
    return B2_RATIO * b1 + max(b1, b2)


def store_best(db, n, factor, result, scan_policy=False):
    """Store one sigma using the requested scan replacement policy."""
    policy = 'either' if scan_policy is True else 'dominance' if scan_policy is False else scan_policy
    if policy not in ('dominance', 'strict', 'normal', 'either'):
        raise ValueError('Unknown sigma selection policy')
    # All durable comparisons use lcm. choose12 is a derived view of point order.
    _, pairs = bounds([(int(p), int(e)) for p, e in json.loads(result['point_factorization'])], 1)
    with db:
        # Serialize the comparison and replacement across simultaneous importers.
        db.execute('BEGIN IMMEDIATE')
        current = db.execute('SELECT * FROM factors WHERE exponent=? AND value=?', (n, str(factor))).fetchone()
        if current is None: raise ValueError('Factor must exist before storing its sigma')
        if current['sigma'] is not None:
            old = (int(current['b1']), int(current['b2']))
            if policy == 'strict':
                pairs = [pair for pair in pairs if pair[0] < old[0] and pair[1] < old[1]]
            elif policy == 'normal':
                pairs = [pair for pair in pairs if bound_score(pair) < bound_score(old) and
                         ((pair[0] < old[0] and pair[1] < old[1]) or pair[1] < 100000 * pair[0])]
                pairs = [min(pairs, key=lambda pair: (bound_score(pair), pair))] if pairs else []
            elif policy == 'either':
                pair = min(pairs)
                both_decrease = pair[0] < old[0] and pair[1] < old[1]
                either_decreases = pair[0] < old[0] or pair[1] < old[1]
                pairs = [pair] if both_decrease or (either_decreases and pair[1] < 100000 * pair[0]) else []
            else:
                pairs = [pair for pair in pairs if pair[0] <= old[0] and pair[1] <= old[1] and pair != old]
        elif policy != 'dominance':
            pairs = [pair for pair in pairs if pair[1] < 100000 * pair[0]]
            if policy == 'normal' and pairs:
                pairs = [min(pairs, key=lambda pair: (bound_score(pair), pair))]
        if not pairs:
            return False
        b1, b2 = min(pairs)
        db.execute("""UPDATE factors SET primality_verified=1,sigma=?,b1=?,b2=?,group_order=?,
                   group_factorization=?,point_order=?,point_factorization=? WHERE exponent=? AND value=?""",
                   (str(result['sigma']),str(b1),str(b2),result['group_order'],result['group_factorization'],
                    result['point_order'],result['point_factorization'],n,str(factor)))
    return True


def analyze(db, n, factor, sigma, gp, torsion=1, timeout=30, retry=False, prepared=None,
            scan_policy=False):
    if not 1 <= n <= 9999: raise ValueError('exponent must be 1..9999')
    if sigma < 6 or sigma > (1 << 64)-1: raise ValueError('sigma outside PARAM0 range')
    if torsion not in (1,12): raise ValueError('torsion must be 1 or 12')
    f = int(factor)
    if f < 2 or pow(2, n, f) != 1: raise ValueError('factor does not divide Mersenne')
    ensure_factor(db, n, f)
    old = db.execute('SELECT * FROM factors WHERE exponent=? AND value=?', (n,str(f))).fetchone()
    cached = old['sigma'] == str(sigma) and old['point_factorization'] is not None and not retry
    result = dict(exponent=n,value=str(f),sigma=str(sigma),torsion=torsion,status='complete',
                  updated=False,cached=cached)
    try:
        if cached:
            result.update({key:old[key] for key in ('group_order','group_factorization','point_order','point_factorization')})
        else:
            text, seconds = prepared if prepared is not None else gp_call(gp, f'ecm_param0_order({f},{sigma});', timeout)
            if not re.search(r'^OK$',text,re.M): raise ValueError('Incomplete GP order evidence')
            group_match = re.search(r'^GROUP\|(\d+)$', text, re.M)
            point_match = re.search(r'^POINT\|(\d+)$', text, re.M)
            if not group_match or not point_match: raise ValueError('Missing GP order evidence')
            group, point = int(group_match[1]), int(point_match[1])
            gf, pf = parse_factors(text, 'G'), parse_factors(text, 'P')
            if point < 1 or product(gf) != group or product(pf) != point or group % point:
                raise ValueError('Order factorization mismatch')
            result.update(group_order=str(group),point_order=str(point),seconds=seconds,
                          group_factorization=json.dumps([[str(p),e] for p,e in gf]),
                          point_factorization=json.dumps([[str(p),e] for p,e in pf]))
            # A proven prime stays proven even when its new sigma is not better.
            with db: db.execute('UPDATE factors SET primality_verified=1 WHERE exponent=? AND value=?',(n,str(f)))
            result['updated'] = store_best(db,n,f,result,scan_policy=scan_policy)
        s1,pairs = bounds([(int(p),int(e)) for p,e in json.loads(result['point_factorization'])],torsion)
        result.update(stage1_min_b1=str(s1),bounds=[[str(a),str(b)] for a,b in pairs])
    except (ValueError,TypeError,subprocess.TimeoutExpired) as error:
        result.update(status='timeout' if isinstance(error,subprocess.TimeoutExpired) else 'error',error=str(error))
    # Candidate data/errors exist only in the return value, never as historical DB rows.
    return result


def infer_exponent(n):
    if n < 3 or (n+1)&n: raise ValueError('Result N is not a full Mersenne; supply --exponent and verify its cofactor')
    return (n+1).bit_length()-1


def ingest(db, path, gp, torsion=1, timeout=30, exponent=None):
    counts = dict(records=0,raw_factors=0,no_factor=0,prime_candidates=0,updated=0,unchanged=0,unresolved=0,analysis_errors=0)
    # Stream results; neither file content nor dedup keys grow with the scan in the DB.
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if row.get('bad_factors',0): raise ValueError('Reject result with bad_factors')
            n = int(row['N_hex'],16)
            if n < 3: raise ValueError('Invalid result modulus')
            e = int(exponent if exponent is not None else row.get('mersenne_exponent') or infer_exponent(n))
            if not 1 <= e <= 9999 or ((1 << e)-1) % n:
                raise ValueError('Result modulus is not a divisor of the claimed Mersenne')
            sigma = int(row['sigma'])
            if not 6 <= sigma < 1 << 64: raise ValueError('sigma outside PARAM0 range')
            if int(row.get('param',0)) != 0: raise ValueError('Only PARAM0 result analysis is supported')
            counts['records'] += 1
            raw_factors = row.get('factors',[])
            if not raw_factors:
                counts['no_factor'] += 1
                continue
            seen = set()  # per result only, bounded by this result's factor count
            for raw in raw_factors:
                f = int(raw)
                if not 1 < f < n or n % f: raise ValueError('Invalid native factor')
                counts['raw_factors'] += 1
                try: parts = factor_integer(gp,f,timeout)
                except (ValueError,subprocess.TimeoutExpired):
                    counts['unresolved'] += 1
                    continue
                for p,_ in parts:
                    if p in seen: continue
                    seen.add(p);counts['prime_candidates'] += 1
                    result = analyze(db,e,p,sigma,gp,torsion,timeout)
                    if result['status'] != 'complete': counts['analysis_errors'] += 1
                    elif result['updated']: counts['updated'] += 1
                    else: counts['unchanged'] += 1
    return counts
