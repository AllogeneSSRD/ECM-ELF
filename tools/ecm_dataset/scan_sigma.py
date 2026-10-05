"""Scan inclusive exponent/sigma ranges and retain one best sigma per factor."""
import argparse
import json
import math
import time
from dataset import DEFAULT_DB, analyze, connect, gp_path


def interval(text):
    """Accept FIRST:LAST or a single integer; never materialize the range."""
    pieces = text.split(':')
    if len(pieces) not in (1,2) or any(not s.isascii() or not s.isdigit() for s in pieces):
        raise argparse.ArgumentTypeError('Use FIRST:LAST (inclusive), or one integer')
    first, last = int(pieces[0]), int(pieces[-1])
    if first > last: raise argparse.ArgumentTypeError('FIRST must not exceed LAST')
    return first, last


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', default=DEFAULT_DB)
    p.add_argument('--exponent-range', type=interval, required=True, metavar='FIRST:LAST')
    p.add_argument('--sigma-range', type=interval, required=True, metavar='FIRST:LAST')
    p.add_argument('--gp', nargs='?', const=None, help='GP path/name; omitted or bare --gp searches PATH')
    p.add_argument('--timeout', type=float, default=30, help='Seconds per factor/sigma GP call')
    p.add_argument('--min-factor-digits', type=int, default=1)
    p.add_argument('--max-factor-digits', type=int, help='Optional decimal digit limit; default includes all factors')
    p.add_argument('--progress-seconds', type=float, default=5)
    args = p.parse_args()
    elo,ehi = args.exponent_range; slo,shi = args.sigma_range
    if not 1 <= elo <= ehi <= 9999: p.error('exponent range must be within 1..9999')
    if not 6 <= slo <= shi < 1 << 64: p.error('sigma range must be within 6..2^64-1')
    if args.min_factor_digits < 1: p.error('min-factor-digits must be positive')
    if args.max_factor_digits is not None and args.max_factor_digits < args.min_factor_digits:
        p.error('max-factor-digits must be at least min-factor-digits')
    if not all(math.isfinite(x) and x > 0 for x in (args.timeout,args.progress_seconds)):
        p.error('timeout and progress-seconds must be finite and positive')
    gp = gp_path(args.gp)
    db = connect(args.db)
    clauses = ['exponent BETWEEN ? AND ?', 'digital >= ?']
    values = [elo,ehi,args.min_factor_digits]
    if args.max_factor_digits is not None:
        clauses.append('digital <= ?'); values.append(args.max_factor_digits)
    where = ' AND '.join(clauses)
    count = db.execute('SELECT count(*) FROM factors WHERE '+where,values).fetchone()[0]
    stats = dict(selected_factors=count,planned=count*(shi-slo+1),completed=0,
                 updated=0,unchanged=0,cached=0,errors=0,timeouts=0,factors_finished=0)
    start = last_progress = time.monotonic()
    def report(kind, **extra):
        print(json.dumps(dict(type=kind,**stats,elapsed_seconds=time.monotonic()-start,**extra)),flush=True)
    report('start',exponent_range=[elo,ehi],sigma_range=[slo,shi],database=str(args.db))
    interrupted = False; failure = None
    try:
        # The query selects only immutable keys/digits; updates cannot alter its membership/order.
        rows = db.execute('SELECT exponent,value FROM factors WHERE '+where+
                          ' ORDER BY exponent,length(value),value',values)
        for row in rows:
            exponent,factor = row['exponent'],int(row['value'])
            for sigma in range(slo,shi+1):
                result = analyze(db,exponent,factor,sigma,gp,timeout=args.timeout)
                stats['completed'] += 1
                if result['status'] == 'complete':
                    stats['updated' if result['updated'] else 'unchanged'] += 1
                    stats['cached'] += int(result['cached'])
                else:
                    stats['timeouts' if result['status'] == 'timeout' else 'errors'] += 1
                now = time.monotonic()
                if now-last_progress >= args.progress_seconds:
                    report('progress',exponent=exponent,factor=str(factor),sigma=str(sigma))
                    last_progress = now
            stats['factors_finished'] += 1
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failure = str(error)
        raise
    finally:
        db.close()
        report('failed' if failure else 'interrupted' if interrupted else 'complete',error=failure)
    return 130 if interrupted else 0


if __name__ == '__main__': raise SystemExit(main())
