"""Read compact CUDA result records; expose legacy aliases only in memory."""


def modulus(row):
    if 'exponent' in row:
        n = (1 << int(row['exponent'])) - 1
    elif 'k' in row:
        n = int(row['k']) * int(row['b']) ** int(row['n']) + int(row['c'])
    elif 'N' in row:
        return int(row['N'])
    else:
        raise ValueError('Result has no exact target identity')
    for factor in row.get('known-factors', []):
        factor = int(factor)
        if factor <= 1 or n % factor:
            raise ValueError('Result known factor does not divide original target')
        n //= factor
    if n <= 1:
        raise ValueError('Invalid result modulus')
    return n


def normalize_result(row):
    if not isinstance(row, dict):
        return row
    engine = row.get('stage2-engine', {})
    if engine.get('name') != 'cuda-ntt':
        return row
    result = dict(row)
    result.update(engine)
    result.update(N_hex=format(modulus(row), 'x'), B1=row['b1'],
                  B2=row.get('b2', engine.get('requested_B2', row['b1'])),
                  curve='ed' if 'Edwards' in row else 'mont',
                  param=None if 'Edwards' in row else row.get('param', 0),
                  sigma=row.get('Edwards', {}).get('sigma', row.get('sigma')))
    result.setdefault('factors', [])
    result.setdefault('hits', len(result['factors']))
    result.setdefault('bad_factors', 0)
    for key in ('relocation', 'relocation_replayed', 'auto_b2', 'requested_factor_only'):
        result.setdefault(key, False)
    result.setdefault('factorization_seconds', 0)
    return result
