"""Read compact CUDA result records; expose legacy aliases only in memory."""


def decimal_integer(value):
    if not isinstance(value,str) or len(value)<=4000:
        return int(value)
    # 16384-bit targets reach4933 decimal digits. Parse canonical integer
    # chunks without disabling Python's process-wide resource guard.
    text=value.strip();sign=1
    if text.startswith(('+','-')):
        if text[0]=='-':sign=-1
        text=text[1:]
    if not text.isascii() or not text.isdecimal():raise ValueError('Invalid decimal result integer')
    result=0
    for i in range(0,len(text),512):
        chunk=text[i:i+512];result=result*10**len(chunk)+int(chunk)
    return sign*result


def modulus(row):
    if 'exponent' in row:
        n = (1 << int(row['exponent'])) - 1
    elif 'k' in row:
        n = int(row['k']) * int(row['b']) ** int(row['n']) + int(row['c'])
    elif 'N' in row:
        return decimal_integer(row['N'])
    else:
        raise ValueError('Result has no exact target identity')
    for factor in row.get('known-factors', []):
        factor = decimal_integer(factor)
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
