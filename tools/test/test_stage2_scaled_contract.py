"""Independent ordinary-coefficient contract; not the GPU implementation or a speed test."""
import argparse, json, random
from pathlib import Path

rng = random.Random(0xEC025CA1)
checked_states = checked_leaves = checked_words = 0

def mul(a, b, n):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % n
    return out

def rem(a, b, n):
    """Independent monic long division, no inverse or scaled recurrence."""
    assert b[-1] == 1
    out = a[:]
    for i in range(len(out) - 1, len(b) - 2, -1):
        q = out[i]
        for j in range(len(b)):
            out[i - len(b) + 1 + j] = (out[i - len(b) + 1 + j] - q * b[j]) % n
    return (out[:len(b) - 1] + [0] * (len(b) - 1))[:len(b) - 1]

def scale(h, f, n):
    """Triangular formal-series solve of rev(h) / rev(f), unit constant 1."""
    d = len(f) - 1
    rhs = list(reversed((h + [0] * d)[:d]))
    rev = list(reversed(f))
    out = []
    for i in range(d):
        out.append((rhs[i] - sum(rev[j] * out[i - j] for j in range(1, i + 1))) % n)
    return out

def tree(roots, n):
    if len(roots) == 1:
        r = roots[0]
        return {'f': [1] if r is None else [(-r) % n, 1], 'root': r}
    mid = len(roots) // 2
    left, right = tree(roots[:mid], n), tree(roots[mid:], n)
    return {'f': mul(left['f'], right['f'], n), 'left': left, 'right': right}

def walk(node, state, original_h, n):
    global checked_states, checked_leaves, checked_words
    f = node['f']; d = len(f) - 1
    remainder = rem(original_h, f, n)
    expected = scale(remainder, f, n)
    assert state == expected, (n, f, state, expected)
    checked_states += 1; checked_words += len(state)
    if 'root' in node:
        if node['root'] is not None:
            horner = 0
            for x in reversed(original_h): horner = (horner * node['root'] + x) % n
            assert state == [horner]
            checked_leaves += 1
        else:
            assert state == []
        return
    left, right = node['left'], node['right']
    a, b = len(left['f']) - 1, len(right['f']) - 1
    ls = mul(state, list(reversed(right['f'])), n)[b:b + a] if d else []
    rs = mul(state, list(reversed(left['f'])), n)[a:a + b] if d else []
    walk(left, ls, original_h, n)
    walk(right, rs, original_h, n)

cases = 0
for n in (15, 21, 35, 101, (1 << 64) - 59, (1 << 521) - 1):
    for degree in (1, 2, 3, 5, 7, 8, 9, 13, 16, 23):
        width = 1 << (degree - 1).bit_length()
        roots = [rng.randrange(n) for _ in range(degree)] + [None] * (width - degree)
        # Shuffle identity padding to cover empty subtrees on either side.
        rng.shuffle(roots)
        t = tree(roots, n)
        for kind in range(8):
            if kind == 0: h = [0] * degree
            elif kind == 1: h = [1] + [0] * (degree - 1)
            elif kind == 2: h = [n - 1] * degree
            else: h = [rng.randrange(n) for _ in range(degree)]
            walk(t, scale(h, t['f'], n), h, n)
            cases += 1
out = {'cases': cases, 'states': checked_states, 'leaves': checked_leaves,
       'state_words': checked_words, 'bad': 0,
       'scope': 'Python exact integer prototype, monic polynomials, arbitrary composite/prime modulus, '
                'unbalanced degrees, shuffled identity padding, zero/constant/full-degree H; '
                'every state checked by independent long division and every real leaf by Horner. '
                'No CUDA implementation, projective normalization, GMP fixture or production speed claim.'}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('build_cuda_cmake/_scaled_contract/summary.json'))
output = parser.parse_args().output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
