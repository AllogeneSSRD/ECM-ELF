"""Check a native Edwards NAF schedule proposed for the GPU Stage1 NTT backend.

Python integers model the modular product batches. The independent endpoint
oracle uses the production Montgomery x-only ladder's coordinate convention.
"""

import math

from test_edwards_gpu_pipeline import curve, edwards_coordinates, mont


def naf(scalar, window):
    digits = []
    radix = 1 << window
    while scalar:
        digit = scalar % radix if scalar % 2 else 0
        if digit >= radix // 2:
            digit -= radix
        digits.append(digit)
        scalar = (scalar - digit) // 2
    return digits[::-1]


def double(point, n, extended=False):
    x, y, z = point
    a, b, c, s = (v*v % n for v in (x, y, z, x+y))
    e, f, g, h = s-a-b, a+b-2*c, a+b, a-b
    result = (e*f % n, g*h % n, f*g % n)
    return (*result, e*h % n) if extended else result


def dictionary(point, d, n, window):
    """Construct projective odd multiples and normalize if the product of Zs is a unit."""
    points = [point]
    x2, y2, z2, t2 = double(point[:3], n, extended=True)
    for _ in range(1, 1 << (window-2)):
        x, y, z, t = points[-1]
        a, b, c, q = x*x2 % n, y*y2 % n, d*t*t2 % n, z*z2 % n
        e = ((x+y)*(x2+y2)-a-b) % n
        f, g, h = q-c, q+c, b-a
        points.append((e*f % n, g*h % n, f*g % n, e*h % n))

    prefix = [1]
    for _, _, z, _ in points:
        prefix.append(prefix[-1]*z % n)
    factor = math.gcd(prefix[-1], n)
    if factor != 1:
        factors = [math.gcd(z, n) for _, _, z, _ in points] if factor == n else [factor]
        return None, next((g for g in factors if 1 < g < n), n)

    inverse = pow(prefix[-1], -1, n)
    table = []
    for i in range(len(points)-1, -1, -1):
        x, y, z, _ = points[i]
        iz = prefix[i]*inverse % n
        inverse = inverse*z % n
        x, y = x*iz % n, y*iz % n
        table.append((x, y, d*x*y % n, (x+y) % n))
    return table[::-1], 1


def step(point, digit, table, n):
    if not digit:
        return double(point, n)

    x, y, z, t = double(point, n, extended=True)
    qx, qy, dqt, qsum = table[abs(digit)//2]
    a, b, c = x*qx % n, y*qy % n, t*dqt % n
    if digit > 0:
        s = (x+y)*qsum % n
        e, f, g, h = s-a-b, z-c, z+c, b-a
    else:
        s = (y-x)*qsum % n
        e, f, g, h = b-a-s, z+c, z-c, b+a
    return e*f % n, g*h % n, f*g % n


def valid_point(point, d, n):
    x, y, z = point
    return any(point) and ((x*x+y*y)*z*z-z**4-d*x*x*y*y) % n == 0


def check_case(n, sigma, b1, window):
    d, x, y = edwards_coordinates(n, sigma)
    base = x, y, 1, x*y % n
    a24, mx = curve(n, sigma)
    scalar = 48*mont.lcm_1_to(b1)
    digits = naf(scalar, window)
    signed = 0
    for digit in digits:
        signed = 2*signed + digit
    assert signed == scalar
    nonzero = [i for i, digit in enumerate(digits) if digit]
    assert all(digit & 1 and abs(digit) < 1 << (window-1) for digit in digits if digit)
    assert all(b-a >= window for a, b in zip(nonzero, nonzero[1:]))

    table, factor = dictionary(base, d, n, window)
    if factor != 1:
        if factor != n:
            assert n % factor == 0
        return 'dictionary-factor' if factor != n else 'dictionary-degenerate', len(digits), len(nonzero)
    point = (0, 1, 1)
    for digit in digits:
        point = step(point, digit, table, n)
        assert valid_point(point, d, n)

    # Each split stores only X/Y/Z. Include cuts immediately after negative
    # additions and after zero columns, with no cached T in either case.
    split = 1 + next((i for i, digit in enumerate(digits) if digit < 0), 0)
    resumed = (0, 1, 1)
    for part in (digits[:split], digits[split:split+1], digits[split+1:]):
        for digit in part:
            resumed = step(resumed, digit, table, n)
        resumed = tuple(resumed)
    assert resumed == point

    _, y, z = point
    xz = (z+y) % n, (z-y) % n
    mx, mz = mont.ladder(scalar, mx, 1, a24, n)
    expected_factor = math.gcd(mz, n)
    assert math.gcd(xz[1], n) == expected_factor
    assert (xz[0]*mz-xz[1]*mx) % n == 0
    return 'factor' if 1 < expected_factor < n else 'endpoint', len(digits), len(nonzero)


def main():
    results = {}
    cases = [(2**p-1, 20260922, 100) for p in (127, 347, 991, 1279, 4253, 8191)]
    cases.extend((1009*(2**127-1), sigma, 100) for sigma in (17, 18, 19, 20))
    cases.append((37, 4, 20))  # A dictionary denominator vanishes on the whole target.
    for n, sigma, b1 in cases:
        for window in (3, 6, 8, 10, 12):
            result, _, _ = check_case(n, sigma, b1, window)
            results[result] = results.get(result, 0) + 1

    assert results == {'endpoint': 30, 'factor': 17, 'dictionary-factor': 3, 'dictionary-degenerate': 5}
    print('Edwards NTT NAF reference PASS', results)
    print('w  dict  cols  nonzero   S/L   M/L    F/L    I/L')
    s = 48*mont.lcm_1_to(12000)
    for window in (4, 6, 8, 10, 12):
        digits = naf(s, window)
        columns, additions = len(digits), sum(bool(digit) for digit in digits)
        rho = additions/columns
        print(f'{window:2} {1 << (window-2):5} {columns:5} {additions:8}  4.00'
              f'  {3+8*rho:4.2f}  {8+8*rho:5.2f}  {7+8*rho:5.2f}')


if __name__ == '__main__':
    main()
