#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Lucas decoding adapted from ECM Library ecm.c (2001-2024), Paul Zimmermann,
# Alexander Kruppa, Cyril Bouvier, David Cleaver, Philip McLaughlin.
"""Source-based Stage1 differential-chain planning, independent of GPU execution.

PRAC control flow: .refactor/p95v3106b01.source/ecm.cpp:2645,2711,2857.
Lucas decoder: .refactor/ecm/ecm.c:549,1027 (ECM Library, LGPL-3.0-or-later).
Scalar relations and point lifetimes are checked while constructing a plan;
this does not prove behavior on singular curve states or GPU performance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import struct
from typing import Iterator


P95_RATIOS = (
    0.6180339887498948, 0.7236067977499790, 0.5801787282954641,
    0.6328398060887063, 0.6124299495094950, 0.6201819808074158,
    0.6172146165344039, 0.6183471196562281, 0.6179144065288179,
    0.6180796684698958,
)


@dataclass(frozen=True)
class Operation:
    kind: str
    inputs: tuple[int, ...]
    scalar: int


@dataclass(frozen=True)
class Plan:
    multiplier: int
    operations: tuple[Operation, ...]
    final: int
    source: str
    initial_d: int | None = None

    @property
    def counts(self) -> tuple[int, int]:
        counts = Counter(op.kind for op in self.operations)
        return counts['dbl'], counts['dadd']

    def lower(self) -> dict:
        """Linear scan, permitting an alias-safe output to reuse a dead input.

        Slots count persistent XZ points, excluding field-operation temporaries.
        All source coordinates must be consumed before an output overwrites them.
        """
        last = {0: 0, self.final: len(self.operations) + 1}
        for index, op in enumerate(self.operations, 1):
            for node in op.inputs:
                last[node] = max(index, last.get(node, 0))
        slots = {0: 0}
        free: list[int] = []
        next_slot = 1
        peak = 1
        program = []
        for index, op in enumerate(self.operations, 1):
            inputs = [slots[node] for node in op.inputs]
            for node in set(op.inputs):
                if last[node] == index:
                    free.append(slots.pop(node))
            if free:
                slot = min(free)
                free.remove(slot)
            else:
                slot = next_slot
                next_slot += 1
            slots[index] = slot
            peak = max(peak, len(slots))
            program.append(dict(op=op.kind, dst=slot, src=inputs))
            if index not in last:
                free.append(slots.pop(index))
        return dict(point_slots=next_slot, peak_live_points=peak,
                    input_slot=0, final_slot=slots[self.final], operations=program)


class Builder:
    def __init__(self) -> None:
        self.values = [1]
        self.operations: list[Operation] = []

    def dbl(self, a: int) -> int:
        return self._append('dbl', (a,), 2 * self.values[a])

    def dadd(self, a: int, b: int, difference: int) -> int:
        x, y, z = (self.values[i] for i in (a, b, difference))
        if z == abs(x - y):
            result = x + y
        elif z == x + y:
            result = abs(x - y)
        else:
            raise ValueError(f'Illegal differential addition: {x}, {y}, diff={z}')
        if result == 0:
            raise ValueError('Zero scalar needs an explicit infinity operation')
        return self._append('dadd', (a, b, difference), result)

    def _append(self, kind: str, inputs: tuple[int, ...], value: int) -> int:
        self.operations.append(Operation(kind, inputs, value))
        self.values.append(value)
        return len(self.operations)

    def finish(self, multiplier: int, final: int, source: str,
               initial_d: int | None = None) -> Plan:
        if self.values[final] != multiplier:
            raise ValueError(f'Chain ended at {self.values[final]}, expected {multiplier}')
        return Plan(multiplier, tuple(self.operations), final, source, initial_d)


def primes(limit: int) -> list[int]:
    """Odd-only sieve for offline studies, O(limit/2) byte storage."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * ((limit + 1) // 2)
    sieve[0] = 0
    for p in range(3, math.isqrt(limit) + 1, 2):
        if sieve[p // 2]:
            start = p * p // 2
            count = (len(sieve) - 1 - start) // p + 1
            sieve[start::p] = b'\x00' * count
    return [2] + [2 * i + 1 for i in range(1, len(sieve)) if sieve[i]]


def prime_multiplicity(p: int, limit: int) -> int:
    """Use floor(log_p(limit)) once, not the sum of 1..exponent."""
    power, exponent = p, 0
    while power <= limit:
        exponent += 1
        if power > limit // p:
            break
        power *= p
    return exponent


def prime_window(low: int, high: int) -> list[int]:
    """Sieve a bounded interval for high-B1 work-density studies."""
    if high < low or low < 2:
        raise ValueError('Expected 2 <= low <= high')
    start = max(3, low | 1)
    length = max(0, (high - start) // 2 + 1)
    sieve = bytearray(b'\x01') * length
    for p in primes(math.isqrt(high)):
        if p == 2:
            continue
        first = max(p * p, ((start + p - 1) // p) * p)
        if first % 2 == 0:
            first += p
        index = (first - start) // 2
        if index < length:
            count = (length - 1 - index) // p + 1
            sieve[index::p] = b'\x00' * count
    return ([2] if low <= 2 <= high else []) + [start + 2 * i for i, v in enumerate(sieve) if v]


def prac_counts(p: int, initial_d: int) -> tuple[int, int]:
    """Prime95's simplified rules for an odd prime, including initial/final ops.

    Composite multipliers needing recursive outer passes are rejected. Stage1
    uses each prime repeatedly, so those passes are not required here.
    """
    if p < 3 or p % 2 == 0 or not p // 2 < initial_d < p:
        raise ValueError('Expected an odd prime and p/2 < d < p')
    e = p - initial_d
    d = initial_d - e
    dbl, add = 1, 1
    while d != e:
        if d < e:
            d, e = e, d
        add += 1
        if 100 * d <= 296 * e:
            d -= e
        elif d % 2 == e % 2:
            d = (d - e) // 2
            dbl += 1
        elif d % 2 == 0:
            d //= 2
            dbl += 1
        else:
            e //= 2
            dbl += 1
    if d != 1:
        raise ValueError('Composite/non-coprime multiplier needs recursive PRAC')
    return dbl, add


@lru_cache(maxsize=8192)
def select_d(p: int, search: int = 7, sqr: float = 1.0) -> tuple[int, int, int]:
    if not 1 <= search <= 50:
        raise ValueError('PracSearch must be in 1..50')
    if not 0 < sqr < math.inf:
        raise ValueError('S/M must be finite and positive')
    best = None
    seen = set()
    for ratio in P95_RATIOS:
        center = math.ceil(float(p) * ratio)
        for d in range(center - search // 2, center - search // 2 + search):
            if d in seen or not p // 2 < d < p:
                continue
            seen.add(d)
            dbl, add = prac_counts(p, d)
            cost = dbl * (3 + 2 * sqr) + add * (4 + 2 * sqr)
            if best is None or cost < best[0]:
                best = cost, d, dbl, add
    if best is None:
        raise ValueError(f'No valid PRAC seed for {p}')
    return best[1], best[2], best[3]


def prac_plan(p: int, initial_d: int | None = None,
              search: int = 7, sqr: float = 1.0) -> Plan:
    """Build a prime multiplier with three logical A/B/C roles.

    Prime95's first-step copy elision has the same field arithmetic as C=A
    followed by ordinary rules. SSA represents this alias without a COPY;
    physical copies and scratch costs depend on the backend.
    """
    builder = Builder()
    if p == 2:
        return builder.finish(p, builder.dbl(0), 'prime95-prac')
    if initial_d is None:
        initial_d, expected_dbl, expected_add = select_d(p, search, sqr)
    else:
        expected_dbl, expected_add = prac_counts(p, initial_d)
    a, b, c = 0, builder.dbl(0), 0
    e = p - initial_d
    d = initial_d - e
    while d != e:
        if d < e:
            d, e = e, d
            a, b = b, a
        if 100 * d <= 296 * e:
            c_new = builder.dadd(a, b, c)
            b, c = c_new, b
            d -= e
        elif d % 2 == e % 2:
            b = builder.dadd(a, b, c)
            a = builder.dbl(a)
            d = (d - e) // 2
        elif d % 2 == 0:
            c = builder.dadd(a, c, b)
            a = builder.dbl(a)
            d //= 2
        else:
            c = builder.dadd(b, c, a)
            b = builder.dbl(b)
            e //= 2
    result = builder.dadd(b, a, c)
    plan = builder.finish(p, result, 'prime95-prac', initial_d)
    if plan.counts != (expected_dbl, expected_add):
        raise ValueError('PRAC counts and emitted plan disagree')
    return plan


def lucas_plan(p: int, code: int = 0) -> Plan:
    """Decode the local ECM fork's uint64 code, checking every relation."""
    if not 0 <= code < 1 << 64:
        raise ValueError('Lucas code must fit uint64')
    builder = Builder()
    nodes = [0, builder.dbl(0)]
    if p == 2:
        return builder.finish(p, nodes[-1], 'ecm-lucas')
    nodes.append(builder.dadd(nodes[1], nodes[0], nodes[0]))

    def append_offsets(first: int, second: int | None = None,
                       difference: int | None = None) -> None:
        parent = len(nodes) - 1
        if not 0 <= first <= parent:
            raise ValueError('Lucas source offset is out of range')
        if second is None:
            node = builder.dbl(nodes[parent - first])
        else:
            if not 0 <= second <= parent:
                raise ValueError('Lucas second offset is out of range')
            x = builder.values[nodes[parent - first]]
            y = builder.values[nodes[parent - second]]
            if difference is None:
                delta = abs(x - y)
                candidates = [i for i, n in enumerate(nodes) if builder.values[n] == delta]
                if not candidates:
                    raise ValueError(f'Lucas code has no difference point {delta}')
                difference = parent - candidates[-1]
            if not 0 <= difference < 15 or difference > parent:
                raise ValueError('Lucas difference offset exceeds decoder contract')
            node = builder.dadd(nodes[parent - first], nodes[parent - second],
                                nodes[parent - difference])
        if builder.values[node] <= builder.values[nodes[-1]]:
            raise ValueError('Lucas code is not a strictly increasing chain')
        nodes.append(node)

    def maximal(count: int) -> None:
        for _ in range(count):
            append_offsets(0, 1)

    if p in (3, 5, 7):
        if p == 5:
            append_offsets(0, 1, 2)
        elif p == 7:
            append_offsets(1)
            append_offsets(0, 1, 3)
        return builder.finish(p, nodes[-1], 'ecm-lucas')
    if p < 11:
        raise ValueError('Lucas decoder expects a prime multiplier')
    start = code & 7
    code >>= 3
    starts = {
        7: ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
        6: ((0, 1, 2), (0, 1, 2), (0, 2, 1)),
        5: ((0, 1, 2), (0, 1, 2), (1, None, None)),
        4: ((0, 1, 2), (0, 2, 1)),
        3: ((0, 1, 2), (1, None, None)),
        2: ((1, None, None), (0, 1, 3)),
        1: ((1, None, None), (0, 3, 1)),
        0: ((1, None, None), (1, None, None)),
    }
    for offsets in starts[start]:
        append_offsets(*offsets)
    while code:
        fragment = code & 15
        code >>= 4
        if fragment == 0:
            extension = code & 3
            code >>= 2
            if extension == 3:
                append_offsets(2)
            else:
                maximal(12 * (extension + 1))
        elif fragment == 1:
            append_offsets(0, 2)
        elif fragment == 2:
            append_offsets(1)
        elif 3 <= fragment <= 8:
            maximal((fragment - 1) // 2)
            if fragment % 2:
                append_offsets(0, 2, 1)
            else:
                append_offsets(1)
        elif 9 <= fragment <= 12:
            extension = code & 3
            code >>= 2
            maximal(extension + (4 if fragment <= 10 else 8))
            if fragment % 2:
                append_offsets(0, 2, 1)
            else:
                append_offsets(1)
        elif fragment == 13:
            extension = code & 3
            code >>= 2
            first, second = ((0, 7), (0, 8), (2, 3), (2, 4))[extension]
            append_offsets(first, second)
        elif fragment == 14:
            extension = code & 3
            code >>= 2
            append_offsets(0, 3 + extension)
        else:
            extension = code & 3
            code >>= 2
            append_offsets(1, 2 + extension)
        if len(nodes) > 64:
            raise ValueError('Lucas code exceeds the local working-chain bound')
    while builder.values[nodes[-1]] < p:
        maximal(1)
    return builder.finish(p, nodes[-1], 'ecm-lucas')


def load_lucas_codes(path: Path) -> tuple[dict[int, int], dict]:
    """Read headerless little-endian uint64 records starting at prime 11.

    Completeness and optimality are not inferred. Each used record's scalar
    relations are validated by lucas_plan().
    """
    import hashlib
    data = path.read_bytes()
    if not data or len(data) % 8:
        raise ValueError('Lucas file must contain complete uint64 records')
    count = len(data) // 8 + 4
    bound = max(32, int(count * (math.log(count) + math.log(math.log(count)))) + 32)
    sequence = primes(bound)
    while len(sequence) < count:
        bound *= 2
        sequence = primes(bound)
    covered = sequence[4:count]
    codes = {p: record[0] for p, record in zip(covered, struct.iter_unpack('<Q', data))}
    return codes, dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                       records=len(codes), first_prime=11, last_prime=covered[-1],
                       endian='little', header=False)


def stage1_terms(limit: int) -> Iterator[tuple[int, int]]:
    for p in primes(limit):
        yield p, prime_multiplicity(p, limit)


def scalar_bits(terms: list[tuple[int, int]], torsion: int) -> int:
    """Exact bit length of t*lcm via a balanced integer product."""
    stack: list[int | None] = []
    for p, exponent in terms:
        value = p ** exponent
        level = 0
        while level < len(stack) and stack[level] is not None:
            value *= stack[level]
            stack[level] = None
            level += 1
        if level == len(stack):
            stack.append(value)
        else:
            stack[level] = value
    product = torsion
    for value in stack:
        if value is not None:
            product *= value
    return product.bit_length()
