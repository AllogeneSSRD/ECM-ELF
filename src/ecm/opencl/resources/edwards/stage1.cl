#ifndef EDWARDS_KERNEL
#define EDWARDS_KERNEL __kernel
#define EDWARDS_LANE get_global_id(0)
#define EDWARDS_TWO_XY TWO_XY
#endif

// SoA: adjacent work-items read the same limb of adjacent curves.
// Dictionary ranges are bounded independently of the chosen NAF window.
EDWARDS_KERNEL void edwards_dictionary(__global const uint *constants,
        __global const uint *base, __global uint *dictionary, __global uint *next,
        uint count, uint np0, uint start, uint stop, uint entries) {
    const size_t lane = EDWARDS_LANE;
    if (lane >= count)
        return;
    const fe n = fe_load(constants, 1);
    const fe d = fe_load(base + lane, count);
    const fe x = fe_load(base + (size_t)LIMBS * count + lane, count);
    const fe y = fe_load(base + (size_t)2 * LIMBS * count + lane, count);
    const fe one = fe_load(constants + LIMBS, 1);
    const ed_point p = {x, y, one, fe_mul(x, y, n, np0)};
    ed_point odd = start ? point_load(next + lane, count) : p;
    ed_point two;
    if (entries > 1) {
        two = ed_double(p, n, np0, 1, EDWARDS_TWO_XY);
        two.t = fe_mul(two.t, d, n, np0);
    }
    for (uint j = start; j < stop; ++j) {
        ed_point cached = odd;
        cached.t = fe_mul(odd.t, d, n, np0);
        point_store(dictionary + (size_t)j * 4 * LIMBS * count + lane, count, cached);
        if (j + 1 < entries)
            odd = ed_add(odd, two, n, np0, 1);
    }
    if (stop < entries)
        point_store(next + lane, count, odd);
}

EDWARDS_KERNEL void edwards_slice(__global const uint *constants,
        __global uint *state, __global const uint *dictionary,
        __global const uint *nonzero, __global const uint *codes,
        uint count, uint np0, uint bits, uint start, uint stop, uint entries) {
    const size_t lane = EDWARDS_LANE;
    if (lane >= count)
        return;
    const fe n = fe_load(constants, 1);
    const fe zero = {0};
    ed_point p = point_load(state + lane, count);
    for (uint pos = start; pos < stop; ++pos) {
        const uint bit = bits - 1 - pos, word = bit >> 5, shift = bit & 31;
        const int add = (nonzero[word] >> shift) & 1;
        const int last = pos + 1 == stop;
        p = ed_double(p, n, np0, add || last, EDWARDS_TWO_XY);
        if (add) {
            const ulong packed = (ulong)codes[word] | ((ulong)codes[word + 1] << 32);
            const uint code = (uint)(packed >> shift) & (2 * entries - 1);
            const size_t offset = (size_t)(code & (entries - 1)) * 4 * LIMBS * count + lane;
            ed_point q = point_load(dictionary + offset, count);
            if (code & entries) {
                q.x = fe_sub(zero, q.x, n);
                q.t = fe_sub(zero, q.t, n);
            }
            p = ed_add(p, q, n, np0, last);
        }
    }
    point_store(state + lane, count, p);
}
