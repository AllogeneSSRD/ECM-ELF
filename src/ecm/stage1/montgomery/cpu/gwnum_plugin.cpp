#include "ecm/stage1/gwnum_abi.h"

extern "C" {
#include <gwnum.h>
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

static_assert(std::string_view(GWNUM_VERSION) == "31.6", "The DLL requires GWNUM 31.6 headers");

namespace {

struct ErrorBuffer {
    char* data;
    uint32_t size;

    template<typename... Args>
    bool report(const char* format, Args... args) const noexcept {
        if (data && size)
            std::snprintf(data, size, format, args...);
        return false;
    }
};

// GWNUM 31.6 lazily initializes the process-wide table mutex in gwsetup.
std::mutex setup_mutex;

struct Point {
    gwnum x{};
    gwnum z{};
};

class Ladder {
    gwhandle gw{};
    std::vector<uint32_t> binary;
    const uint32_t words;
    Point points[3];
    Point* p0 = &points[0];
    Point* p1 = &points[1];
    Point* sum = &points[2];
    gwnum a24{};
    gwnum xdiff{};
    gwnum plus{};
    gwnum minus{};
    gwnum other_plus{};
    gwnum other_minus{};
    gwnum a{};
    gwnum b{};
    gwnum e{};

    void add_sub(Point p, gwnum plus_dst, gwnum minus_dst) {
        gwaddsub4o(&gw, p.x, p.z, plus_dst, minus_dst, GWADD_FORCE_NORMALIZE);
    }

    void finish_double(Point out) {
        // a = (X+Z)^2, b = (X-Z)^2, e = a-b.
        gwsub3o(&gw, a, b, e, GWADD_FORCE_NORMALIZE);
        gwmul3(&gw, a, b, out.x, GWMUL_PRESERVE_S2);
        gwmuladd4(&gw, a24, e, b, a, GWMUL_PRESERVE_S1 | GWMUL_PRESERVE_S2);
        gwmul3(&gw, e, a, out.z, 0);
    }

    void double_point(Point out, Point p) {
        add_sub(p, plus, minus);
        gwsquare2(&gw, plus, a, 0);
        gwsquare2(&gw, minus, b, 0);
        finish_double(out);
    }

    void advance_bit(Point next, Point doubled, Point other) {
        add_sub(doubled, plus, minus);
        add_sub(other, other_plus, other_minus);

        // xADD and xDBL share the transforms of the doubled point's X±Z.
        gwmul3(&gw, plus, other_minus, a, GWMUL_FFT_S1);
        gwmul3(&gw, minus, other_plus, b, GWMUL_FFT_S1);
        gwaddsub4o(&gw, a, b, other_plus, other_minus, GWADD_FORCE_NORMALIZE);
        gwsquare2(&gw, other_plus, next.x, 0);
        gwsquare2(&gw, other_minus, e, 0);
        gwmul3(&gw, e, xdiff, next.z, GWMUL_PRESERVE_S2);

        gwsquare2(&gw, plus, a, 0);
        gwsquare2(&gw, minus, b, 0);
        finish_double(doubled);
    }

    void put(gwnum dst, const uint32_t* value) {
        binarytogw(&gw, value, words, dst);
    }

    bool get(uint32_t* out, gwnum src, ErrorBuffer error) {
        const long count = gwtobinary(&gw, src, binary.data(), static_cast<uint32_t>(binary.size()));
        if (count < 0)
            return error.report("GWNUM residue conversion failed (%ld)", count);
        if (static_cast<unsigned long>(count) > words)
            return error.report("GWNUM residue exceeds the caller's modulus width");
        std::copy_n(binary.data(), count, out);
        std::fill(out + count, out + words, 0u);
        return true;
    }

    bool ok(ErrorBuffer error) {
        const double roundoff = gw_get_maxerr(&gw);
        const int flags = gw_test_for_error(&gw);
        if (!flags && std::isfinite(roundoff) && roundoff <= 0.40)
            return true;
        return error.report("GWNUM arithmetic at FFT length %lu: flags=%d, max roundoff=%.6f",
                            gwfftlen(&gw), flags, roundoff);
    }

public:
    explicit Ladder(uint32_t width) : words(width) {
        // Upstream lazily detects the CPU in the first call to gwinit.
        static const bool cpu_ready = [] {
            gwhandle first{};
            gwinit(&first);
            gwdone(&first);
            return true;
        }();
        (void)cpu_ready;
        gwinit(&gw);
    }

    ~Ladder() {
        const std::lock_guard<std::mutex> lock(setup_mutex);
        gwdone(&gw);
    }
    Ladder(const Ladder&) = delete;
    Ladder& operator=(const Ladder&) = delete;

    bool setup(const uint32_t* modulus, uint32_t exponent, ErrorBuffer error) {
        gwset_num_threads(&gw, 1);
        gwclear_use_benchmarks(&gw);
        gwset_will_error_check(&gw);

        int result;
        {
            const std::lock_guard<std::mutex> lock(setup_mutex);
            result = exponent ? gwsetup(&gw, 1.0, 2, exponent, -1)
                              : gwsetup_general_mod(&gw, modulus, words);
        }
        if (result) {
            char reason[192]{};
            gwerror_text(&gw, result, reason, sizeof(reason));
            return error.report("GWNUM setup failed (%d): %s", result, reason);
        }
        gwerror_checking(&gw, 1);
        // General plans can have an internal bit length larger than the modulus.
        binary.resize(static_cast<size_t>(gw.bit_length) / 32 + 8);

        for (gwnum* field : {&a24, &xdiff, &plus, &minus, &other_plus, &other_minus, &a, &b, &e}) {
            *field = gwalloc(&gw);
            if (!*field)
                return error.report("GWNUM scratch allocation failed");
        }
        for (auto& p : points) {
            p = {gwalloc(&gw), gwalloc(&gw)};
            if (!p.x || !p.z)
                return error.report("GWNUM point allocation failed");
        }
        return true;
    }

    bool initialize(const uint32_t* a24_value, const uint32_t* xdiff_value,
                    const uint32_t* x0, const uint32_t* z0,
                    const uint32_t* x1, const uint32_t* z1, ErrorBuffer error) {
        put(a24, a24_value);
        put(xdiff, xdiff_value);
        gwfft(&gw, a24, a24);
        gwfft(&gw, xdiff, xdiff);
        put(p0->x, x0);
        put(p0->z, z0);
        if (x1) {
            put(p1->x, x1);
            put(p1->z, z1);
        } else {
            double_point(*p1, *p0);
        }
        return ok(error);
    }

    bool advance(const uint8_t* bits, size_t count, ErrorBuffer error) {
        for (size_t i = 0; i < count; ++i) {
            Point*& doubled = bits[i] ? p1 : p0;
            Point*& other = bits[i] ? p0 : p1;
            advance_bit(*sum, *doubled, *other);
            std::swap(other, sum);
        }
        return ok(error);
    }

    bool read(uint32_t* x0, uint32_t* z0, uint32_t* x1, uint32_t* z1, ErrorBuffer error) {
        return ok(error) && get(x0, p0->x, error) && get(z0, p0->z, error) &&
               (!x1 || (get(x1, p1->x, error) && get(z1, p1->z, error))) && ok(error);
    }
};

bool is_mersenne(const uint32_t* modulus, uint32_t words, uint32_t exponent) {
    if ((exponent - 1) / 32 + 1 != words)
        return false;
    const uint32_t full_limb = std::numeric_limits<uint32_t>::max();
    const uint32_t high_limb = full_limb >> ((32 - exponent % 32) % 32);
    return modulus[words - 1] == high_limb &&
           std::all_of(modulus, modulus + words - 1, [=](uint32_t limb) { return limb == full_limb; });
}

bool is_residue(const uint32_t* value, const uint32_t* modulus, uint32_t words) {
    return std::lexicographical_compare(std::make_reverse_iterator(value + words),
                                        std::make_reverse_iterator(value),
                                        std::make_reverse_iterator(modulus + words),
                                        std::make_reverse_iterator(modulus));
}

void* create(const uint32_t* modulus, uint32_t words, uint32_t mersenne_exponent,
             const uint32_t* a24, const uint32_t* xdiff, const uint32_t* x0,
             const uint32_t* z0, const uint32_t* x1, const uint32_t* z1,
             char* error_data, uint32_t error_size) noexcept {
    const ErrorBuffer error{error_data, error_size};
    if (!modulus || !words || words > static_cast<uint32_t>(std::numeric_limits<int>::max()) / 32 ||
        modulus[words - 1] == 0 || !(modulus[0] & 1) || (words == 1 && modulus[0] < 3) ||
        !a24 || !xdiff || !x0 || !z0 || bool(x1) != bool(z1)) {
        error.report("Invalid GWNUM ladder modulus or coordinates");
        return nullptr;
    }
    if (mersenne_exponent && !is_mersenne(modulus, words, mersenne_exponent)) {
        error.report("GWNUM Mersenne exponent does not match the modulus");
        return nullptr;
    }
    for (const auto* coordinate : {a24, xdiff, x0, z0, x1, z1}) {
        if (coordinate && !is_residue(coordinate, modulus, words)) {
            error.report("GWNUM ladder residue is outside the modulus");
            return nullptr;
        }
    }
    try {
        auto ladder = std::make_unique<Ladder>(words);
        if (ladder->setup(modulus, mersenne_exponent, error) &&
            ladder->initialize(a24, xdiff, x0, z0, x1, z1, error))
            return ladder.release();
    } catch (const std::exception& exception) {
        error.report("GWNUM setup failed: %s", exception.what());
    } catch (...) {
        error.report("GWNUM setup failed");
    }
    return nullptr;
}

int advance(void* context, const uint8_t* bits, size_t count,
            char* error_data, uint32_t error_size) noexcept {
    const ErrorBuffer error{error_data, error_size};
    if (!context || (!bits && count)) {
        error.report("Invalid GWNUM ladder input");
        return 1;
    }
    return static_cast<Ladder*>(context)->advance(bits, count, error) ? 0 : 1;
}

int read(void* context, uint32_t* x0, uint32_t* z0, uint32_t* x1, uint32_t* z1,
         char* error_data, uint32_t error_size) noexcept {
    const ErrorBuffer error{error_data, error_size};
    if (!context || !x0 || !z0 || bool(x1) != bool(z1)) {
        error.report("Invalid GWNUM ladder output");
        return 1;
    }
    return static_cast<Ladder*>(context)->read(x0, z0, x1, z1, error) ? 0 : 1;
}

void destroy(void* context) noexcept {
    delete static_cast<Ladder*>(context);
}

const ecm_gwnum_api api{
    ECM_GWNUM_ABI_VERSION,
    sizeof(ecm_gwnum_api),
    GWNUM_VERSION,
    create,
    advance,
    read,
    destroy
};

}  // namespace

extern "C" __declspec(dllexport) const ecm_gwnum_api* ecm_gwnum_get_api(uint32_t abi_version) noexcept {
    return abi_version == ECM_GWNUM_ABI_VERSION ? &api : nullptr;
}
