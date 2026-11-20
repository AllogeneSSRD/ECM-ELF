#include "ecm/io/results.h"
#include "ecm/integer.h"
#include "io/files.h"
#include "io/generated/p95_fft_reference.h"
#include "io/json.h"

#include "ecm/platform/program_version.h"
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ecm_results {
void append_text_record(const std::filesystem::path& path, const std::string& line) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    ecm_io::FileLock lock(path.u8string());
    auto bytes = std::filesystem::exists(path) ? ecm_io::read_file(path) : std::string{};
    std::istringstream input(bytes);
    std::string old;
    while (std::getline(input, old)) {
        if (!old.empty() && old.back() == '\r')
            old.pop_back();
        if (old == line)
            return;
    }
    if (!bytes.empty() && bytes.back() != '\n')
        bytes += '\n';
    ecm_io::write_atomic(path, bytes + line + '\n');
}

std::string quoted(const std::string& s) { return json::quote(s); }

std::string decimal(const mpz_t n) {
    std::vector<char> b(mpz_sizeinbase(n, 10) + 3);
    mpz_get_str(b.data(), 10, n);
    return b.data();
}

std::string normalized_integer(const std::string& text) {
    ecm_edwards::Integer v;
    if (mpz_set_str(v.v, text.c_str(), 10))
        throw std::runtime_error("invalid decimal metadata");
    return decimal(v.v);
}

std::string utc_timestamp() {
    const auto now = std::time(nullptr);
    std::tm t{};
#ifdef _WIN32
    gmtime_s(&t, &now);
#else
    gmtime_r(&now, &t);
#endif
    std::ostringstream out;
    out << std::put_time(&t, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

uint32_t reference_fft(uint64_t exponent) {
    for (const auto& entry : p95_fft_capacities)
        if (exponent <= entry.max_exponent)
            return entry.length;
    throw std::runtime_error("target exceeds Prime95 FFT reference capacity");
}

// Verify original M metadata against the actual modulus. Only exact Mersenne
// inputs are inferred; a cofactor needs its original exponent and known factors.
uint64_t target_exponent(const Submission& r, const mpz_t n) {
    ecm_edwards::Integer m, f;
    if (r.kbnc && (r.k != "1" || r.b != 2 || r.c != -1)) {
        if (!r.exponent || r.exponent > 1000000000 || r.b < 2 ||
            mpz_set_str(f.v, r.k.c_str(), 10) || mpz_sgn(f.v) <= 0)
            throw std::runtime_error("invalid k*b^n+c submission target");
        mpz_ui_pow_ui(m.v, r.b, static_cast<unsigned long>(r.exponent));
        mpz_mul(m.v, m.v, f.v);
        if (r.c < 0)
            mpz_sub_ui(m.v, m.v, static_cast<unsigned long>(-static_cast<int64_t>(r.c)));
        else
            mpz_add_ui(m.v, m.v, r.c);
        for (const auto& text : r.known) {
            if (text.empty() || text.find_first_not_of("0123456789") != text.npos ||
                mpz_set_str(f.v, text.c_str(), 10) || mpz_cmp_ui(f.v, 1) <= 0 ||
                !mpz_divisible_p(m.v, f.v))
                throw std::runtime_error("invalid submission known factor");
            mpz_divexact(m.v, m.v, f.v);
        }
        if (mpz_cmp(m.v, n))
            throw std::runtime_error("submission target differs from saved modulus");
        return 0;
    }
    if (!r.exponent) {
        if (!r.known.empty()) {
            // A save-only task has the exact cofactor N, but no original expression.
            // Keep known factors as metadata and report the explicit N.
            for (const auto& text : r.known)
                if (text.empty() || text.find_first_not_of("0123456789") != text.npos ||
                    mpz_set_str(f.v, text.c_str(), 10) || mpz_cmp_ui(f.v, 1) <= 0)
                    throw std::runtime_error("invalid submission known factor");
            return 0;
        }
        mpz_add_ui(m.v, n, 1);
        return mpz_popcount(m.v) == 1 ? mpz_scan1(m.v, 0) : 0;
    }
    if (r.exponent > 1000000000)
        throw std::runtime_error("submission exponent exceeds supported range");
    mpz_set_ui(m.v, 1);
    mpz_mul_2exp(m.v, m.v, static_cast<mp_bitcnt_t>(r.exponent));
    mpz_sub_ui(m.v, m.v, 1);
    for (const auto& text : r.known) {
        if (text.empty() || text.find_first_not_of("0123456789") != text.npos ||
            mpz_set_str(f.v, text.c_str(), 10) || mpz_cmp_ui(f.v, 1) <= 0 ||
            !mpz_divisible_p(m.v, f.v))
            throw std::runtime_error("invalid submission known factor");
        mpz_divexact(m.v, m.v, f.v);
    }
    if (mpz_cmp(m.v, n))
        throw std::runtime_error("submission target differs from saved modulus");
    return r.exponent;
}

uint32_t crc32(const std::string& s) {
    uint32_t c = 0xffffffffu;
    for (unsigned char b : s) {
        c ^= b;
        for (int i = 0; i < 8; ++i)
            c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0);
    }
    return c ^ 0xffffffffu;
}

std::string crc_hex(const std::string& s) {
    std::ostringstream out;
    out << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << crc32(s);
    return out.str();
}

// Compute from the final record, including its source version, model marker,
// actual/reference FFT field and final timestamp. Sealing must call this again.
static json::Value result_with_checksum(json::Value row) {
    if (row.type != 'o')
        throw std::runtime_error("result is not a JSON object");
    if (!row.has("errors"))
        row.set("errors", json::parse("{\"invariant\":0}"));
    if (!row.has("os"))
        row.set("os", json::parse("{\"os\":" + quoted(ecm_program::host_os) +
                                  ",\"architecture\":" + quoted(ecm_program::host_arch) + "}"));
    std::string target = row.get("exponent").text();
    if (target.empty()) {
        target = row.get("N").text();
        if (target.empty() && row.has("k")) {
            ecm_edwards::Integer n, k, f;
            mpz_ui_pow_ui(n.v, std::stoul(row.get("b").text()), std::stoul(row.get("n").text()));
            if (mpz_set_str(k.v, row.get("k").text().c_str(), 10))
                throw std::runtime_error("invalid checksum target");
            mpz_mul(n.v, n.v, k.v);
            const auto c = std::stoll(row.get("c").text());
            if (c < 0)
                mpz_sub_ui(n.v, n.v, static_cast<unsigned long>(-c));
            else
                mpz_add_ui(n.v, n.v, static_cast<unsigned long>(c));
            for (const auto& v : row.get("known-factors").items) {
                if (mpz_set_str(f.v, v.text().c_str(), 10) || mpz_cmp_ui(f.v, 1) <= 0 ||
                    !mpz_divisible_p(n.v, f.v))
                    throw std::runtime_error("invalid checksum known factor");
                mpz_divexact(n.v, n.v, f.v);
            }
            target = decimal(n.v);
        }
        if (target.empty())
            throw std::runtime_error("missing checksum target");
        target = "N=" + target;
    }
    const auto& model = row.has("Edwards") ? row.get("Edwards") : row.get("GMP-ECM-param3");
    const auto sigma =
        (row.has("Edwards") ? "E" : "") + model.get("sigma").text(row.get("sigma").text());
    const auto b1 = row.get("b1").text(), b2 = row.get("b2").text();
    const auto bound2 = !b2.empty() && std::stoull(b2) > std::stoull(b1) ? b2 : std::string{};
    const auto &program = row.get("program"), &os = row.get("os");
    const std::vector<std::string> slots = {target,
                                            "ECM",
                                            json::joined(row.get("factors")),
                                            json::joined(row.get("known-factors")),
                                            b1,
                                            bound2,
                                            sigma,
                                            row.get("fft-length").text(),
                                            "invariant:" +
                                                row.get("errors").get("invariant").text("0"),
                                            program.get("name").text(),
                                            program.get("version").text(),
                                            "",
                                            "",
                                            os.get("os").text(),
                                            os.get("architecture").text(),
                                            row.get("timestamp").text()};
    std::string canonical;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (i)
            canonical += ';';
        canonical += slots[i];
    }
    row.set("checksum",
            json::parse("{\"version\":1,\"checksum\":" + quoted(crc_hex(canonical)) + "}"));
    return row;
}

std::string with_result_checksum(const std::string& line) {
    return result_with_checksum(json::parse(line)).dump();
}

// Internal Stage1/Stage2 streams may share a target. The engine marker identifies
// the producer; Stage2 can legitimately report a factor found at saved Stage1 X.
static bool stage2_internal_row(const json::Value& row) {
    if (row.type != 'o' || row.get("program").get("name").text() != ecm_program::name ||
        row.get("worktype").text() != "ECM" ||
        row.has("stage1-engine") == row.has("stage2-engine") ||
        row.get(row.has("stage2-engine") ? "stage2-engine" : "stage1-engine").type != 'o')
        throw std::runtime_error("internal output is not an owned ECM result");
    if (row.get("checksum").dump() != result_with_checksum(row).get("checksum").dump())
        throw std::runtime_error("internal result checksum mismatch");
    return row.has("stage2-engine");
}

std::string submission_json(const Submission& r, const mpz_t n) {
    const bool hit = !r.factors.empty();
    if (!r.curves || r.b1 < 2 ||
        (!hit && (r.stage || (r.stage1_only ? r.b2 != r.b1 : r.b2 <= r.b1))) ||
        (hit && r.stage != 1 && r.stage != 2) ||
        (r.stage1_only && (r.stage == 2 || r.d || r.poly_size || r.stage2_fft_length)) ||
        (!r.edwards && r.param != 0 && r.param != 2 && r.param != 3))
        throw std::runtime_error("inconsistent ECM submission result");
    if ((r.d == 0) != (r.poly_size == 0) || (r.d && (r.stage == 1 || r.d < 6 || r.d % 2)))
        throw std::runtime_error("inconsistent Stage2 submission geometry");
    if (r.stage2_fft_length && (r.stage == 1 || !r.d))
        throw std::runtime_error("inconsistent Stage2 FFT metadata");
    ecm_edwards::Integer f;
    for (const auto& text : r.factors)
        if (mpz_set_str(f.v, text.c_str(), 10) || mpz_cmp_ui(f.v, 1) <= 0 || mpz_cmp(f.v, n) >= 0 ||
            !mpz_divisible_p(n, f.v))
            throw std::runtime_error("invalid submission factor");
    if (!r.aid.empty() &&
        (r.aid.size() != 32 || r.aid.find_first_not_of("0123456789abcdefABCDEF") != r.aid.npos))
        throw std::runtime_error("invalid submission assignment id");
    const auto exponent = target_exponent(r, n);
    // Generic N follows a conservative zero-padded capacity reference. This is
    // protocol reporting metadata, independent of the engine's actual transform.
    const auto fft = reference_fft(exponent ? exponent : 2 * mpz_sizeinbase(n, 2) + 128);
    std::ostringstream out;
    out << "{\"status\":" << quoted(hit ? "F" : "NF");
    if (r.kbnc && !exponent)
        out << ",\"k\":" << normalized_integer(r.k) << ",\"b\":" << r.b << ",\"n\":" << r.exponent
            << ",\"c\":" << r.c;
    else if (exponent)
        out << ",\"exponent\":" << exponent;
    else {
        const auto value = decimal(n);
        out << ",\"N\":" << (value.size() > 2000 ? quoted(value) : value);
    }
    out << ",\"worktype\":\"ECM\"";
    if (hit) {
        out << ",\"factors\":[";
        for (size_t i = 0; i < r.factors.size(); ++i) {
            if (i)
                out << ',';
            out << quoted(r.factors[i]);
        }
        out << ']';
    }
    out << ",\"b1\":" << r.b1;
    if (r.stage != 1)
        out << ",\"b2\":" << r.b2;
    if (r.d)
        out << ",\"d\":" << r.d << ",\"poly-size\":" << r.poly_size;
    if (r.stage2_fft_length)
        out << ",\"stage2-fft-length\":" << r.stage2_fft_length;
    if (r.edwards) {
        out << ",\"Edwards\":{";
        if (hit)
            out << "\"sigma\":" << r.sigma;
        out << '}';
    } else if (r.param == 3) {
        out << ",\"GMP-ECM-param3\":{";
        if (hit)
            out << "\"sigma\":" << r.sigma;
        out << '}';
    } else if (hit) {
        out << ",\"sigma\":" << r.sigma;
        if (r.param)
            out << ",\"param\":" << r.param;
    }
    if (hit)
        out << ",\"stage\":" << r.stage;
    out << ",\"curves\":" << r.curves << ",\"fft-length\":" << fft;
    out << ",\"program\":{\"name\":" << quoted(ecm_program::name)
        << ",\"version\":" << quoted(ecm_program::version);
#ifdef _WIN64
    out << ",\"port\":4";
#endif
    out << "},\"timestamp\":" << quoted(r.when.empty() ? utc_timestamp() : r.when);
    if (!r.known.empty()) {
        out << ",\"known-factors\":[";
        for (size_t i = 0; i < r.known.size(); ++i) {
            if (i)
                out << ',';
            out << quoted(r.known[i]);
        }
        out << ']';
    }
    if (!r.user.empty())
        out << ",\"user\":" << quoted(r.user);
    if (!r.computer.empty())
        out << ",\"computer\":" << quoted(r.computer);
    if (!r.aid.empty())
        out << ",\"aid\":" << quoted(r.aid);
    out << '}';
    const auto line = with_result_checksum(out.str());
    // An arbitrary long integer has no compact PrimeNet assignment identity.
    // Keep its exact local export rather than preventing Stage1 computation.
    if ((exponent || r.kbnc) && line.size() >= 2000)
        throw std::runtime_error("ECM submission exceeds Prime95 JSON message capacity");
    return line + '\n';
}
} // namespace ecm_results

namespace ecm_results {
// Same canonical slots as the CPU Stage2 result path. Runtime metadata and
// personal identity are not included in the arithmetic-result checksum.
std::string internal_json(Submission r, const mpz_t n, uint64_t actual_fft,
                          const std::string& source_build, const std::string& extra_fields) {
    const auto exponent = target_exponent(r, n);
    const bool hit = !r.factors.empty();
    if (r.when.empty())
        r.when = utc_timestamp();
    std::ostringstream out;
    out << "{\"status\":" << quoted(hit ? "F" : "NF");
    if (r.kbnc && !exponent)
        out << ",\"k\":" << normalized_integer(r.k) << ",\"b\":" << r.b << ",\"n\":" << r.exponent
            << ",\"c\":" << r.c;
    else if (exponent)
        out << ",\"exponent\":" << exponent;
    else {
        const auto value = decimal(n);
        out << ",\"N\":" << (value.size() > 2000 ? quoted(value) : value);
    }
    out << ",\"worktype\":\"ECM\",\"b1\":" << r.b1;
    if (r.stage != 1)
        out << ",\"b2\":" << r.b2;
    if (r.d)
        out << ",\"d\":" << r.d << ",\"poly-size\":" << r.poly_size;
    out << ",\"curves\":1";
    if (r.edwards)
        out << ",\"Edwards\":{\"sigma\":" << r.sigma << '}';
    else
        out << ",\"sigma\":" << r.sigma << ",\"param\":" << r.param;
    out << ",\"fft-length\":" << actual_fft;
    if (hit) {
        out << ",\"stage\":" << r.stage << ",\"factors\":[";
        for (size_t i = 0; i < r.factors.size(); ++i) {
            if (i)
                out << ',';
            out << quoted(r.factors[i]);
        }
        out << ']';
    }
    if (!r.known.empty()) {
        out << ",\"known-factors\":[";
        for (size_t i = 0; i < r.known.size(); ++i) {
            if (i)
                out << ',';
            out << quoted(r.known[i]);
        }
        out << ']';
    }
    out << ",\"errors\":{\"invariant\":0},\"program\":{\"name\":" << quoted(ecm_program::name)
        << ",\"version\":" << quoted(ecm_program::version)
        << ",\"port\":" << quoted(std::string(ecm_program::host_os) + " " + ecm_program::host_arch)
        << '}' << ",\"source-build\":" << quoted(source_build)
        << ",\"os\":{\"os\":" << quoted(ecm_program::host_os)
        << ",\"architecture\":" << quoted(ecm_program::host_arch)
        << "},\"timestamp\":" << quoted(r.when);
    if (!r.user.empty())
        out << ",\"user\":" << quoted(r.user);
    if (!r.computer.empty())
        out << ",\"computer\":" << quoted(r.computer);
    if (!r.aid.empty())
        out << ",\"aid\":" << quoted(r.aid);
    out << extra_fields << '}';
    return with_result_checksum(out.str()) + '\n';
}
} // namespace ecm_results

namespace ecm_results {

namespace fs = std::filesystem;
namespace {
constexpr size_t anchor_bytes = 256;

ecm_io::FileLock lock_result(const fs::path& path) {
    if (path.has_parent_path())
        fs::create_directories(path.parent_path());
    return ecm_io::FileLock(path.u8string());
}

std::string existing_bytes(const fs::path& path) {
    return fs::exists(path) ? ecm_io::read_file(path) : std::string{};
}

bool contains_batch(const std::string& bytes, const std::string& batch, size_t offset) {
    for (auto at = bytes.find(batch, offset); at != bytes.npos; at = bytes.find(batch, at + 1))
        if (at == 0 || bytes[at - 1] == '\n')
            return true;
    return false;
}

bool contains_legacy_report(const std::string& bytes, const std::string& report) {
    const auto line = report.substr(0, report.size() - 1);
    for (auto at = bytes.find(line); at != bytes.npos; at = bytes.find(line, at + 1)) {
        const auto end = at + line.size();
        if ((at == 0 || bytes[at - 1] == '\n') &&
            (end == bytes.size() || bytes[end] == '\n' || bytes.compare(end, 2, "\r\n") == 0))
            return true;
    }
    return false;
}
} // namespace

void append_checkpointed(const fs::path& path, const std::string& batch,
                         AppendCheckpoint& state, const std::function<void()>& save,
                         const std::function<void()>& acknowledge, bool adopt) {
    if (batch.empty() || batch.back() != '\n')
        throw std::runtime_error("invalid checkpoint publication batch");
    auto lock = lock_result(path);
    auto bytes = existing_bytes(path);
    if (!state.pending) {
        if (adopt && contains_legacy_report(bytes, batch)) {
            acknowledge();
            return;
        }
        state.pending = true;
        state.offset = bytes.size();
        state.anchor = bytes.substr(bytes.size() - (std::min)(bytes.size(), anchor_bytes));
        save();
    } else {
        if (state.offset > bytes.size() ||
            state.anchor.size() != (std::min)(state.offset, uint64_t(anchor_bytes)) ||
            bytes.compare(size_t(state.offset) - state.anchor.size(), state.anchor.size(), state.anchor))
            throw std::runtime_error("cannot recover pending append at byte " +
                std::to_string(state.offset) + "; restore the output file: " + path.u8string());
        if (contains_batch(bytes, batch, size_t(state.offset))) {
            state = {};
            acknowledge();
            return;
        }
        const auto start = size_t(state.offset) +
            (state.offset && state.anchor.back() != '\n' && state.offset < bytes.size() &&
             bytes[size_t(state.offset)] == '\n');
        const auto visible = bytes.size() - start;
        if (visible && visible < batch.size() && bytes.compare(start, visible, batch, 0, visible) == 0)
            throw std::runtime_error("pending publication output contains a partial batch: " +
                                     path.u8string());
    }
    // Other workers can have appended since a crashed worker saved its cursor.
    // Keep their bytes and append the missing batch under the same writer lock.
    if (!bytes.empty() && bytes.back() != '\n')
        bytes += '\n';
    bytes += batch;
    ecm_io::write_atomic(path, bytes);
    state = {};
    acknowledge();
}

std::string aggregate_nf(const std::string& report, uint64_t count) {
    auto row = json::parse(report);
    if (!count || row.get("status").text() != "NF" || row.get("curves").text() != "1")
        throw std::runtime_error("invalid Stage2 checkpoint NF report");
    row.set("curves", json::Value::number(std::to_string(count)));
    return with_result_checksum(row.dump());
}

namespace {
size_t stage1_rows(const fs::path& path, const std::function<void(const json::Value&)>& visit) {
    if (!fs::exists(path))
        return 0;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    size_t count = 0;
    if (!in)
        throw std::runtime_error("cannot read Stage1 internal results");
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        auto row = json::parse(line);
        if (stage2_internal_row(row))
            continue;
        ++count;
        visit(row);
    }
    if (in.bad())
        throw std::runtime_error("cannot finish reading Stage1 internal results");
    return count;
}

std::string checked_checkpoint(const fs::path& path) {
    auto bytes = ecm_io::read_file(path);
    const auto split = bytes.find('\n');
    if (split == bytes.npos ||
        bytes.substr(0, split) != std::to_string(ecm_io::fingerprint(bytes.substr(split + 1))))
        throw std::runtime_error("Stage1 publication checkpoint checksum mismatch: " + path.u8string());
    return bytes.substr(split + 1);
}

struct Stage1Publication {
    std::string identity, internal_path, submission_path, internal, submission;
    std::set<uint64_t> early_factors;
    bool internal_done = false, completed = false;
    AppendCheckpoint append;

    void load(const fs::path& path) {
        if (!fs::exists(path))
            return;
        const auto row = json::parse(checked_checkpoint(path));
        if (row.get("schema").text() != "1")
            throw std::runtime_error("unsupported Stage1 publication checkpoint: " + path.u8string());
        identity = row.get("identity").text();
        internal_path = row.get("internal-path").text();
        submission_path = row.get("submission-path").text();
        internal = row.get("internal").text();
        submission = row.get("submission").text();
        internal_done = row.get("internal-done").text() == "1";
        completed = row.get("completed").text() == "1";
        append.pending = row.get("pending").text() == "1";
        append.offset = std::stoull(row.get("offset").text());
        append.anchor = row.get("anchor").text();
        for (const auto& sigma : row.get("early-factors").items)
            early_factors.insert(std::stoull(sigma.text()));
    }

    void save(const fs::path& path) const {
        auto row = json::parse("{}");
        row.set("schema", json::Value::number("1"));
        row.set("identity", json::Value::string(identity));
        row.set("internal-path", json::Value::string(internal_path));
        row.set("submission-path", json::Value::string(submission_path));
        row.set("internal", json::Value::string(internal));
        row.set("submission", json::Value::string(submission));
        row.set("internal-done", json::Value::number(internal_done ? "1" : "0"));
        row.set("completed", json::Value::number(completed ? "1" : "0"));
        row.set("pending", json::Value::number(append.pending ? "1" : "0"));
        row.set("offset", json::Value::number(std::to_string(append.offset)));
        row.set("anchor", json::Value::string(append.anchor));
        json::Value factors;
        factors.type = 'a';
        for (const auto sigma : early_factors)
            factors.items.push_back(json::Value::number(std::to_string(sigma)));
        row.set("early-factors", std::move(factors));
        const auto payload = row.dump();
        ecm_io::write_atomic(path, std::to_string(ecm_io::fingerprint(payload)) + '\n' + payload);
    }
};

uint64_t result_sigma(const json::Value& row) {
    const auto text = row.has("Edwards")          ? row.get("Edwards").get("sigma").text()
                      : row.has("GMP-ECM-param3") ? row.get("GMP-ECM-param3").get("sigma").text()
                                                  : row.get("sigma").text();
    if (text.empty() || text.find_first_not_of("0123456789") != text.npos)
        throw std::runtime_error("result has no numeric sigma");
    return std::stoull(text);
}

void result_modulus(const json::Value& row, mpz_ptr saved) {
    ecm_edwards::Integer factor;
    if (row.has("N")) {
        if (mpz_set_str(saved, row.get("N").text().c_str(), 10))
            throw std::runtime_error("invalid saved result modulus");
    } else {
        const auto exponent =
            std::stoull(row.has("exponent") ? row.get("exponent").text() : row.get("n").text());
        if (exponent > 1000000000)
            throw std::runtime_error("oversized saved result target");
        mpz_ui_pow_ui(saved, row.has("exponent") ? 2 : std::stoul(row.get("b").text()),
                      static_cast<unsigned long>(exponent));
        if (!row.has("exponent")) {
            if (mpz_set_str(factor.v, row.get("k").text().c_str(), 10))
                throw std::runtime_error("invalid saved result k");
            mpz_mul(saved, saved, factor.v);
        }
        if (mpz_set_str(factor.v, row.has("exponent") ? "-1" : row.get("c").text().c_str(), 10))
            throw std::runtime_error("invalid saved result c");
        mpz_add(saved, saved, factor.v);
        for (const auto& known : row.get("known-factors").items) {
            if (mpz_set_str(factor.v, known.text().c_str(), 10) || mpz_cmp_ui(factor.v, 1) <= 0 ||
                !mpz_divisible_p(saved, factor.v))
                throw std::runtime_error("invalid saved result known factor");
            mpz_divexact(saved, saved, factor.v);
        }
    }
}

bool stage1_modulus_matches(const json::Value& row, mpz_srcptr n) {
    ecm_edwards::Integer saved;
    result_modulus(row, saved.v);
    return mpz_cmp(saved.v, n) == 0;
}

std::string stage1_factor_identity(const json::Value& row) {
    if (row.get("status").text() != "F" || row.get("stage").text() != "1")
        return {};
    ecm_edwards::Integer n;
    result_modulus(row, n.v);
    const auto model = row.has("Edwards")
                           ? "ed"
                           : "mont" + row.get("param").text(row.has("GMP-ECM-param3") ? "3" : "0");
    return ecm_edwards::hex(n.v) + "|" + std::to_string(result_sigma(row)) + "|" +
           row.get("b1").text() + "|" + model + "|" + row.get("factors").dump();
}

std::string stage2_factor_identity(const json::Value& row) {
    if (row.get("status").text() != "F")
        return {};
    const auto stage1 = stage1_factor_identity(row);
    if (!stage1.empty())
        return "1|" + stage1;
    ecm_edwards::Integer n;
    result_modulus(row, n.v);
    const auto model = row.has("Edwards") ? "ed" :
        "mont" + row.get("param").text(row.has("GMP-ECM-param3") ? "3" : "0");
    return "2|" + ecm_edwards::hex(n.v) + "|" + std::to_string(result_sigma(row)) + "|" +
        row.get("b1").text() + "|" + row.get("b2").text() + "|" + model + "|" +
        row.get("factors").dump();
}

bool stage1_work_matches(const json::Value& row, const Submission& r, mpz_srcptr n) {
    return row.get("b1").text() == std::to_string(r.b1) && row.has("Edwards") == r.edwards &&
           (r.edwards || row.get("param").text(row.has("GMP-ECM-param3") ? "3" : "0") ==
                             std::to_string(r.param)) &&
           stage1_modulus_matches(row, n);
}

bool contains_stage1_output(const fs::path& path, mpz_srcptr n, uint64_t b1, bool edwards,
                            unsigned param, const std::vector<uint64_t>& sigmas) {
    std::set<uint64_t> wanted(sigmas.begin(), sigmas.end());
    if (wanted.empty())
        return false;
    Submission work;
    work.b1 = b1;
    work.edwards = edwards;
    work.param = int(param);
    stage1_rows(path, [&](const json::Value& row) {
        const auto sigma = result_sigma(row);
        if (wanted.count(sigma) && stage1_work_matches(row, work, n))
            wanted.erase(sigma);
    });
    return wanted.empty();
}


} // namespace

std::string factor_identity(const std::string& report) {
    const auto row = json::parse(report);
    if (row.get("status").text() != "F" ||
        row.get("checksum").dump() != result_with_checksum(row).get("checksum").dump())
        throw std::runtime_error("invalid checkpoint factor report");
    return stage2_factor_identity(row);
}

bool contains_stage1_curves(const fs::path& path, mpz_srcptr n, uint64_t b1, bool edwards,
                            unsigned param, const std::vector<uint64_t>& sigmas) {
    if (!fs::exists(path))
        return false;
    auto lock = lock_result(path);
    return contains_stage1_output(path, n, b1, edwards, param, sigmas);
}

fs::path stage1_publication_checkpoint(const fs::path& arithmetic) {
    return fs::u8path(arithmetic.u8string() + ".publish.ckpt");
}

void discard_completed_stage1_publication(const fs::path& checkpoint) {
    if (!fs::exists(checkpoint))
        return;
    auto lock = lock_result(checkpoint);
    Stage1Publication state;
    state.load(checkpoint);
    if (state.completed)
        fs::remove(checkpoint);
}

void publish_stage1_batch(const fs::path& internal, const fs::path& submission,
                          std::vector<Stage1BatchRecord> records, mpz_srcptr n,
                          const std::string& identity, const fs::path& checkpoint, bool early) {
    if (records.empty() || identity.empty() || checkpoint.empty() ||
        ecm_io::same_file_resource(internal, submission) ||
        ecm_io::same_file_resource(checkpoint, internal) ||
        ecm_io::same_file_resource(checkpoint, submission))
        throw std::runtime_error("invalid Stage1 publication batch or output paths");
    if (early && (records.size() != 1 || records.front().submission.factors.empty()))
        throw std::runtime_error("early Stage1 publication requires one factor result");
    auto lock = lock_result(checkpoint);
    Stage1Publication state;
    state.load(checkpoint);
    const auto internal_path = ecm_io::file_identity(internal);
    const auto submission_path = ecm_io::file_identity(submission);
    if ((!state.internal_path.empty() && state.internal_path != internal_path) ||
        (!state.submission_path.empty() && state.submission_path != submission_path))
        throw std::runtime_error("Stage1 publication checkpoint belongs to different output paths");
    const auto sigma = records.front().submission.sigma;
    if (early && state.early_factors.count(sigma))
        return;
    if (state.completed) {
        if (!early && state.identity == identity)
            return;
        throw std::runtime_error("Stage1 publication checkpoint belongs to a different completed batch");
    }
    const auto save = [&] { state.save(checkpoint); };
    if (state.identity.empty()) {
        state.identity = identity;
        state.internal_path = internal_path;
        state.submission_path = submission_path;
        std::stable_sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
            return a.submission.sigma < b.submission.sigma;
        });
        Submission nf;
        uint64_t nf_count = 0;
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& record = records[i];
            const auto& r = record.submission;
            if (!r.stage1_only || r.curves != 1 || r.b1 != records.front().submission.b1 ||
                r.edwards != records.front().submission.edwards ||
                r.param != records.front().submission.param ||
                (i && records[i - 1].submission.sigma == r.sigma))
                throw std::runtime_error("invalid Stage1 batch identity");
            if (state.early_factors.count(r.sigma))
                continue;
            const auto row = json::parse(record.internal);
            const auto report = submission_json(r, n);
            if (!row.has("stage1-engine") || result_sigma(row) != r.sigma ||
                !stage1_work_matches(row, r, n) ||
                row.get("factors").dump() != json::parse(report).get("factors").dump())
                throw std::runtime_error("Stage1 internal record differs from submission");
            state.internal += with_result_checksum(row.dump()) + '\n';
            if (r.factors.empty()) {
                nf = r;
                ++nf_count;
            } else {
                state.submission += report;
            }
        }
        if (nf_count) {
            nf.curves = nf_count;
            state.submission += submission_json(nf, n);
        }
        save();
    } else if (state.identity != identity) {
        throw std::runtime_error("Stage1 publication checkpoint has a different pending batch");
    }
    if (!state.internal_done && !state.internal.empty())
        append_checkpointed(internal, state.internal, state.append, save, [&] {
            state.internal_done = true;
            state.internal.clear();
            save();
        });
    const auto acknowledge = [&] {
        state.submission.clear();
        if (early) {
            state.early_factors.insert(sigma);
            state.identity.clear();
            state.internal_done = false;
        } else {
            state.completed = true;
        }
        save();
    };
    if (state.submission.empty()) {
        acknowledge();
    } else {
        // The acknowledgement clears the checkpoint's payload.
        const auto batch = state.submission;
        append_checkpointed(submission, batch, state.append, save, acknowledge);
    }
}
} // namespace ecm_results
