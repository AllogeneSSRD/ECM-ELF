#pragma once

#include "ecm/io/publication.h"
#include "io/files.h"
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>

namespace ecm_handoff {

enum class Status : unsigned { Open, Sealed, Factored };

// Producer recovery and late factors are scoped to a save range.
// Stage2 PART ranges do not depend on producer batch boundaries.
struct Producer {
    Status status = Status::Open;
    std::string modulus;
    uint32_t published_curves = 0, saved_count = 0, prefix_count = 0, pending_count = 0;
    uint64_t saved_start = 0, pending_skip = 0, factor_record = UINT64_MAX;
};

struct CurveResult {
    uint64_t point_key = 0, b2_min = 0, b2 = 0;
    std::string report;
};

struct Publication {
    std::string path, report;
    ecm_results::AppendCheckpoint append;
    bool published = false;
};

struct Ledger {
    std::map<std::string, Producer> producers;
    std::map<std::string, std::map<uint64_t, CurveResult>> curves;
    std::map<std::string, Publication> publications;

    const Producer* cancelled(const std::string& modulus, uint64_t record) const {
        for (const auto& entry : producers) {
            const auto& producer = entry.second;
            if (producer.status == Status::Factored && producer.modulus == modulus &&
                record > producer.saved_start && record - producer.saved_start <= producer.saved_count)
                return &producer;
        }
        return nullptr;
    }

    static Ledger read(const std::filesystem::path& path) {
        Ledger ledger;
        if (!std::filesystem::exists(path)) return ledger;
        std::istringstream input(ecm_io::read_file(path));
        const auto invalid = [&] { return std::runtime_error("invalid shared Stage2 checkpoint: " + path.u8string()); };
        std::string format;
        size_t producers_count, targets_count, publications_count;
        if (!(input >> format >> producers_count >> targets_count >> publications_count) ||
            format != "ECMSTAGE2SHARED1") throw invalid();
        for (size_t i = 0; i < producers_count; ++i) {
            std::string id;
            unsigned status;
            Producer producer;
            if (!(input >> id >> status >> std::quoted(producer.modulus) >> producer.published_curves >>
                  producer.saved_start >> producer.saved_count >> producer.prefix_count >>
                  producer.pending_skip >> producer.pending_count >> producer.factor_record) ||
                status > unsigned(Status::Factored) || producer.modulus.empty() ||
                producer.saved_start > UINT64_MAX - producer.saved_count ||
                producer.prefix_count > producer.saved_count || producer.saved_count > producer.published_curves ||
                (producer.pending_count && (producer.pending_skip < producer.saved_start ||
                 producer.pending_count > producer.saved_count ||
                 producer.pending_skip - producer.saved_start > producer.saved_count - producer.pending_count)) ||
                (producer.factor_record != UINT64_MAX && (producer.factor_record < producer.saved_start ||
                 producer.factor_record - producer.saved_start >= producer.saved_count))) throw invalid();
            producer.status = static_cast<Status>(status);
            if (!ledger.producers.emplace(std::move(id), std::move(producer)).second) throw invalid();
        }
        for (size_t i = 0; i < targets_count; ++i) {
            std::string modulus;
            size_t count;
            if (!(input >> std::quoted(modulus) >> count) || modulus.empty()) throw invalid();
            auto [entry, inserted] = ledger.curves.try_emplace(std::move(modulus));
            if (!inserted) throw invalid();
            for (size_t j = 0; j < count; ++j) {
                uint64_t record;
                CurveResult result;
                if (!(input >> record >> result.point_key >> result.b2_min >> result.b2 >> std::quoted(result.report)) ||
                    !record || result.report.empty() || !entry->second.emplace(record, std::move(result)).second)
                    throw invalid();
            }
        }
        for (size_t i = 0; i < publications_count; ++i) {
            std::string id;
            Publication publication;
            if (!(input >> std::quoted(id) >> publication.published >> std::quoted(publication.path) >>
                  std::quoted(publication.report) >> publication.append.pending >> publication.append.offset >>
                  std::quoted(publication.append.anchor)) || id.empty() ||
                (publication.published ? publication.append.pending : publication.report.empty() || publication.path.empty()) ||
                !ledger.publications.emplace(std::move(id), std::move(publication)).second) throw invalid();
        }
        input >> std::ws;
        if (!input.eof()) throw invalid();
        return ledger;
    }

    void write(const std::filesystem::path& path) const {
        std::ostringstream out;
        out << "ECMSTAGE2SHARED1 " << producers.size() << ' ' << curves.size() << ' ' << publications.size() << '\n';
        for (const auto& [id, p] : producers)
            out << id << ' ' << unsigned(p.status) << ' ' << std::quoted(p.modulus) << ' '
                << p.published_curves << ' ' << p.saved_start << ' ' << p.saved_count << ' '
                << p.prefix_count << ' ' << p.pending_skip << ' ' << p.pending_count << ' ' << p.factor_record << '\n';
        for (const auto& [modulus, rows] : curves) {
            out << std::quoted(modulus) << ' ' << rows.size() << '\n';
            for (const auto& [record, r] : rows)
                out << record << ' ' << r.point_key << ' ' << r.b2_min << ' ' << r.b2 << ' '
                    << std::quoted(r.report) << '\n';
        }
        for (const auto& [id, p] : publications)
            out << std::quoted(id) << ' ' << p.published << ' ' << std::quoted(p.path) << ' '
                << std::quoted(p.report) << ' ' << p.append.pending << ' ' << p.append.offset << ' '
                << std::quoted(p.append.anchor) << '\n';
        ecm_io::write_atomic(path, out.str());
    }
};

inline std::filesystem::path checkpoint_path(std::filesystem::path save) {
    save += ".stage2.ckpt";
    return save;
}

} // namespace ecm_handoff
