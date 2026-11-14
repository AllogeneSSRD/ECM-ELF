#pragma once

// Lossless GUI editor for the known ecm.toml schema. Unchanged lines and comments
// keep their order; rewritten values are encoded according to their schema type.
// New keys are appended to their table, and saves replace the file atomically.

#include <string>
#include <vector>
#include "../configuration/generated/ecm_config_generated.h"

namespace ecmgui {

// Computation previews are advisory. Reject only the field edited by the user;
// unknown or invalid worker values stay in the TOML for its executable to check.
template<class Values, size_t N>
inline Values preview_values(const ecm_config::Entries& entries,
                             const ecm_config::Binding<Values> (&bindings)[N]) {
    Values result;
    for(const auto& entry:entries) {
        auto candidate=result;
        try { ecm_config::apply(candidate,{entry},bindings);result=std::move(candidate); }
        catch(const std::exception&) {}
    }
    return result;
}
inline ecm_config::Stage1Values preview_stage1(const ecm_config::Entries& entries) {
    return preview_values(entries,ecm_config::stage1_bindings);
}
inline ecm_config::Stage2Values preview_stage2(const ecm_config::Entries& entries) {
    return preview_values(entries,ecm_config::stage2_bindings);
}

class TomlFile {
public:
    using Line = ecm_config::TomlLine;

    // Reads `path`. Returns false with an error for missing or invalid TOML.
    bool load(const std::string &path, std::string &err);
    // Starts a new versioned file at `path` (used before the first save).
    void reset(const std::string &path);

    const std::string &path() const { return path_; }
    bool loaded() const { return loaded_; }
    bool dirty() const { return dirty_; }
    int config_version() const { return config_version_; }

    // Lookup and set use stable binding names. These helpers expose the
    // spelling and values written by the current file version.
    std::pair<std::string, std::string> display_path(const std::string &section,
                                                     const std::string &key) const;
    std::string display_value(const std::string &section, const std::string &key) const;
    std::string binding_value(const std::string &section, const std::string &key,
                              const std::string &value) const;

    // Exact table/key lookup; returns `def` when absent.
    std::string get(const std::string &section, const std::string &key,
                    const std::string &def = std::string()) const;
    bool has(const std::string &section, const std::string &key) const;

    // Replaces the value in place, or appends the key to that section (creating the
    // section if needed). Never reorders or drops anything else.
    void set(const std::string &section, const std::string &key, const std::string &value);

    // Convenience accessors used by the GUI settings.
    int get_int(const std::string &section, const std::string &key, int def) const;
    double get_double(const std::string &section, const std::string &key, double def) const;
    void set_int(const std::string &section, const std::string &key, int value);
    void set_double(const std::string &section, const std::string &key, double value);

    // Table names in file order.
    std::vector<std::string> sections() const;

    // Writes the file back: <path>.tmp -> <path>, keeping one <path>.bak generation.
    // Returns false and fills `err` on failure (the original file is left untouched).
    bool save(std::string &err);

    const std::vector<Line> &lines() const { return lines_; }

private:
    std::string path_;
    std::vector<Line> lines_;
    int config_version_ = 1;
    bool loaded_ = false;
    bool dirty_ = false;
};

} // namespace ecmgui
