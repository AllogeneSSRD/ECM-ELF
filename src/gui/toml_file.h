#pragma once

// Lossless GUI editor for the known ecm.toml schema. Unchanged lines and comments
// keep their order; rewritten values are encoded according to their schema type.
// New keys are appended to their table, and saves replace the file atomically.

#include <string>
#include <vector>
#include "../ecm/application/configuration/ecm_toml.h"

namespace ecmgui {

class TomlFile {
public:
    using Line = ecm_config::TomlLine;

    // Reads `path`. Returns false with an error for missing or invalid TOML.
    bool load(const std::string &path, std::string &err);
    // Marks the object as "empty file at `path`" (used before the first save).
    void reset(const std::string &path);

    const std::string &path() const { return path_; }
    bool loaded() const { return loaded_; }
    bool dirty() const { return dirty_; }

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
    bool loaded_ = false;
    bool dirty_ = false;
};

} // namespace ecmgui
