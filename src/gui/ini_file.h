#pragma once

// Structured INI access for ecm_gui.
//
// Why this is not the driver's ecm_queue_config_load(): that loader is a one-shot
// "give me the values" pass which drops the file structure. The GUI has to WRITE
// the ini as well (window geometry, NumWorkers, autostart, language...), and the
// rule is (docs/DEV_ECM_GUI.md 4.3):
//
//   * unknown keys, comments, blank lines and the user's own ordering survive;
//   * only the VALUE of a key the GUI owns is ever rewritten;
//   * new keys are appended at the end of their section, new sections at EOF;
//   * the file is replaced atomically (tmp + MoveFileEx) with one .bak generation.
//
// Section semantics mirror the driver (D1): keys before the first [Section] line
// are the global section, stored here under the empty section name "".

#include <string>
#include <vector>

namespace ecmgui {

class IniFile {
public:
    struct Line {
        enum class Kind { Blank, Comment, Section, KeyValue, Other };
        Kind kind = Kind::Blank;
        std::string raw;       // the line as it will be written back (no newline)
        std::string section;   // owning section ("" = global) for KeyValue
        std::string key;       // for KeyValue
        std::string value;     // for KeyValue
    };

    // Reads `path`. Returns false and fills `err` when the file cannot be opened.
    bool load(const std::string &path, std::string &err);
    // Marks the object as "empty file at `path`" (used before the first save).
    void reset(const std::string &path);

    const std::string &path() const { return path_; }
    bool loaded() const { return loaded_; }
    bool dirty() const { return dirty_; }

    // Value lookup: `section` = "" is the global section. Returns `def` when absent.
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

    // Section names in file order ("" first when the file starts with global keys).
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
