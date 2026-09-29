#pragma once

// Localization for ecm_gui: one XML file per language, Notepad++-style layout.
//
//   <?xml version="1.0" encoding="utf-8" ?>
//   <EcmGui>
//     <Native-Langue name="English" filename="english.xml" version="1">
//       <Panel id="workers">
//         <Item id="start" name="Start"/>
//
// Rules (docs/DEV_ECM_GUI.md 10):
//   * localization/english.xml is the mandatory baseline: every key must exist
//     there, and a missing key in another language falls back to it;
//   * files are UTF-8 (with or without BOM); a non-UTF-8 file is reported instead
//     of turning into mojibake;
//   * missing key -> the fallback, then the literal "panel.id" (never empty);
//   * the GUI can reload at runtime (hot reload) -- see Localization::load().

#include <string>
#include <vector>

namespace ecmgui {

class Localization {
public:
    struct Entry {
        std::string panel;
        std::string id;
        std::string text;
    };

    // Loads <dir>/<language>.xml, after <dir>/english.xml (the baseline).
    // `language` is a file stem, e.g. "chineseSimplified"; "english" loads the
    // baseline only. Returns false (with `err`) when the baseline or the requested
    // file cannot be parsed; in that case the previously loaded strings are kept.
    bool load(const std::string &dir, const std::string &language, std::string &err);

    // Text for a key: loaded language -> english baseline -> "panel.id".
    std::string t(const std::string &panel, const std::string &id) const;

    const std::string &language() const { return language_; }
    const std::string &dir() const { return dir_; }
    std::size_t count() const { return current_.size(); }
    std::size_t baseline_count() const { return baseline_.size(); }

    // Keys defined in the baseline but missing from the loaded language: the
    // localization panel lists them so a translator can see what is left.
    std::vector<std::string> missing_keys() const;

    // True when any loaded string contains a non-ASCII byte, i.e. the UI needs a
    // CJK-capable font (see platform.h / load_ui_font).
    bool needs_cjk_font() const;

    // Language file stems (without ".xml") found in `dir`, sorted; "english" first.
    static std::vector<std::string> available(const std::string &dir);

private:
    std::vector<Entry> baseline_;
    std::vector<Entry> current_;
    std::string dir_;
    std::string language_ = "english";
};

// Parses one localization XML file into `out`. Exposed for the self-test.
bool parse_localization_xml(const std::string &path, std::vector<Localization::Entry> &out,
                            std::string &err);

} // namespace ecmgui
