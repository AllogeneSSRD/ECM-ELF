#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ecmgui {

// Storage shared by the generator's editable ImGui field and its parsed input.
class GenInputText {
public:
    GenInputText() : buffer_(4096, '\0') {}

    void set_text(std::string text) {
        text_ = std::move(text);
        if (buffer_.size() < text_.size() + 1) buffer_.resize(text_.size() + 1);
        std::copy(text_.begin(), text_.end(), buffer_.begin());
        buffer_[text_.size()] = '\0';
    }

    void accept_edit() { text_.assign(buffer_.data()); }

    void resize_buffer(std::size_t size) { buffer_.resize(size); }

    std::vector<char> &buffer() { return buffer_; }
    const std::string &text() const { return text_; }

private:
    std::string text_;
    std::vector<char> buffer_;
};

} // namespace ecmgui
