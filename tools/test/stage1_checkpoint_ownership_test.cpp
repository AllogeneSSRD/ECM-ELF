#include "io/stage1_checkpoint.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
    check(argc == 2, "pass fresh directory");
    const auto root = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(root);
    const auto path = (root / "one.ckpt").string();
    ecm_stage1_checkpoint::Context one, two;
    one.set_path(path);
    two.set_path((root / "two.ckpt").string());
    one.begin("one", 2, true, 26);
    two.begin("two", 3, true, 40);
    check(one.sigma(0, 26) == 26 && one.sigma(1, 27) == 27, "first seeds");
    for (unsigned i = 0; i < 3; ++i) check(two.sigma(i, 40 + i) == 40 + i, "second seeds");
    check(one.write_blob(ecm_stage1_curve_record(0), "first"), "first payload");
    check(two.write_blob(ecm_stage1_curve_record(0), "second"), "second payload");
    bool rejected = false;
    try {
        ecm_stage1_checkpoint::Context conflict;
        conflict.set_path(path);
        conflict.begin("one", 2, true, 26);
    } catch (const std::exception&) { rejected = true; }
    check(rejected, "same physical batch has two owners");
    std::string bytes;
    check(one.read_blob(ecm_stage1_curve_record(0), bytes) && bytes == "first", "owners shared records");
    two.close(false);
    check(one.read_blob(ecm_stage1_curve_record(0), bytes) && bytes == "first", "closing another context changed records");
    one.close(false);
    {
        ecm_stage1_checkpoint::Context restored;
        restored.set_path(path);
        restored.begin("one", 2, true, 26);
        check(restored.read_blob(ecm_stage1_curve_record(0), bytes) && bytes == "first", "restore failed");
        restored.close(true);
    }
    check(!std::filesystem::exists(path), "completed owner retained primary");
    std::cout << "checkpoint_ownership=PASS isolated_records=1 begin_failure_releases=1 restore=1\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
