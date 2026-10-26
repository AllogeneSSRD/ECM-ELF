#include "ecm/montgomery_stage1/cpu/checkpoint.h"
#include "ecm/montgomery_stage1/opencl/checkpoint.h"
#include "io/stage1_checkpoint.h"

#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

static void check(bool value, const char *message) {
    if (!value) { std::cerr << message << '\n'; std::exit(1); }
}

int main(int argc, char **argv) {
    ecm_stage1_checkpoint::Context checkpoint;

    namespace fs = std::filesystem;
    const fs::path dir = argc > 1 ? fs::path(argv[1]) : fs::path("data/experiments/stage1_ckpt_test");
    fs::create_directories(dir);
    const std::string path = (dir / "rotation_c0000000.ckpt").string();
    checkpoint.remove(path);
    checkpoint.set_backup_limit(2);

    mpz_t N;
    mpz_init_set_ui(N, 1019);
    mont_ckpt_t write_state, read_state;
    mont_ckpt_init(&write_state);
    mont_ckpt_init(&read_state);
    mont_ckpt_ident(&write_state, 0, 26, 1000, 1, 100, "mpn");
    write_state.bitnum = 10;
    mpz_set_ui(write_state.X0, 5);
    mpz_set_ui(write_state.Z0, 1);
    mpz_set_ui(write_state.X1, 7);
    mpz_set_ui(write_state.Z1, 1);
    for (size_t bit = 10; bit <= 40; bit += 10) {
        write_state.bitnum = bit;
        check(mont_ckpt_write(checkpoint, path, N, &write_state) == 0, "checkpoint write failed");
    }
    check(fs::exists(path) && fs::exists(ecm_stage1_backup_path(path, 1)) &&
          fs::exists(ecm_stage1_backup_path(path, 2)), "missing rotated generation");
    check(!fs::exists(ecm_stage1_backup_path(path, 3)), "backup limit exceeded");

    // A torn primary must fall back to the latest complete backup (bit 30).
    { std::ofstream corrupt(path, std::ios::binary | std::ios::trunc); corrupt << "torn"; }
    std::string why;
    check(mont_ckpt_read(checkpoint, path, N, 1000, 1, 100, &read_state, &why) == 1,
          "corrupt primary did not fall back");
    check(read_state.bitnum == 30, "recovered the wrong backup generation");
    write_state.bitnum = 50;
    check(mont_ckpt_write(checkpoint, path, N, &write_state) == 0,
          "write after backup recovery failed");
    check(mont_ckpt_read(checkpoint, ecm_stage1_backup_path(path, 1), N, 1000, 1, 100,
                         &read_state, &why) == 1 && read_state.bitnum == 30,
          "corrupt primary replaced the last good backup");

    checkpoint.set_backup_limit(1);
    write_state.bitnum = 60;
    check(mont_ckpt_write(checkpoint, path, N, &write_state) == 0,
          "checkpoint write after reducing backup cap failed");
    check(!fs::exists(ecm_stage1_backup_path(path, 2)),
          "backup above the reduced cap remained");
    checkpoint.set_backup_limit(0);
    write_state.bitnum = 70;
    check(mont_ckpt_write(checkpoint, path, N, &write_state) == 0 &&
          !fs::exists(ecm_stage1_backup_path(path, 1)),
          "zero backup cap did not remove backups");

    // Completion removes the primary and every retained generation.
    checkpoint.remove(path);
    check(!fs::exists(path) && !fs::exists(ecm_stage1_backup_path(path, 1)) &&
          !fs::exists(ecm_stage1_backup_path(path, 2)), "completion left checkpoints");

    const std::string binary = (dir / ".ecm_ckpt_test.dat").string();
    checkpoint.remove(binary);
    checkpoint.set_backup_limit(2);
    opencl_ecm_checkpoint_header_t header{};
    header.magic = OPENCL_ECM_CHECKPOINT_MAGIC;
    header.version = OPENCL_ECM_CHECKPOINT_VERSION;
    header.s_num_bits = 100;
    header.curves = 1;
    header.data_size = 4 * sizeof(uint32_t);
    header.timestamp = std::time(nullptr);
    uint32_t words[4] = {1, 2, 3, 4};
    for (unsigned bit : {10u, 20u, 30u}) {
        header.s_partial = bit;
        check(opencl_ecm_checkpoint_save(checkpoint, binary.c_str(), &header, words, sizeof(words)) == 0,
              "binary checkpoint write failed");
    }
    // Keep the file length valid; only the checksum detects this bit flip.
    {
        std::fstream corrupt(binary, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(sizeof(header) + 1);
        corrupt.put('X');
    }
    opencl_ecm_checkpoint_header_t loaded{};
    uint32_t *payload = nullptr;
    size_t payload_size = 0;
    check(opencl_ecm_checkpoint_load(checkpoint, binary.c_str(), &loaded, &payload, &payload_size) != 0,
          "damaged binary checkpoint passed checksum");
    const auto backup = ecm_stage1_backup_path(binary, 1);
    check(opencl_ecm_checkpoint_load(checkpoint, backup.c_str(), &loaded, &payload, &payload_size) == 0 &&
          loaded.s_partial == 20 && payload_size == sizeof(words),
          "binary backup recovery failed");
    free(payload);
    checkpoint.remove(binary);
    check(!fs::exists(binary) && !fs::exists(backup), "binary backup cleanup failed");

#ifdef _WIN32
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    check(event != nullptr, "stop event creation failed");
    check(ecm_stage1_set_stop_handle(reinterpret_cast<uintptr_t>(event)),
          "stop event was rejected");
    check(!ecm_stage1_stop_requested(), "unexpected stop request");
    SetEvent(event);
    check(ecm_stage1_stop_requested(), "GUI stop event was not observed");
    CloseHandle(event);
#endif
    mont_ckpt_clear(&read_state);
    mont_ckpt_clear(&write_state);
    mpz_clear(N);
    std::cout << "Stage1 rotation, fallback, cleanup and stop event OK\n";
}
