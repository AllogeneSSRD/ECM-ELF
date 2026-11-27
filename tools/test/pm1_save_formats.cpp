#include "io/pm1_saves.h"
#include "io/files.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
    const fs::path root = (argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path()) /
        ("pm1-format-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const fs::path owl = root / "p1-37";
    // zlib.crc32(struct.pack('<Q', 3)) = 0xebadd88a; v1 lacks a B2 field.
    const std::string seed = "OWL P1 1 N=1*2^37-1 B1=3 k=1 bits=9 err=2 time=3.25\n" +
        std::string("\x8a\xd8\xad\xeb\x03\x00\x00\x00\x00\x00\x00\x00", 12);
    ecm_io::write_atomic(owl / "37-000000001.p1", seed);
    auto loaded = ecm_pm1::load_owl(root, 37, 3, 9);
    require(loaded && loaded->k == 1 && loaded->residue == std::vector<uint32_t>{3, 0} &&
            loaded->errors == 2 && loaded->seconds == 3.25, "v1 import failed");
    // 3^444 mod M37 = 77413221271, gcd(x-1,M37) = 223.
    const ecm_pm1::OwlSave final{9, 2, 4.5, {0x06300397, 0x12}};
    ecm_pm1::write_owl(root, 37, 3, 17, 9, final);
    const fs::path latest = owl / "37-000000009.p1";
    const std::string v2 = ecm_io::read_file(latest);
    require(v2 == "OWL P1 2 N=1*2^37-1 B1=3 B2=17 k=9 bits=9 err=2 time=4.500000\n" +
        std::string("\x05\x15\xa5\x00\x97\x03\x30\x06\x12\x00\x00\x00", 12), "v2 layout/CRC");
    loaded = ecm_pm1::load_owl(root, 37, 3, 9);
    require(loaded && loaded->k == 9 && loaded->residue == final.residue, "v2 import failed");
    auto damaged = v2;
    damaged.back() ^= 1;
    ecm_io::write_atomic(latest, damaged);
    loaded = ecm_pm1::load_owl(root, 37, 3, 9);
    require(loaded && loaded->k == 1, "damaged checkpoint did not fall back");
    require(!ecm_pm1::load_owl(root, 37, 7, 9), "resumed a different B1");
    const fs::path foreign = owl / "37-000000002.p1";
    const auto seed_body = seed.substr(seed.find('\n') + 1);
    ecm_io::write_atomic(foreign, "OWL P1 1 N=1*2^37-1 B1=7 k=2 bits=9 err=0 time=0.0\n" + seed_body);
    ecm_pm1::write_owl(root, 37, 3, 17, 9, final, 0);
    require(!fs::exists(owl / "37-000000001.p1") && fs::exists(foreign) && fs::exists(latest),
            "checkpoint rotation must preserve other assignments");

    const fs::path save = ecm_pm1::write_prime95(root, 37, 3, final.residue);
    require(save.filename() == "m0000037", "wrong Prime95 filename");
    const std::string prime95 = ecm_io::read_file(save);
    // Independent byte fixture from struct.pack('<IIdIIi12sdIIQQII2I', ...).
    const unsigned char expected[] = {
        0x4b,0x39,0x7a,0x31, 8,0,0,0, 0,0,0,0,0,0,0xf0,0x3f, 2,0,0,0, 37,0,0,0,
        255,255,255,255, 'S','5',0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0xf0,0x3f,
        0xb9,3,0x30,6, 5,0,0,0, 3,0,0,0,0,0,0,0, 3,0,0,0,0,0,0,0, 1,0,0,0,
        2,0,0,0, 0x97,3,0x30,6, 0x12,0,0,0};
    require(prime95 == std::string(reinterpret_cast<const char*>(expected), sizeof(expected)),
            "Prime95 header, giant or checksum differs");
    require(ecm_pm1::write_prime95(root, 37, 3, final.residue) == save, "retry failed");
    bool conflict = false;
    try { ecm_pm1::write_prime95(root, 37, 5, final.residue); }
    catch (const std::runtime_error&) { conflict = true; }
    require(conflict && ecm_io::read_file(save) == prime95, "overwrote a different Prime95 save");
    std::cout << "P-1 save formats OK: " << root << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
