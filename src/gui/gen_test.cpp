// gen_test.cpp — unit test for the worktodo generator (M6 scope A) plus an --emit mode used
// by tools/test/test_gui_generator.ps1 for the byte-for-byte comparison with
// tools/ecm_worktodo/ecm.py.
//
// Everything here is a pure function over strings and files: no window, no GPU, no ImGui.

#include "worktodo_gen.h"
#include "gen_input_text.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ecmgui;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const std::string &name, const std::string &detail = "") {
    if (ok) {
        ++g_pass;
        std::printf("  [ok]   %s\n", name.c_str());
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s%s\n", name.c_str(),
                    detail.empty() ? "" : (" -- " + detail).c_str());
    }
}

std::string read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::string();
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The `--gpu-info` report of a 60-SM card (RTX 4070 Ti), trimmed to a few tiers: real
// numbers, so the recommendations below are the ones the driver would make.
const char *kGpuInfo =
    "gpu_info=1\n"
    "backend=CUDA/CGBN\n"
    "device=0\n"
    "name=NVIDIA GeForce RTX 4070 Ti\n"
    "sm_count=60\n"
    "cc=8.9\n"
    "curve_model=mont\n"
    "gpu_param=0\n"
    "fold=0\n"
    "carry_bits=6\n"
    "picked=0\n"
    "tier_count=4\n"
    "ini=-\n"
    "worker=1\n"
    "tier bits=128 tpb=128 tpi=4 ipb=32 blocks_per_sm=10 blocks_min=60 curves_min=1920 blocks_wave=600 curves_wave=19200\n"
    "tier bits=256 tpb=128 tpi=4 ipb=32 blocks_per_sm=9 blocks_min=60 curves_min=1920 blocks_wave=540 curves_wave=17280\n"
    "tier bits=768 tpb=128 tpi=8 ipb=16 blocks_per_sm=9 blocks_min=60 curves_min=960 blocks_wave=540 curves_wave=8640\n"
    "tier bits=1280 tpb=128 tpi=8 ipb=16 blocks_per_sm=9 blocks_min=60 curves_min=960 blocks_wave=540 curves_wave=8640\n";

const char *kEdwardsGpuInfo =
    "gpu_info=1\ncurve_model=ed\nsm_count=60\ngpu_param=-1\ncarry_bits=1\n"
    "tier bits=512 tpb=128 tpi=4 ipb=32 blocks_per_sm=0 blocks_min=60 curves_min=1920\n"
    "tier bits=768 tpb=128 tpi=8 ipb=16 blocks_per_sm=0 blocks_min=60 curves_min=960\n";

// Two assignments of very different size (521 bits -> 768 tier, 1019 bits -> 1280 tier).
const char *kInput =
    "# a comment, skipped\n"
    "ECM2=AID0000000000000000000000000001,1,2,521,-1,110e6,0,120,\"3,5\"\n"
    "ECM2=1,2,1019,-1,1e6,0,8\n"
    "ECMSTAGE2=1,2,101,-1,\"x_1e3.save\",0,0,1\n"
    "this is not an assignment\n";

GenOptions base_options() {
    GenOptions o;
    o.valid = true;
    o.sort_by = "n";
    o.worker_devices = {{1, 0}, {2, 0}, {3, 1}};
    o.target_device = 0;
    return o;
}

} // namespace

int main(int argc, char **argv) {
    // ---- --emit mode (used by the byte-for-byte test) --------------------------------
    //   ecm_gui_gen_test.exe --emit <input> [--curves N] [--save-pattern P] [--skip N]
    //                        [--sort-by F] [--no-dedup] [--set-b1 V] [--set-b2 V]
    //                        [--set-has-na] [--workers "1=0,2=0"] [--device D] [--write F]
    std::string emit_input, save_pattern = "m{n}_{b1}.save", sort_by = "n", set_b1, set_b2;
    std::string workers = "1=0", write_to;
    long long curves = -1, skip = 0;
    bool dedup = true, set_has_na = false;
    int device = 0;
    bool emit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_next = (i + 1 < argc);
        if (a == "--emit" && has_next) { emit = true; emit_input = argv[++i]; }
        else if (a == "--curves" && has_next) curves = std::stoll(argv[++i]);
        else if (a == "--save-pattern" && has_next) save_pattern = argv[++i];
        else if (a == "--skip" && has_next) skip = std::stoll(argv[++i]);
        else if (a == "--sort-by" && has_next) sort_by = argv[++i];
        else if (a == "--set-b1" && has_next) set_b1 = argv[++i];
        else if (a == "--set-b2" && has_next) set_b2 = argv[++i];
        else if (a == "--workers" && has_next) workers = argv[++i];
        else if (a == "--device" && has_next) device = std::stoi(argv[++i]);
        else if (a == "--write" && has_next) write_to = argv[++i];
        else if (a == "--no-dedup") dedup = false;
        else if (a == "--set-has-na") set_has_na = true;
    }

    if (emit) {
        const std::string text = read_file(emit_input);
        if (text.empty() && !emit_input.empty()) {
            std::fprintf(stderr, "cannot read %s\n", emit_input.c_str());
            return 2;
        }
        GpuProfile gpu;
        std::string err;
        parse_gpu_info(kGpuInfo, gpu, err);

        GenOptions opt = base_options();
        opt.save_pattern = save_pattern;
        opt.normalize_curve_prefix = false; // legacy Python byte-comparison mode
        opt.skip_curves = skip;
        opt.curves_fixed = curves;
        opt.use_recommended = (curves < 0);
        opt.sort_by = sort_by;
        opt.dedup = dedup;
        opt.set_b1 = set_b1;
        opt.set_b2 = set_b2;
        opt.set_has_na = set_has_na;
        opt.target_device = device;
        opt.worker_devices.clear();
        {
            std::stringstream ss(workers);
            std::string item;
            while (std::getline(ss, item, ',')) {
                const std::size_t eq = item.find('=');
                if (eq == std::string::npos) continue;
                opt.worker_devices.push_back(
                    {std::stoi(item.substr(0, eq)), std::stoi(item.substr(eq + 1))});
            }
        }
        const GenResult r = generate(text, gpu, opt);
        if (!r.ok) {
            std::fprintf(stderr, "generator error: %s\n", r.error.c_str());
            return 1;
        }
        std::fwrite(r.text.data(), 1, r.text.size(), stdout);
        if (!write_to.empty()) {
            std::ofstream out(write_to, std::ios::binary | std::ios::trunc);
            out.write(r.text.data(), static_cast<std::streamsize>(r.text.size()));
        }
        return 0;
    }

    std::printf("worktodo generator unit test\n");

    // ---- --gpu-info parsing ----------------------------------------------------------
    std::printf("[1] --gpu-info parsing\n");
    {
        GpuProfile gpu;
        std::string err;
        check(parse_gpu_info(kGpuInfo, gpu, err), "a real report parses", err);
        check(gpu.valid && gpu.sm_count == 60, "sm_count is read");
        check(gpu.tiers.size() == 4, "four tiers are read",
              std::to_string(gpu.tiers.size()));
        check(gpu.tiers[0].bits == 128 && gpu.tiers[0].ipb == 32, "the first tier is 128/32");
        check(gpu.tiers[3].bits == 1280 && gpu.tiers[3].ipb == 16, "the last tier is 1280/16");
        check(gpu.name == "NVIDIA GeForce RTX 4070 Ti", "the device name is read", gpu.name);
        check(gpu.carry_bits == 6, "carry_bits is read");

        // pick(): the kernel's own rule (smallest tier with bits >= n + carry).
        check(gpu.pick(521) && gpu.pick(521)->bits == 768, "521 bits -> the 768 tier");
        check(gpu.pick(1019) && gpu.pick(1019)->bits == 1280, "1019 bits -> the 1280 tier");
        check(gpu.pick(64) && gpu.pick(64)->bits == 128, "64 bits -> the 128 tier");
        check(gpu.pick(1) && gpu.pick(1)->bits == 128, "tiny N still gets a tier");
        GpuProfile tiny;
        tiny.tiers = gpu.tiers;
        tiny.carry_bits = 6;
        check(tiny.pick(20000) == nullptr, "no tier -> nullptr");

        GpuProfile na;
        check(!parse_gpu_info("gpu_info=not_applicable\nbackend=OpenCL\nreason=not applicable to the OpenCL backend\n",
                              na, err),
              "the OpenCL 'not_applicable' answer is rejected");
        check(na.not_applicable, "and it is flagged as such", err);
        GpuProfile junk;
        check(!parse_gpu_info("hello\n", junk, err), "junk is rejected", err);
        check(parse_gpu_info(kGpuInfo, gpu, err) && gpu.valid, "the report is still parsed");
    }

    // ---- effective bits -------------------------------------------------------------
    std::printf("[2] effective bits (no bignum in the GUI)\n");
    {
        int bits = 0;
        std::string err;
        check(effective_bits("1", "2", 521, "-1", {}, bits, err) && bits == 521,
              "2^521-1 has 521 bits", std::to_string(bits));
        check(effective_bits("1", "2", 1019, "-1", {}, bits, err) && bits == 1019,
              "2^1019-1 has 1019 bits", std::to_string(bits));
        check(effective_bits("3", "2", 100, "1", {}, bits, err) && bits == 102,
              "3*2^100+1 has 102 bits", std::to_string(bits));
        // (2^677-1)/1943118631: subtracting the factor's bits must drop the tier.
        check(effective_bits("1", "2", 677, "-1", {"1943118631"}, bits, err) && bits < 677 && bits > 640,
              "known factors shrink the estimate", std::to_string(bits));
        check(effective_bits("1", "2", 677, "-1", {"not-a-number"}, bits, err) == false,
              "a non-numeric factor is refused");
    }

    // ---- parsing one assignment -----------------------------------------------------
    std::printf("[3] assignment parsing (mirrors ecm.py)\n");
    {
        GenTask t;
        std::string err;
        check(parse_assignment("ECM2=AID,1,2,521,-1,110e6,0,120,\"3,5\"", 0, t, err),
              "a full PrimeNet line parses", err);
        check(t.aid == "AID" && t.k == "1" && t.b == "2" && t.n == 521 && t.c == "-1",
              "the numeric fields are read");
        check(t.b1 == "110e6", "B1 keeps its spelling", t.b1);
        check(t.curves == 120, "curves are read", std::to_string(t.curves));
        check(t.factors.size() == 2 && t.factors[0] == "3", "known factors are split");

        GenTask t2;
        check(parse_assignment("ECM=FFT2=4096,1,2,991,-1,1e4", 1, t2, err),
              "FFT2= and a missing AID are handled", err);
        check(t2.fft2 == "4096" && t2.aid.empty(), "FFT2 is captured, AID stays empty");

        GenTask t3;
        check(!parse_assignment("ECM2=garbage", 2, t3, err), "too few fields is an error", err);
        check(!parse_assignment("ECM2=1,2,-5,-1,1e3", 3, t3, err), "a negative n is an error", err);
        check(!parse_assignment("ECM2=1,2,521,-1,0", 4, t3, err), "B1 = 0 is an error", err);
        check(!parse_assignment("SOMETHING=1,2,3", 5, t3, err), "another prefix is an error");
    }

    // ---- save names -----------------------------------------------------------------
    std::printf("[4] save names\n");
    {
        GenTask t;
        std::string err;
        parse_assignment("ECM2=1,2,521,-1,110e6", 0, t, err);
        check(render_save_name("m{n}_{b1}.save", t) == "m521_110e6.save",
              "the legacy custom pattern renders", render_save_name("m{n}_{b1}.save", t));
        t.bits=521;
        check(render_save_name("{curve}_{target}_{b1}.save",t,"e")=="e_m521_110e6.save",
              "Edwards Mersenne save name uses e and original exponent");
        check(render_save_name("{curve}_{target}_{b1}.save",t,"m2")=="m2_m521_110e6.save" &&
              render_save_name("{curve}_{target}_{b1}.save",t,"m3")=="m3_m521_110e6.save",
              "Montgomery parameterizations use distinct save prefixes");
        t.k="3";t.c="1";t.bits=9;t.b1="11000000";
        check(render_save_name("{curve}_{target}_{b1}.save",t,"m0")=="m0_bit9_11000000.save",
              "generic target uses estimated bits and preserves the input B1 spelling");
        check(check_save_name("m521_110e6.save") == nullptr, "a good name passes");
        check(check_save_name("m521.save") != nullptr, "a name without _<B1> fails");
        check(check_save_name("m521_.save") != nullptr, "an empty B1 token fails");
        check(check_save_name("m521_abc.save") != nullptr, "a non-numeric B1 token fails");
        check(check_save_name("m521_0.save") != nullptr, "B1 = 0 fails");
        GenOptions generic=base_options();generic.use_recommended=false;generic.curves_fixed=8;
        const GenResult generated=generate("ECM2=3,2,7,1,1e5\n",GpuProfile{},generic);
        check(generated.ok && generated.text.find("\"m0_bit9_1e5.save\"")!=std::string::npos &&
              !generated.warnings.empty(),
              "non-Mersenne default name uses estimated bits and warns",generated.text);
        generic.curve_type="m1";
        check(!generate("ECM2=3,2,7,1,1e5\n",GpuProfile{},generic).ok,
              "unsupported construction is refused");
    }

    // ---- the pipeline ---------------------------------------------------------------
    std::printf("[5] pipeline\n");
    {
        GpuProfile gpu;
        std::string err;
        parse_gpu_info(kGpuInfo, gpu, err);
        GenOptions opt = base_options();
        const GenResult r = generate(kInput, gpu, opt);
        check(r.ok, "the pipeline succeeds", r.error);
        check(r.skipped_comment == 1, "the comment is skipped");
        check(r.skipped_other == 1, "the ECMSTAGE2= line is skipped (output format, scope A)",
              std::to_string(r.skipped_other));
        check(r.skipped_unknown == 1, "the prose line is skipped",
              std::to_string(r.skipped_unknown));
        check(r.parse_errors.empty(), "a line that is not an assignment is not an error");
        check(r.tasks.size() == 2, "two tasks remain", std::to_string(r.tasks.size()));
        check(r.tasks[0].n == 521 && r.tasks[1].n == 1019, "sorted by n ascending");
        // Recommended curves: n_blocks_per_sm(2) * sm_count(60) * ipb, but never below the
        // kernel's own curves_min for that tier.
        check(r.tasks[0].curves_out == 2 * 60 * 16, "521 bits -> 2*60*16 curves",
              std::to_string(r.tasks[0].curves_out));
        check(r.tasks[1].curves_out == 2 * 60 * 16, "1019 bits -> the 1280 tier as well",
              std::to_string(r.tasks[1].curves_out));
        // Two workers share device 0 -> one segment each, round-robin in sorted order.
        check(r.segments.size() == 2, "two worker segments", std::to_string(r.segments.size()));
        check(r.segments[0].worker == 1 && r.segments[1].worker == 2, "workers 1 and 2");
        check(r.segments[0].lines.size() == 1 && r.segments[1].lines.size() == 1,
              "one line each");
        check(r.text.find("[Worker #1]\r\n") == 0, "the text starts with the worker 1 header");
        check(r.text.find("[Worker #3]") == std::string::npos,
              "no section for a worker with no tasks");
        check(r.text.find("ECMSTAGE1=AID0000000000000000000000000001,1,2,521,-1,\"m0_m521_110e6.save\",0,0,1920,\"3,5\"") != std::string::npos,
              "the emitted line keeps the AID, the save name and the factors", r.text);
        GpuProfile ed_gpu;
        check(parse_gpu_info(kEdwardsGpuInfo, ed_gpu, err), "Edwards GPU tiers parse", err);
        check(ed_gpu.pick(512) && ed_gpu.pick(512)->bits == 768,
              "Edwards needs one spare bit at the 512-bit boundary");
        GenOptions e=opt;e.curve_type="e";e.save_pattern="m{n}_{b1}.save";
        const GenResult er=generate("ECM2=1,2,521,-1,1e5\n",ed_gpu,e);
        check(er.ok && er.text.find("\"e_m521_1e5.save\"")!=std::string::npos,
              "a custom pattern still uses the selected curve prefix",er.text);
        check(er.ok && er.tasks[0].curves_out == 1920,
              "Edwards recommendation uses Edwards TPI", er.error);
        check(!generate("ECM2=1,2,521,-1,1e5\n",gpu,e).ok,
              "Edwards cannot use Montgomery tier data");
        check(r.text.find("\r\n\r\n") != std::string::npos, "sections end with a blank line");
    }

    // ---- per-line tier: a bigger N really gets its own tier --------------------------
    std::printf("[6] per-line tier selection\n");
    {
        GpuProfile gpu;
        std::string err;
        parse_gpu_info(kGpuInfo, gpu, err);
        // 1019 bits -> 1280 tier (ipb 16); 1500 bits -> no tier in this trimmed report.
        GenOptions opt = base_options();
        opt.worker_devices = {{1, 0}};
        const GenResult ok = generate("ECM2=1,2,1019,-1,1e6\n", gpu, opt);
        check(ok.ok, "a 1019-bit task resolves", ok.error);
        const GenResult too_big = generate("ECM2=1,2,5000,-1,1e6\n", gpu, opt);
        check(!too_big.ok && too_big.error.find("no kernel tier") != std::string::npos,
              "a task no tier covers is refused", too_big.error);
        // The OpenCL answer must not silently produce 0-curve lines.
        GpuProfile na;
        check(!generate("ECM2=1,2,521,-1,1e6\n", na, opt).ok,
              "without a GPU profile the recommendation is refused");
        opt.use_recommended = false;
        opt.curves_fixed = 192;
        const GenResult fixed = generate("ECM2=1,2,521,-1,1e6\n", na, opt);
        check(fixed.ok && fixed.tasks[0].curves_out == 192,
              "an explicit curve count needs no GPU", fixed.error);
    }

    // ---- filters / dedup / rewrites ---------------------------------------------------
    std::printf("[7] filters, dedup, rewrites\n");
    {
        GpuProfile gpu;
        std::string err;
        parse_gpu_info(kGpuInfo, gpu, err);
        GenOptions opt = base_options();
        opt.use_recommended = false;
        opt.curves_fixed = 8;

        // Same (k,b,n,c) twice: the higher B1 wins, factors are merged.
        const char *dup =
            "ECM2=1,2,521,-1,1e5,0,8\n"
            "ECM2=1,2,521,-1,1e6,0,8,\"7\"\n"
            "ECM2=1,2,700,-1,1e5,0,8\n";
        const GenResult r = generate(dup, gpu, opt);
        check(r.ok && r.tasks.size() == 2, "the duplicate is removed", r.error);
        check(r.duplicates == 1, "the duplicate is counted");
        check(r.tasks[0].b1 == "1e6", "the higher B1 won", r.tasks[0].b1);
        check(r.tasks[0].factors.size() == 1 && r.tasks[0].factors[0] == "7",
              "the known factors were merged");

        GenOptions f = opt;
        f.min_n = 600;
        const GenResult rf = generate(dup, gpu, f);
        check(rf.ok && rf.tasks.size() == 1 && rf.tasks[0].n == 700, "min_n filters");
        // Filters run BEFORE dedup (ecm.py's order), so both 521 rows are filtered.
        check(rf.filtered == 2, "the filter is counted per row", std::to_string(rf.filtered));

        GenOptions w = opt;
        w.set_b1 = "2e6";
        w.set_b2 = "1e9";
        w.set_has_na = true;
        const GenResult rw = generate("ECM2=1,2,521,-1,1e5\n", gpu, w);
        // ECMSTAGE1= has NO B1 field: the driver takes B1 from the save name, so --set-b1
        // shows up in the save name and in nothing else.
        check(rw.ok && rw.text.find("_2e6.save\"") != std::string::npos,
              "set_b1 lands in the save name", rw.text);
        check(rw.text.find(",1e9,0,8") != std::string::npos, "set_b2 is written", rw.text);
        w.set_b2 = "0";
        const GenResult r0 = generate("ECM2=1,2,521,-1,1e5,1e9,8\n", gpu, w);
        check(r0.ok && r0.text.find(",0,0,8") != std::string::npos,
              "B2=0 overrides a nonzero source B2", r0.text);
        w.set_b2 = "-1";
        check(!generate("ECM2=1,2,521,-1,1e5\n", gpu, w).ok,
              "negative B2 override is rejected");
        check(rw.text.find("ECMSTAGE1=N/A,1,2,521") != std::string::npos,
              "set_has_na writes N/A", rw.text);

        GenOptions s = opt;
        s.sort_by = "n";
        s.sort_desc = true;
        const GenResult rs = generate("ECM2=1,2,521,-1,1e5\nECM2=1,2,700,-1,1e5\n", gpu, s);
        check(rs.ok && rs.tasks[0].n == 700, "--desc reverses the order");

        GenOptions bad = opt;
        bad.sort_by = "nope";
        check(!generate("ECM2=1,2,521,-1,1e5\n", gpu, bad).ok, "an unknown sort field is refused");
    }

    // ---- apply: the mtime guard ------------------------------------------------------
    std::printf("[8] apply append + mtime guard\n");
    {
        const std::string path = "ecm_gui_gen_test_target.txt";
        std::remove(path.c_str());
        FileStamp stamp;
        check(!stamp_file(path, stamp) && !stamp.exists, "a missing file has no stamp");

        std::string err;
        check(apply_append(path, "A\r\n", stamp, err), "the first append works", err);
        check(read_file(path) == "A\r\n", "and wrote exactly the payload", read_file(path));

        FileStamp s1;
        check(stamp_file(path, s1) && s1.exists, "the file now has a stamp");
        // Someone else touches the file after the preview -> refuse.
        {
            std::ofstream extra(path, std::ios::binary | std::ios::app);
            extra << "SOMEBODY ELSE\r\n";
        }
        check(!apply_append(path, "B\r\n", s1, err), "a changed target is refused", err);
        check(err.find("changed") != std::string::npos, "and the reason says so", err);

        FileStamp s2;
        stamp_file(path, s2);
        check(apply_append(path, "B\r\n", s2, err), "after a fresh stamp it appends", err);
        check(read_file(path) == "A\r\nSOMEBODY ELSE\r\nB\r\n", "the file keeps its old lines",
              read_file(path));

        // A file that does not end with a newline gets a separating one (ecm.py's rule).
        const std::string path2 = "ecm_gui_gen_test_target2.txt";
        {
            std::ofstream out(path2, std::ios::binary | std::ios::trunc);
            out << "LASTLINE-NO-NEWLINE";
        }
        FileStamp s3;
        stamp_file(path2, s3);
        check(apply_append(path2, "NEXT\r\n", s3, err), "appending to an unterminated file", err);
        check(read_file(path2) == "LASTLINE-NO-NEWLINE\r\nNEXT\r\n", "a newline was inserted",
              read_file(path2));
        check(!apply_append(path2, "", s3, err), "an empty payload is refused");
        std::remove(path.c_str());
        std::remove(path2.c_str());

        const std::string section_path = "ecm_gui_gen_test_sections.txt";
        {
            std::ofstream out(section_path, std::ios::binary | std::ios::trunc);
            out << "# shared queue\r\n[Worker #1]\r\nOLD1\r\n\r\n"
                   "[Worker #2]\r\nOLD2\r\n";
        }
        FileStamp section_stamp;
        stamp_file(section_path, section_stamp);
        const std::string section_batch =
            "[Worker #1]\r\nNEW1\r\n\r\n[Worker #2]\r\nNEW2\r\n\r\n";
        check(apply_append(section_path, section_batch, section_stamp, err),
              "a batch can be added to a sectioned queue", err);
        const std::string section_result = read_file(section_path);
        const std::size_t new1 = section_result.find("NEW1");
        const std::size_t worker2 = section_result.find("[Worker #2]");
        check(new1 != std::string::npos && worker2 != std::string::npos && new1 < worker2,
              "worker 1 additions stay inside worker 1's section", section_result);
        check(section_result.find("[Worker #1]", section_result.find("[Worker #1]") + 1) ==
                  std::string::npos &&
                  section_result.find("[Worker #2]", worker2 + 1) == std::string::npos,
              "existing worker headers are reused", section_result);
        FileStamp after_section;
        stamp_file(section_path, after_section);
        check(!apply_append(section_path, "UNROUTED\r\n", after_section, err) &&
                  read_file(section_path) == section_result,
              "a sectionless preview cannot land in an arbitrary worker section", err);
        {
            std::fstream edit(section_path, std::ios::binary | std::ios::in | std::ios::out);
            edit.seekp(2);
            edit.put('X');
        }
        FileStamp changed_section;
        stamp_file(section_path, changed_section);
        check(changed_section.size == after_section.size &&
                  changed_section.hash != after_section.hash &&
                  !apply_append(section_path, section_batch, after_section, err),
              "same-size queue edits after preview are refused", err);
        std::remove(section_path.c_str());

        const std::string repeated_path = "ecm_gui_gen_test_repeated_sections.txt";
        {
            std::ofstream out(repeated_path, std::ios::binary | std::ios::trunc);
            out << "[Worker #2]\r\nFIRST2\r\n[ worker  # 1 ]\r\nOLD1\r\n"
                   "[Worker #2]\r\nLAST2";
        }
        FileStamp repeated_stamp;
        stamp_file(repeated_path, repeated_stamp);
        const std::string repeated_batch =
            "[Worker #1]\r\nNEW1\r\n\r\n[Worker #2]\r\nNEW2\r\n\r\n"
            "[Worker #3]\r\nNEW3\r\n\r\n";
        check(apply_append(repeated_path, repeated_batch, repeated_stamp, err),
              "repeated and missing worker sections accept additions", err);
        const std::string repeated_result = read_file(repeated_path);
        const std::size_t last2 = repeated_result.rfind("[Worker #2]");
        const std::size_t added3 = repeated_result.find("[Worker #3]");
        check(repeated_result.find("FIRST2\r\n[ worker  # 1 ]\r\nOLD1\r\n") !=
                  std::string::npos &&
                  repeated_result.find("NEW1") < last2 &&
                  repeated_result.find("LAST2\r\nNEW2") != std::string::npos &&
                  added3 != std::string::npos && repeated_result.find("NEW2") < added3,
              "new lines use the final matching section and missing workers get a new section",
              repeated_result);
        std::remove(repeated_path.c_str());
    }

    // ---- rejected input is reported, never silently dropped --------------------------
    std::printf("[9] error reporting\n");
    {
        GpuProfile gpu;
        std::string err;
        parse_gpu_info(kGpuInfo, gpu, err);
        GenOptions opt = base_options();
        // ecm.py exits on a broken ECM= line; the GUI keeps the other assignments and says
        // which line was rejected (and why).
        const GenResult r = generate(
            "ECM2=1,2,521,-1,1e5\n"
            "ECM2=1,2,-5,-1,1e5\n"
            "ECM2=1,2,700,-1,1e5\n", gpu, opt);
        check(r.ok, "the good assignments still generate", r.error);
        check(r.tasks.size() == 2, "two of the three lines survive", std::to_string(r.tasks.size()));
        check(r.parse_errors.size() == 1, "the rejected line is reported",
              std::to_string(r.parse_errors.size()));
        check(!r.parse_errors.empty() && r.parse_errors[0].find("invalid exponent") != std::string::npos,
              "with the reason", r.parse_errors.empty() ? "" : r.parse_errors[0]);
    }

    // ---- generator text field: typing and file loads share the same buffer -----------
    std::printf("[10] generator input field\n");
    {
        GenInputText input;
        check(input.buffer().size() > 1, "the empty editor has room for typed characters");
        const std::string typed = "ECM2=1,2,521,-1,1e5";
        std::memcpy(input.buffer().data(), typed.c_str(), typed.size() + 1);
        input.accept_edit();
        check(input.text() == typed, "typing updates the generator input");
        GenOptions edit_opt = base_options();
        edit_opt.use_recommended = false;
        edit_opt.curves_fixed = 8;
        const GenResult typed_result = generate(input.text(), GpuProfile{}, edit_opt);
        check(typed_result.ok && typed_result.tasks.size() == 1 && typed_result.tasks[0].n == 521,
              "typed text reaches the generator");

        const std::string loaded = "ECM2=1,2,700,-1,2e5\n";
        input.set_text(loaded);
        check(input.text() == loaded && std::string(input.buffer().data()) == loaded,
              "loading a file replaces the editor and the generator input");
        const GenResult loaded_result = generate(input.text(), GpuProfile{}, edit_opt);
        check(loaded_result.ok && loaded_result.tasks.size() == 1 && loaded_result.tasks[0].n == 700,
              "loaded text reaches the generator");

        const std::string large(8192, 'A');
        input.set_text(large);
        check(input.buffer().size() >= large.size() + 1 && input.text() == large &&
              std::string(input.buffer().data()) == large,
              "a file larger than the initial editor buffer is preserved");
        input.resize_buffer(input.buffer().size() + 4096);
        check(input.buffer().size() > large.size() + 1 && input.text() == large,
              "the editor buffer can grow while typing");
    }

    std::printf("\npassed: %d   failed: %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
