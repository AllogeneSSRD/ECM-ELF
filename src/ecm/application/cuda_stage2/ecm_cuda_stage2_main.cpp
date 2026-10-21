// Standalone save -> CUDA Stage2 driver. Completed curve receipts commit queue
// progress; invalid input tasks are commented out. Each child owns engine state.
#define NOMINMAX
#include <windows.h>
#include "../../backend/cuda/stage2/ecm_cuda_stage2.h"
#include "../../core/stage2/ecm_stage2_modulus.h"
#include "../configuration/ecm_expr.h"
#include "../configuration/ecm_queue_config.h"
#include "../../storage/queue/ecm_worktodo.h"
#include "ecm/platform/program_version.h"
#include "ecm/platform/fingerprint.h"
#include "../../service/planning/stage2.h"
#include "../../service/stage2/task.h"
#include "../../service/stage2/input.h"
#include "../../platform/process/stage2_curve_process.h"
#include "../../backend/cuda/stage2/host_policy.h"
#include "../../service/planning/ecm_stage2_cost_profile.h"
#include "../../service/planning/ecm_stage2_tune_format.h"
#include "../../service/planning/ecm_stage2_tune_ecm.h"
#include "../../service/planning/ecm_stage2_tune_prediction.h"
#include "../../service/planning/ecm_stage2_tune_auto.h"
#include "../../service/planning/ecm_stage2_tune_components.h"
#include "../../service/planning/ecm_stage2_tune_ntt_profile.h"
#include "../../service/planning/ecm_stage1_tune_profile.h"
#include "../../service/planning/ecm_stage2_portable_model.h"
#include "../../service/planning/ecm_stage2_width_prediction.h"
#include "../../service/planning/ecm_stage2_runtime_tune.h"
#include "../../core/stage2/ecm_stage2_geometry.h"
#include "../../platform/ecm_stage2_logging.h"
#include "../../storage/save/ecm_edwards_io.h"
#include "ecm/storage/results/publication.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>

#ifndef NTT_GL_ADD_SUB_MASK
#define NTT_GL_ADD_SUB_MASK 0
#endif

namespace s2prod {
namespace fs = std::filesystem;

// Exact decimal/scientific input: no double rounding for sigma or B2.

double positive(const std::string &s,const char *name) {
    return ecm_config::positive(s,name);
}
Options arguments(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + a);
            return argv[i];
        };
        auto num = [&]() { return u64(value(), a.c_str()); };
        if (a == "--help" || a == "-h") o.help = true;
        else if (a == "--config") o.ini = value();
        else if (a == "--save") o.save = value();
        else if (a == "--worktodo") o.worktodo = value();
        else if (a == "--results") o.results = value();
        else if (a == "--submission") o.submission=value();
        else if (a == "--exponent") o.submission_exponent=num();
        else if (a == "--known-factor") o.submission_known.push_back(value());
        else if (a == "--aid") o.submission_aid=value();
        else if (a == "--log") o.log = value();
        else if (a == "--log-level") o.log_level = stage2_log::parse(value());
        else if (a == "--debug-log-file") { o.debug_file=value(); o.debug_log=true; }
        else if (a == "--queue-receipt") o.receipt=value();
        else if (a == "--publication-task") o.publication_task=value();
        else if (a == "--stage2-stop-handle") o.stop_handle=num();
        else if (a == "--gui-events") o.gui_events=true;
        else if (a == "--b2") o.b2 = num();
        else if (a == "--d") { o.d = num(); o.has_d = true; }
        else if (a == "--carrier-exponent") {
            const auto p=num();
            if(p && (p<2 || p>ecm_stage2::max_input_bits))throw std::runtime_error("carrier-exponent must be 0 or 2..16384");
            o.carrier_exponent=static_cast<unsigned>(p);
            o.has_carrier=true;
        }
        else if (a == "--auto-carrier") {const auto n=num();if(n>1)throw std::runtime_error("auto-carrier must be 0 or 1");o.auto_carrier=n!=0;o.portable_cli.insert(a);}
        else if (a == "--prediction-policy") {o.prediction_policy=value();o.portable_cli.insert(a);}
        else if (a == "--tune-arithmetic") {o.tune_arithmetic=value();o.portable_cli.insert(a);}
        else if (a == "--skip-curves") { o.skip = num(); o.selection = true; }
        else if (a == "--curves") { o.curves = num(); o.selection = true; }
        else if (a == "--batch-mb") { o.batch = num(); o.has_batch = true; }
        else if (a == "--arena-mb") { o.arena = num(); o.has_arena = true; }
        else if (a == "--device" || a == "--worker") {
            const auto n = num();
            if (n > INT_MAX || (a == "--worker" && !n)) throw std::runtime_error("invalid " + a);
            if (a == "--device") o.device = static_cast<int>(n); else o.worker = static_cast<int>(n);
        }
        else if (a == "--dry-run") o.dry = true;
        else if (a == "--plan-only") o.plan_only = true;
        else if (a == "--factorize-hits") o.factorize_hits = true;
        else if (a == "--factor-only") o.factor_only = true;
        else if (a == "--auto-b2") o.auto_b2=true;
        else if (a == "--cost-profile") o.cost_profile=value();
        else if (a == "--tune-profile") o.tune_profile=value();
        else if (a == "--tune-runtime") {const auto n=num();if(n>1)throw std::runtime_error("tune-runtime must be 0 or 1");o.tune_runtime=n!=0;o.portable_cli.insert(a);}
        else if (a == "--tune-runtime-samples") {o.runtime_samples=num();if(o.runtime_samples<2 || o.runtime_samples>1000)throw std::runtime_error("tune-runtime-samples must be 2..1000");o.portable_cli.insert(a);}
        else if (a == "--runtime-remaining-curves") {o.runtime_remaining=num();}
        else if (a == "--runtime-prepare-cache") {const auto n=num();if(n>1)throw std::runtime_error("runtime-prepare-cache must be 0 or 1");o.runtime_prepare_cache=n!=0;}
        else if (a == "--runtime-prepare-hints") {ecm_stage2::tune::portable::RuntimePauseHints hints;hints.assign(value());o.runtime_prepare_hints=hints.text();}
        else if (a == "--runtime-route-hints") {const auto text=value();ecm_stage2::tune::portable::RuntimeRouteHints hints;if(open_runtime_routes(text,hints))o.runtime_route_hints=text;}
        else if (a == "--short-ntt-batch-tune") {const auto n=num();if(n>1)throw std::runtime_error("short-ntt-batch-tune must be 0 or 1");o.short_batch_tune=n!=0;o.portable_cli.insert(a);}
        else if (a == "--ntt-counts") {const auto n=num();if(n>1)throw std::runtime_error("ntt-counts must be 0 or 1");o.ntt_counts=n!=0;o.portable_cli.insert(a);}
        else if (a == "--relocation") {const auto n=num();if(n>1)throw std::runtime_error("relocation must be 0 or 1");o.relocation=n!=0;o.portable_cli.insert(a);}
        else if (a == "--pack-zero") {const auto n=num();if(n>1)throw std::runtime_error("pack-zero must be 0 or 1");o.pack_zero=n!=0;o.portable_cli.insert(a);}
        else if (a == "--tune-ignore") {o.tune_ignore=value();o.portable_cli.insert(a);ecm_stage2::tune::portable::Ignore::parse(o.tune_ignore);}
        else if (a == "--stage1-cost-csv") {o.stage1_csv=value();o.portable_cli.insert(a);}
        else if (a == "--tune-condition-tag") {o.condition_tag=value();o.portable_cli.insert(a);}
        else if (a == "--short-calibration") {const auto n=num();if(n>1)throw std::runtime_error("short-calibration must be 0 or 1");o.short_calibration=n!=0;o.portable_cli.insert(a);}
        else if (a == "--tune-warmup") {const auto n=num();if(n>1)throw std::runtime_error("tune-warmup must be 0 or 1");o.tune_warmup=n!=0;o.portable_cli.insert(a);}
        else if (a == "--inferred-carrier-exponent") {const auto n=num();if(n>16384)throw std::runtime_error("invalid inferred carrier");o.inferred_carrier=unsigned(n);}
        else if (a == "--tune-budget-seconds" || a == "--short-calibration-seconds" || a == "--stage1-cost-mhz" || a == "--target-factor-bits") {
            const auto n=num();o.portable_cli.insert(a);
            if(a=="--tune-budget-seconds"){if(!n||n>86400)throw std::runtime_error("tune budget must be 1..86400");o.tune_budget=n;}
            else if(a=="--short-calibration-seconds"){if(!n||n>600)throw std::runtime_error("short budget must be 1..600");o.short_budget=n;}
            else if(a=="--stage1-cost-mhz"){if(n>10000)throw std::runtime_error("Stage1 MHz must be 0..10000");o.stage1_mhz=n;}
            else {if(n==1||n>16384)throw std::runtime_error("factor bits must be 0 or 2..16384");o.factor_bits=n;}
        }
        else if (a == "--tune-error-limit") {o.tune_error=positive(value(),"tune error limit");if(o.tune_error>1)throw std::runtime_error("tune error limit must be <=1");o.portable_cli.insert(a);}
        else if (a == "--cost-device-info") o.cost_device_info=true;
        else if (a == "--auto-min-b2") o.auto_min=num();
        else if (a == "--auto-max-b2") o.auto_max=num();
        else if (a == "--owner-budget-mb") {o.owner_mb=num();o.has_owner=true;if(o.owner_mb>ecm_config::limits::stage2_stage2_fold_mb_maximum)throw std::runtime_error("invalid owner budget");}
        else if (a == "--stage1-batch") {o.stage1_batch=num();o.has_stage1_batch=true;if(!o.stage1_batch||o.stage1_batch>ecm_config::limits::stage2_stage1_batch_maximum)throw std::runtime_error("invalid Stage1 batch");}
        else if (a == "--stage1-seconds-per-curve") {o.stage1_seconds=positive(value(),"Stage1 seconds");o.has_stage1_seconds=true;}
        else if (a == "--stage1-tune-profile") o.stage1_profile=value();
        else if (a == "--check-stage1-tune-profile") o.check_stage1_profile=value();
        else if (a == "--lcm") {
            const auto construction=value();o.has_stage1_exponent=true;
            if(construction=="1")o.stage1_exponent="lcm";
            else if(construction=="12")o.stage1_exponent="choose12";
            else throw std::runtime_error("lcm must be 1 or 12");
        }
        else if (a == "--stage2-ratio-adjust") {o.ratio_adjust=positive(value(),"Stage2 ratio adjust");o.has_ratio=true;}
        else if (a == "--gp") { o.gp = value(); o.has_gp = true; }
        else if (a == "--factor-timeout") {
            const auto seconds = num();
            if (!seconds || seconds > ecm_config::limits::stage2_stage2_factor_timeout_maximum) throw std::runtime_error("factor-timeout must be 1..600 seconds");
            o.factor_timeout = static_cast<unsigned>(seconds);
            o.has_factor_timeout = true;
        }
        else if (a == "--tune") o.tune = value();
        else if (a == "--ecm-tune-worker") o.tune_child=true;
        else if (a == "--tune-save") {o.tune_save=value();o.tune_options=true;}
        else if (a == "--tune-worktodo") {o.tune_worktodo=value();o.tune_options=true;}
        else if (a == "--tune-merge") {o.tune_merge.push_back(value());o.tune_options=true;}
        else if (a == "--tune-ntt-profile") {o.tune_ntt_profiles.push_back(value());o.tune_options=true;}
        else if (a == "--tune-max-batches") {
            o.tune_max_batches=num();if(o.tune_max_batches>1048576)throw std::runtime_error("tune-max-batches must be 0..1048576");
            o.has_tune_max_batches=true;o.tune_options=true;
        }
        else if (a == "--tune-tail-samples") {
            const auto n=num();if(n>16)throw std::runtime_error("tune-tail-samples must be 0..16");
            o.tune_tail_samples=static_cast<unsigned>(n);o.has_tune_tail_samples=true;o.tune_options=true;
        }
        else if (a == "--tune-carrier-exponent") {
            const auto p=num();if(p<2 || p>16384)throw std::runtime_error("tune-carrier-exponent must be 2..16384");
            o.tune_carrier=static_cast<unsigned>(p);o.tune_options=true;
        }
        else if (a == "--tune-d" || a == "--tune-b2" || a == "--tune-exponents" || a=="--tune-slices") {
            const auto text=value();std::vector<uint64_t> values;size_t start=0;
            for(;;) {const auto end=text.find(',',start);values.push_back(u64(text.substr(start,end==text.npos?end:end-start),a.c_str()));
                if(end==text.npos)break;start=end+1;}
            if(values.size()>64 || std::set<uint64_t>(values.begin(),values.end()).size()!=values.size())throw std::runtime_error("tune grid must contain 1..64 distinct values");
            if(a=="--tune-d"){for(auto d:values)if(d<6 || d%2 || d>200000000)throw std::runtime_error("tune-d must be even, 6..200000000");o.tune_ds=values;}
            else if(a=="--tune-b2"){for(auto b:values)if(b<3 || b>(uint64_t)INT64_MAX-8192)throw std::runtime_error("invalid tune-b2");o.tune_b2s=values;}
            else if(a=="--tune-slices"){for(auto b:values)if(!b || b>65535)throw std::runtime_error("tune-slices must be 1..65535");o.tune_slices=values;o.has_tune_slices=true;}
            else {for(auto p:values)if(p<2 || p>16384)throw std::runtime_error("invalid tune exponent");o.tune_exponents=values;}
            o.tune_options=true;
        }
        else if (a == "--tune-level") {
            const auto level=num();if(level<1 || level>10)throw std::runtime_error("tune-level must be 1..10");
            o.tune_level=static_cast<int>(level);o.tune_options=true;
        }
        else if (a == "--tune-file") { o.tune_file = value(); o.tune_options = true; }
        else if (a == "--length-log2") {
            o.tune_options = true;
            const auto range = value();
            const auto colon = range.find(':');
            const auto first = u64(range.substr(0, colon), "length-log2");
            const auto last = colon == range.npos ? first : u64(range.substr(colon + 1), "length-log2");
            if (first < 3 || last > 27 || first > last)
                throw std::runtime_error("length-log2 must be 3..27 or FIRST:LAST");
            o.tune_first = static_cast<int>(first); o.tune_last = static_cast<int>(last);o.has_tune_lengths=true;
        }
        else if (a == "--tune-repeats") {
            o.tune_options = true; const auto n = num();
            if (!n || n > 1000) throw std::runtime_error("tune-repeats must be 1..1000");
            o.tune_repeats = static_cast<int>(n);o.has_tune_repeats=true;
        }
        else if (a == "--tune-memory-mb") {
            o.has_tune_memory=true;
            o.tune_options = true; o.tune_memory_mb = num();
            if (!o.tune_memory_mb || o.tune_memory_mb > 1048576)
                throw std::runtime_error("invalid tune memory budget in MiB");
        }
        else if (a == "--once") o.once = true;
        else if (a == "--curve-worker") o.child = true;
        else if (a == "--record-offset") o.offset = num();
        else if (a == "--record-hash") o.hash = num();
        else if (a == "--record-index") o.index = num();
        else throw std::runtime_error("unknown option: " + a);
    }
    if(o.prediction_policy!="model" && o.prediction_policy!="paired")throw std::runtime_error("prediction-policy must be model or paired");
    if(o.tune_arithmetic!="both" && o.tune_arithmetic!="generic" && o.tune_arithmetic!="mersenne")throw std::runtime_error("tune-arithmetic must be both, generic or mersenne");
    return o;
}
using Settings=ecm_config::Stage2Values;

Settings stage2_ini(const fs::path &path,int worker) {
    std::vector<ecm_config::TomlLine> lines;
    ecm_config::read_toml(path.string(),lines);
    return ecm_config::read_stage2(ecm_config::cli_entries(lines,worker,true));
}

// Optional diagnostic for the calling process's CUDA context. Use the same
// linked runtime as the CUDA translation unit.
// The schedule bits affect CPU waits, not kernel ordering or arithmetic.

void help() {
    std::cout << "ecm_cuda_stage2 - resume normalized Suyama PARAM=0 or ECM-EDWARDS-AM Stage1 text saves\n"
        "  ecm_cuda_stage2 --save FILE --b2 B2 [--curves N] [--skip-curves N]\n"
        "  ecm_cuda_stage2 [--config ecm.toml] [--worktodo FILE] [--worker N] [--once]\n"
        "Options: --device N --d D --batch-mb MB --arena-mb MB --results INTERNAL.jsonl\n"
        "         --submission results.json.txt --exponent P --known-factor F (repeatable)\n"
        "         --ntt-counts 0|1 (per-curve NTT length table; TOML stage2_ntt_counts)\n"
        "         --relocation 0|1 (optional conservative suffix, original-range replay on hits/nonunits)\n"
        "         --carrier-exponent p (experimental: arithmetic modulo 2^p-1, factors still target saved N; 0=off)\n"
        "         --auto-carrier 0|1 (override stage2_auto_carrier; verified worktodo candidates, tune selection required)\n"
        "Tune: --tune ntt|ecm --tune-level 1..10 --tune-file FILE.toml --tune-repeats N\n"
        "      ECM: configured worktodo by default; --tune-worktodo FILE or --tune-exponents bits,...\n"
        "           --tune-save FILE [--tune-carrier-exponent p]; verified tasks pair ordinary/carrier modes.\n"
        "           --tune-budget-seconds 1800 --tune-error-limit 0.08; incremental final summaries.\n"
        "           --tune-d D,... --tune-b2 B2,...; --tune-warmup 0|1 (default 0); separate ECM processes.\n"
        "           --tune-arithmetic both|generic|mersenne: arithmetic families for explicit width grids.\n"
        "           Per width/B2: start at default D; follow measured improvements through adjacent D.\n"
        "           TOML format 7: matrices with column names in the first row; regenerate older files.\n"
        "           --tune-max-batches N limits G trees per curve; 0 removes the effort limit.\n"
        "      Merge: --tune ecm --tune-merge A.toml --tune-merge B.toml --tune-file combined.toml\n"
        "             Separate measurement conditions; only a strictly faster median replaces measured data.\n"
        "Selection: --tune-profile FILE.toml --tune-ignore gpu,driver,cuda,backend,environment\n"
        "           B1 never gates reuse; unmeasured widths/B2 retain estimate/uncertainty labels.\n"
        "           --prediction-policy model|paired (default model); model supplies a no-tune basic prior.\n"
        "           --tune-runtime 0|1 (default 1): accumulate checked NF curves and test legal carriers first.\n"
        "           --tune-runtime-samples N (2..1000, default 10): per-condition/path observation cap.\n"
        "           --short-calibration 0|1 --short-calibration-seconds 10\n"
        "           --short-ntt-batch-tune 0|1: experimental batch selection for lengths <=2^19 (default 0).\n"
        "           --pack-zero 0|1: fuse Stage2 input packing and zero padding (stage2_pack_zero in TOML).\n"
        "           --stage1-cost-csv FILE --stage1-cost-mhz MHz --target-factor-bits N\n"
        "           Explicit nonzero D locks D; explicit carrier-exponent (including 0) locks arithmetic.\n"
        "           --tune-ntt-profile FILE (repeatable): attach checked NTT summaries to ECM tune/merge.\n"
        "         --log FILE --log-level quiet|curve|phases|batches|debug (0..4)\n"
        "         Production console default: phases; readable file: batches. --dry-run --help\n"
        "         --factorize-hits [--gp gp.exe] [--factor-timeout 30]\n"
        "         --factor-only (skip optional prime-witness naming; raw factors may be composite)\n"
        "Auto B2: --auto-b2 --tune-profile FILE.toml --stage1-seconds-per-curve S\n"
        "         Or: --stage1-tune-profile FILE.toml --stage1-batch N [--lcm 1|12]\n"
        "         Offline validation: --check-stage1-tune-profile FILE.toml\n"
        "         Legacy: --auto-b2 --cost-profile FILE [--stage1-batch N]\n"
        "         [--stage1-seconds-per-curve S] [--stage2-ratio-adjust R]\n"
        "         [--auto-min-b2 B2 --auto-max-b2 B2] [--owner-budget-mb MB]\n"
        "         Portable profiles allow labeled estimates; legacy cost-profile requires a measured matching scope.\n"
        "         --plan-only (does not consume tasks; may run startup calibration unless --short-calibration 0)\n"
        "Tune: --tune ntt --device N [--tune-level 1..10] [--length-log2 3:27] [--tune-slices 1,4,16] [--tune-repeats 5]\n"
        "      [--tune-memory-mb 1024] [--tune-file stage2_tune.toml]\n"
        "      Requires a fixed Goldilocks backend; measures field convolution only.\n"
        "Queue: ECMSTAGE2=[AID,]k,b,n,c,save[,B2-or-zero][,skip][,count][,\"factors\"]\n"
        "TOML: [shared] worktodo; [stage2] stage2_finished, stage2_progress_dir, stage2_log_file;\n"
        "stage2_progress_file imports legacy single-file progress only.\n"
        "     shared tmp_dir/device/verbose; stage2_save_dir/device overrides; stage2_b2, stage2_d,\n"
        "     stage2_batch_mb, stage2_arena_mb, stage2_pack_zero, stage2_results_file, stage2_log_level.\n"
        "     stage2_debug_log=true enables separate diagnostics (stage2_debug_log_file).\n"
        "D=0 chooses automatically; count/curves=0 means all remaining; shortages are clamped.\n"
        "Queue only: resumes completed curves. Ctrl+C finishes current curve; twice stops now.\n"
        "Missing implicit TOML uses defaults. Worker overrides use [workers.N.stage2].\n";
}

// Access known scalar fields in the engine's trusted structured output. Nested
// records are sliced first; keys are never obtained from external profile text.

int driver(Options o) {
    if (o.help) { help(); return 0; }
    if (o.child) return curve_worker(o);
    if(o.stop_handle) {
        inherited_stop_event=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(o.stop_handle));
        if(WaitForSingleObject(inherited_stop_event,0)==WAIT_FAILED)throw std::runtime_error("invalid inherited Stage2 stop event");
    }
    if(!o.check_stage1_profile.empty()) {
        if(!o.save.empty() || !o.worktodo.empty() || !o.tune.empty() || o.auto_b2 || o.plan_only || o.dry ||
           o.b2 || o.has_carrier || !o.tune_profile.empty() || !o.stage1_profile.empty())
            throw std::runtime_error("Stage1 profile validation is independent of curve/queue options");
        const auto p=ecm_stage2::tune::Stage1Profile::load(absolute_from(fs::current_path(),o.check_stage1_profile));
        std::cout<<"{\"type\":\"stage1_profile_checked\",\"samples\":"<<p.samples.size()<<",\"valid\":true}"<<std::endl;
        return 0;
    }
    if(o.tune_child)throw std::runtime_error("ecm-tune-worker requires the internal curve-worker mode");
    if(o.auto_b2&&o.b2)throw std::runtime_error("explicit --auto-b2 conflicts with nonzero --b2");
    if(o.cost_device_info) {
        if(!o.save.empty()||!o.worktodo.empty()||!o.tune.empty()||o.auto_b2||o.plan_only||o.dry||o.b2||o.has_carrier||!o.tune_profile.empty())
            throw std::runtime_error("--cost-device-info is independent of curve/queue options");
        EcmStage2DeviceInfo info;if(ecm_cuda_stage2_device_info(o.device<0?0:o.device,&info))throw std::runtime_error("cannot query cost-profile device");
        std::cout<<"{\"type\":\"cost_device\",\"uuid_hex\":\""<<info.uuid_hex<<"\",\"major\":"<<info.major<<",\"minor\":"<<info.minor
            <<",\"runtime\":"<<info.runtime<<",\"driver\":"<<info.driver<<",\"fixed_mode\":"<<info.fixed_mode<<",\"outer_unroll_u\":"<<info.outer_unroll_u
            <<",\"free_bytes\":"<<info.free_bytes<<",\"total_bytes\":"<<info.total_bytes<<"}"<<std::endl;return 0;
    }
    if (o.tune.empty() && o.tune_options) throw std::runtime_error("tune options require --tune ntt|ecm");
    if (!o.tune.empty() && ((o.tune != "ntt" && o.tune!="ecm") || !o.save.empty() || !o.worktodo.empty() ||
        o.b2 || o.has_d || o.selection || o.dry || o.plan_only || o.once || o.auto_b2 || !o.cost_profile.empty() || !o.tune_profile.empty() || o.has_carrier))
        throw std::runtime_error("--tune ntt|ecm is independent of save/queue/curve planning options; ECM accepts --tune-save");
    if(o.tune=="ntt" && (!o.tune_save.empty() || !o.tune_worktodo.empty() || o.tune_carrier || !o.tune_ds.empty() || !o.tune_b2s.empty() || !o.tune_exponents.empty() || o.has_tune_max_batches || o.has_tune_tail_samples || !o.tune_merge.empty() || !o.tune_ntt_profiles.empty()))throw std::runtime_error("ECM tune grid/merge options require --tune ecm");
    if(o.tune=="ecm" && o.has_tune_lengths)throw std::runtime_error("length-log2 requires --tune ntt");
    if(o.tune=="ecm" && o.has_tune_slices)throw std::runtime_error("tune-slices requires --tune ntt");
    if(o.tune=="ecm" && o.has_tune_memory)throw std::runtime_error("ECM tune uses batch/arena/owner budgets; tune-memory-mb requires --tune ntt");
    if (o.plan_only && o.dry) throw std::runtime_error("choose --plan-only or --dry-run");
    const fs::path cwd = fs::current_path();
    const fs::path ini = o.ini.empty() ? executable().parent_path() / "ecm.toml" : absolute_from(cwd, o.ini);
    EcmQueueConfig cfg;
    if (!ecm_queue_config_load(ini.string(), o.worker, cfg) && (!o.ini.empty() || fs::exists(ini)))
        throw std::runtime_error("cannot read explicit TOML: " + ini.string());
    Settings s = stage2_ini(ini, o.worker);
    const PackZeroDefaults pack_defaults{s.has_pack_zero,s.pack_zero,s.cost_profile};
    configure_pack_zero(o,&pack_defaults);
    if(!o.portable_cli.count("--auto-carrier"))o.auto_carrier=s.auto_carrier;
    if(!o.portable_cli.count("--prediction-policy"))o.prediction_policy=s.prediction_policy;
    if(!o.portable_cli.count("--tune-arithmetic"))o.tune_arithmetic=s.tune_arithmetic;
    if(!o.portable_cli.count("--tune-runtime"))o.tune_runtime=s.tune_runtime;
    if(!o.portable_cli.count("--tune-runtime-samples"))o.runtime_samples=s.runtime_samples;
    if(o.prediction_policy!="model" && o.prediction_policy!="paired")throw std::runtime_error("prediction-policy must be model or paired");
    if(o.tune_arithmetic!="both" && o.tune_arithmetic!="generic" && o.tune_arithmetic!="mersenne")throw std::runtime_error("tune-arithmetic must be both, generic or mersenne");
    if(!o.portable_cli.count("--tune-ignore"))o.tune_ignore=s.tune_ignore;
    if(!o.portable_cli.count("--tune-budget-seconds"))o.tune_budget=s.tune_budget;
    if(!o.portable_cli.count("--tune-warmup"))o.tune_warmup=s.tune_warmup;
    if(!o.portable_cli.count("--short-calibration-seconds"))o.short_budget=s.short_budget;
    if(!o.portable_cli.count("--short-calibration"))o.short_calibration=s.short_calibration;
    if(!o.portable_cli.count("--short-ntt-batch-tune"))o.short_batch_tune=s.short_batch_tune;
    if(!o.portable_cli.count("--ntt-counts"))o.ntt_counts=s.ntt_counts;
    if(!o.portable_cli.count("--relocation"))o.relocation=s.relocation;
    if(o.relocation && (o.tune=="ecm" || o.tune_child))throw std::runtime_error("relocation does not collect legacy ECM tune samples; use --relocation 0 for --tune ecm");
    if(!o.portable_cli.count("--tune-error-limit"))o.tune_error=s.tune_error;
    if(!o.portable_cli.count("--stage1-cost-mhz"))o.stage1_mhz=s.stage1_mhz;
    if(!o.portable_cli.count("--target-factor-bits"))o.factor_bits=s.factor_bits;
    if(!o.portable_cli.count("--tune-condition-tag"))o.condition_tag=s.condition_tag;
    ecm_stage2::tune::portable::Ignore::parse(o.tune_ignore);
    if(o.tune_error>1 || o.factor_bits==1)throw std::runtime_error("invalid portable tune error/factor size");
    if(o.log_level<0)o.log_level=s.log_level<0?(ecm_cuda_stage2_default_log_level()==stage2_log::debug ? stage2_log::debug : (cfg.verbose?stage2_log::phases:stage2_log::curve)):s.log_level;
    o.debug_log=o.debug_log||s.debug_log||o.log_level==stage2_log::debug;
    if(ecm_cuda_stage2_default_log_level()!=stage2_log::debug)o.log_level=std::min(o.log_level,static_cast<int>(stage2_log::batches));
    if(ecm_cuda_stage2_set_log_level(o.log_level))throw std::runtime_error("engine rejected log level (development requires debug)");
    stage2_log::level=o.log_level;
    if(ecm_cuda_stage2_check_configuration())throw std::runtime_error("Stage2 engine configuration rejected");
    if(s.factorize_hits)o.factorize_hits=true;
    if(s.factor_only)o.factor_only=true;
    if(o.factor_only && _putenv_s("NTT_NAME_HITS","0"))throw std::runtime_error("cannot disable optional prime-witness naming");
    o.auto_b2=(o.auto_b2||s.auto_b2) && !cfg.b2_ratio;
    if(!o.auto_min)o.auto_min=s.auto_min;if(!o.auto_max)o.auto_max=s.auto_max;
    if(!o.has_owner){if(s.has_owner)o.owner_mb=s.owner_mb;else if(const char *v=std::getenv("NTT_FOLD_DEVICE_MAX_MB"))o.owner_mb=ecm_stage2::cost::integer(v);}
    if(o.owner_mb>ecm_config::limits::stage2_stage2_fold_mb_maximum)throw std::runtime_error("invalid owner budget");
    if(!o.has_stage1_batch)o.stage1_batch=s.stage1_batch;
    if(!o.has_stage1_seconds)o.stage1_seconds=s.stage1_seconds;
    if(!o.has_stage1_exponent)o.stage1_exponent=cfg.scalar_construction=="12"?"choose12":"lcm";
    if(!o.has_ratio)o.ratio_adjust=s.ratio_adjust;
    if(!o.has_gp && !s.gp.empty())o.gp=s.gp;
    if(!o.has_factor_timeout)o.factor_timeout=static_cast<unsigned>(s.factor_timeout);
    const fs::path base = ini.parent_path();
    o.submission_ini=ini.string();
    if(s.has_submission && s.submission.empty())throw std::runtime_error("empty submission path");
    o.submission=(!o.submission.empty()?absolute_from(cwd,o.submission):
        absolute_from(base,s.has_submission?s.submission:ecm_config::worker_file(s.submission,o.worker))).string();
    if(!o.cost_profile.empty())o.cost_profile=absolute_from(cwd,o.cost_profile).string();
    else if(!s.cost_profile.empty())o.cost_profile=absolute_from(base,s.cost_profile).string();
    if(!o.tune_profile.empty())o.tune_profile=absolute_from(cwd,o.tune_profile).string();
    else if(!s.tune_profile.empty())o.tune_profile=absolute_from(base,s.tune_profile).string();
    else if(o.cost_profile.empty())o.tune_profile=absolute_from(base,"stage2_ecm_tune.toml").string();
    if(!o.stage1_csv.empty())o.stage1_csv=absolute_from(cwd,o.stage1_csv).string();
    else if(!s.stage1_csv.empty())o.stage1_csv=absolute_from(base,s.stage1_csv).string();
    if(!o.stage1_profile.empty())o.stage1_profile=absolute_from(cwd,o.stage1_profile).string();
    else if(!s.stage1_profile.empty())o.stage1_profile=absolute_from(base,s.stage1_profile).string();
    const fs::path worktodo = o.worktodo.empty() ? absolute_from(base, s.worktodo) : absolute_from(cwd, o.worktodo);
    const fs::path finished = absolute_from(base, s.finished);
    const fs::path tmp = absolute_from(base, s.save_dir.empty()?cfg.tmp_dir:s.save_dir);
    const std::string suffix = o.worker == 1 ? "" : "_" + std::to_string(o.worker);
    const fs::path results = !o.results.empty() ? absolute_from(cwd, o.results) :
        absolute_from(base, s.results.empty() ? ecm_config::worker_file(ecm_config::defaults::stage2_stage2_results_file,o.worker) : s.results);
    fs::path log;
    if (!o.log.empty()) log = absolute_from(cwd, o.log);
    else if (!s.has_log) log = absolute_from(base, ecm_config::worker_file(ecm_config::defaults::stage2_stage2_log_file,o.worker));
    else if (!s.log.empty()) log = absolute_from(base, s.log);
    if(o.debug_log) {
        if(o.debug_file.empty())o.debug_file=absolute_from(base,s.debug_file.empty()?ecm_config::worker_file(ecm_config::defaults::stage2_stage2_debug_log_file,o.worker):s.debug_file).string();
        else o.debug_file=absolute_from(cwd,o.debug_file).string();
    }
    const fs::path legacy_progress=s.progress.empty()?fs::path(worktodo.string()+suffix+".progress"):absolute_from(base,s.progress);
    const fs::path progress_dir=!s.progress_dir.empty()?absolute_from(base,s.progress_dir):
        !s.progress.empty()?legacy_progress.parent_path():worktodo.parent_path();
    const int device = o.device < 0 ? (s.device<0?cfg.device:s.device) : o.device;
    const uint64_t d = o.has_d ? o.d : s.d;
    o.device=device;o.d=d;o.has_d=d!=0;
    if (device < 0 || (d && (d < 6 || d % 2))) throw std::runtime_error("device must be >=0; D must be even and >=6");
    const uint64_t batch = o.has_batch ? o.batch : s.batch, arena = o.has_arena ? o.arena : s.arena;
    o.arena=arena;
    o.batch=batch;
    if (batch < ecm_config::limits::stage2_stage2_batch_mb_minimum || batch > ecm_config::limits::stage2_stage2_batch_mb_maximum || arena > ecm_config::limits::stage2_stage2_arena_mb_maximum) throw std::runtime_error("invalid Stage2 memory budget in MB");
    _putenv_s("NTT_FOLD_DEVICE_MAX_MB",std::to_string(o.owner_mb).c_str());
    _putenv_s("NTT_S4_BATCH_MB", std::to_string(batch).c_str());
    _putenv_s("NTT_S4_SHORT_BATCH_TUNE",o.short_batch_tune?"1":"");
    if (arena) _putenv_s("NTT_ARENA_CAP_KB", std::to_string(arena * 1024).c_str());
    if (!o.tune.empty()) {
        if(o.tune_level && o.tune=="ntt") {
            const auto effort=ecm_stage2::tune::effort(o.tune_level);
            if(!o.has_tune_lengths){o.tune_first=effort.first;o.tune_last=effort.last;}
            if(!o.has_tune_repeats)o.tune_repeats=effort.repeats;
            if(!o.has_tune_slices)o.tune_slices.assign(effort.slices.begin(),effort.slices.end());
        }
        const fs::path destination = absolute_from(cwd, o.tune_file.empty() ? (o.tune=="ecm"?"stage2_ecm_tune.toml":"stage2_tune.toml") : o.tune_file);
        auto same_path = [&](const fs::path &other) {
            std::error_code error;
            if (fs::equivalent(destination, other, error) && !error) return true;
            return upper(destination.lexically_normal().string()) == upper(other.lexically_normal().string());
        };
        const bool toml=upper(destination.extension().string())==".TOML";
        if (!toml && upper(destination.extension().string()) != ".JSONL")
            throw std::runtime_error("tune-file must use the .toml or .jsonl extension");
        if (same_path(executable()) || same_path(ini) || same_path(worktodo) || same_path(finished) ||
            same_path(results) || same_path(legacy_progress) || (!log.empty() && same_path(log)) ||
            (!o.debug_file.empty() && same_path(fs::path(o.debug_file))) || (!o.tune_save.empty() && same_path(absolute_from(cwd,o.tune_save))) ||
            (!o.tune_worktodo.empty() && same_path(absolute_from(cwd,o.tune_worktodo))) ||
            (o.tune!="ecm" && !o.tune_profile.empty() && same_path(fs::path(o.tune_profile))) ||
            (!o.stage1_profile.empty() && same_path(fs::path(o.stage1_profile))) ||
            (!o.stage1_csv.empty() && same_path(fs::path(o.stage1_csv))))
            throw std::runtime_error("tune-file must differ from executable, config, queue and result files");
        for(const auto &input:o.tune_merge)if(same_path(absolute_from(cwd,input)))
            throw std::runtime_error("tune-file must differ from merge inputs");
        for(const auto &input:o.tune_ntt_profiles)if(same_path(absolute_from(cwd,input)))
            throw std::runtime_error("tune-file must differ from NTT inputs");
        if(o.tune=="ecm") {
            if(!toml)throw std::runtime_error("ECM tune requires a .toml profile");
            if(!o.tune_merge.empty()){merge_portable_tune(o,destination);return 0;}
            run_portable_tune(o,TaskDefaults{s.b2,s.arena,s.owner_mb},destination,worktodo,tmp);return 0;
        }
        if(toml){run_portable_ntt(o,destination);return 0;}
        if (!destination.parent_path().empty()) fs::create_directories(destination.parent_path());
        const fs::path partial(destination.string() + ".partial." + std::to_string(GetCurrentProcessId()));
        std::ofstream output(partial, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write tune-file: " + partial.string());
        const auto binary_hash = ecm_stage2::sha256_file(executable());
        const auto manifest = executable().parent_path() / "build_manifest.json";
        const auto manifest_hash = fs::is_regular_file(manifest) ? ecm_stage2::sha256_file(manifest) : "";
        if(toml)output << "# ECM Stage2 tuning data. Times in seconds; capacities in bytes.\n"
            << "[profile]\nformat = 2\nunit = \"field_convolution\"\nmin_log2 = " << o.tune_first
            << "\nmax_log2 = " << o.tune_last << "\nrepeats = " << o.tune_repeats << "\neffort_level = " << o.tune_level << '\n';
        else output << "{\"type\":\"profile\",\"schema\":2,\"unit\":\"field_convolution\""
               << ",\"min_log2\":" << o.tune_first << ",\"max_log2\":" << o.tune_last
               << ",\"repeats\":" << o.tune_repeats << ",\"effort_level\":" << o.tune_level << "}\n";
        const auto env=tune_environment();
        if(toml) {
            output<<"slices = [";for(size_t i=0;i<o.tune_slices.size();++i){if(i)output<<", ";output<<o.tune_slices[i];}output<<"]\n";
            output<<"\n[policy]\naccounting = \"cuda_events_two_forward_product_inverse_v1\"\n\n[policy.environment]\n";
            for(const auto &entry:env)output<<entry.first<<" = "<<entry.second<<'\n';
        } else {
            output<<"{\"type\":\"policy\",\"accounting\":\"cuda_events_two_forward_product_inverse_v1\",\"slices\":[";
            for(size_t i=0;i<o.tune_slices.size();++i){if(i)output<<',';output<<o.tune_slices[i];}
            output<<"],\"environment\":{";size_t i=0;for(const auto &entry:env){if(i++)output<<',';output<<json_string(entry.first)<<':'<<entry.second;}output<<"}}\n";
        }
        struct TuneOutput {std::ofstream &file;bool toml;};
        TuneOutput tune_output{output,toml};
        auto sink = [](const char *json, void *ctx) {
            auto &writer = *static_cast<TuneOutput *>(ctx);
            auto &out=writer.file;
            out << (writer.toml ? ecm_stage2::tune::table(json,true) : std::string(json)+"\n"); out.flush();
            if (!out) throw std::runtime_error("tune profile write failed");
            std::cout << json << std::endl;
        };
        const int code = ecm_cuda_stage2_tune_ntt_batches(device, o.tune_first, o.tune_last, o.tune_repeats,
            o.tune_memory_mb * 1048576,o.tune_slices.data(),o.tune_slices.size(),sink,&tune_output,o.tune_warmup);
        if (code) throw std::runtime_error("tune failed or measured no shapes; partial profile retained: " + partial.string());
        if (binary_hash != ecm_stage2::sha256_file(executable()) ||
            (!manifest_hash.empty() && manifest_hash != ecm_stage2::sha256_file(manifest)))
            throw std::runtime_error("tune binary/build manifest changed; partial profile retained");
        output.flush(); if (!output) throw std::runtime_error("tune profile flush failed");
        output.close(); if (!output) throw std::runtime_error("tune profile close failed");
        if (!MoveFileExW(partial.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot publish tune profile; partial profile retained: " + partial.string());
        std::cout << "tune_complete: profile=" << destination.string() << std::endl;
        return 0;
    }
    return run_tasks(std::move(o),{cwd,ini,worktodo,finished,tmp,results,log,legacy_progress,progress_dir},
                     {cfg.b2_ratio,cfg.worktodo_priority,cfg.worktodo_skip,cfg.finished},
                     {s.b2,s.arena,s.owner_mb});
}

} // namespace s2prod

int main(int argc, char **argv) {
    for (int i=1;i<argc;++i) if (std::string(argv[i])=="--gui-queue-protocol") {
        std::puts("ECM_QUEUE_OWNER=1"); return 0;
    }
    try { return s2prod::driver(s2prod::arguments(argc, argv)); }
    catch (const std::exception &e) {
        const std::string message=std::string("ecm_cuda_stage2: ")+e.what();
        std::cerr<<message<<std::endl;
        try {if(!s2prod::fatal_log.empty())s2prod::append(s2prod::fatal_log,message);}catch(...){}
        return 2;
    }
}
