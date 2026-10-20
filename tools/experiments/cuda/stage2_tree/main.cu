#include "engine.cuh"
int main(int argc, char **argv)
{
    s2g_install_crash_handler();          /* first: a crash must never print nothing */
    s2g_install_terminate();              /* and neither must an escaping exception */
    ladder_cap_init();                    /* the ladder launch length (NTT_LADDER_CAP) */
    {
        const char *enp = std::getenv("NTT_NO_PROGRESS");   /* progress lines on by default */
        if (enp && *enp && std::atoi(enp) != 0) g_s4_batched_progress = false;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char *check_F = nullptr, *gpu_dump = nullptr;
    const char *real_n = nullptr;
    bool selftest = false, evaluate = false, evaluate_batched = false;
    bool real = false, real_hex = false, choose_d = false, real_s2 = false, d_plan_only = false;
    unsigned long long r_sigma = 26, r_b1 = 1000, r_b2 = 1000000, r_d = 0;
    int r_curves = 1;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (!std::strcmp(a, "--check-F")) check_F = next();
        else if (!std::strcmp(a, "--dump-F-gpu")) gpu_dump = next();
        else if (!std::strcmp(a, "--device")) g_device = std::atoi(next());
        else if (!std::strcmp(a, "--evaluate")) evaluate = true;
        else if (!std::strcmp(a, "--evaluate-batched")) evaluate_batched = true;
        else if (!std::strcmp(a, "--selftest")) selftest = true;
        /* slice S4: the real-shape path (no CPU reference can exist at S = 5261) */
        else if (!std::strcmp(a, "--real")) real = true;
        else if (!std::strcmp(a, "--n")) real_n = next();
        else if (!std::strcmp(a, "--n-hex")) { real_n = next(); real_hex = true; }
        else if (!std::strcmp(a, "--sigma")) r_sigma = (unsigned long long)std::strtod(next(), nullptr);
        /* strtod, NOT strtoull(base 10): "--b2 1e11" is how the plan and Prime95's own
           parameters write B2, and strtoull stops at the 'e' and returns 1 (measured:
           a whole run computed a B2=1 shape and printed giant_points=2, section 27) */
        else if (!std::strcmp(a, "--b1")) r_b1 = (unsigned long long)std::strtod(next(), nullptr);
        else if (!std::strcmp(a, "--b2")) r_b2 = (unsigned long long)std::strtod(next(), nullptr);
        else if (!std::strcmp(a, "--d")) r_d = (unsigned long long)std::strtod(next(), nullptr);
        else if (!std::strcmp(a, "--choose-d")) choose_d = true;
        else if (!std::strcmp(a, "--d-plan-only")) d_plan_only = true;
        else if (!std::strcmp(a, "--s2")) real_s2 = true;
        else if (!std::strcmp(a, "--curves")) r_curves = std::atoi(next());
        else {
            std::fprintf(stderr,
                         "usage: stage2_tree_gpu --check-F <cpu dump> [--dump-F-gpu <file>] "
                         "[--evaluate] [--evaluate-batched] [--device N]\n"
                         "       stage2_tree_gpu --selftest [--device N]\n"
                         "       stage2_tree_gpu --real --n <decimal>|--n-hex <hex> "
                         "[--sigma S] [--b1 B1] [--b2 B2] [--d D | --choose-d] [--d-plan-only] [--s2] "
                         "[--curves K] [--device N]\n");
            return 2;
        }
    }
    if (selftest) return run_selftest();
    if (real) {
        if (!real_n) {
            std::fprintf(stderr, "%s: --real needs --n <decimal> or --n-hex <hex>\n",
                         NTT_PROBE_NAME);
            return 2;
        }
        if (!choose_d && r_d == 0) {
            std::fprintf(stderr, "%s: --real needs --d <D> or --choose-d\n", NTT_PROBE_NAME);
            return 2;
        }
        return run_real(real_n, real_hex, r_sigma, r_b1, r_b2, r_d, choose_d, real_s2, r_curves,nullptr,nullptr,d_plan_only);
    }
    if (!check_F) {
        std::fprintf(stderr, "%s: --check-F <file> is required (write one with "
                             "stage2_tree_ref --dump-F)\n", NTT_PROBE_NAME);
        return 2;
    }
    return run_check_F(check_F, gpu_dump, evaluate, evaluate_batched);
}
