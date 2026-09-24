#include <cstdlib>
#include <gmp.h>
#include "ecm.h"

extern "C" void ecm_init(ecm_params q)
{
    __ell_curve_struct *ptrE = (__ell_curve_struct *) malloc(sizeof(__ell_curve_struct));

    q->method = ECM_ECM; /* default method */
    mpz_init_set_ui (q->x, 0);
    mpz_init_set_ui (q->y, 0);
    mpz_init_set_ui (q->sigma, 0);
    q->sigma_is_A = 0;
    mpz_init_set_ui (ptrE->a1, 0);
    mpz_init_set_ui (ptrE->a3, 0);
    mpz_init_set_ui (ptrE->a2, 0);
    mpz_init_set_ui (ptrE->a4, 0);
    mpz_init_set_ui (ptrE->a6, 0);
    ptrE->type = ECM_EC_TYPE_MONTGOMERY;
    ptrE->disc = 0;
    mpz_init_set_ui (ptrE->sq[0], 1);
    q->E = ptrE;
    q->param = ECM_PARAM_DEFAULT;
    mpz_init_set_ui (q->go, 1);
    q->B1done = ECM_DEFAULT_B1_DONE;
    mpz_init_set_si (q->B2min, -1.0); /* default: B2min will be set to B1 */
    mpz_init_set_si (q->B2, ECM_DEFAULT_B2);
    q->k = ECM_DEFAULT_K;
    q->S = ECM_DEFAULT_S; /* automatic choice of polynomial */
    q->repr = ECM_MOD_DEFAULT; /* automatic choice of representation */
    q->nobase2step2 = 0; /* continue special base 2 code in ecm step 2, if used */
    q->verbose = 0; /* no output (default in library mode) */
    q->os = stdout; /* standard output */
    q->es = stderr; /* error output */
    q->chkfilename = NULL;
    q->TreeFilename = NULL;
    q->maxmem = 0.0;
    q->stage1time = 0.0;
    gmp_randinit_default (q->rng);
    mpz_set_ui (q->rng->_mp_seed, 0);
    q->use_ntt = 1;
    q->stop_asap = NULL;
    q->batch_last_B1_used = 1.0;
    mpz_init_set_ui (q->batch_s, 1);
    q->gpu = 0; /* no gpu by default in library mode */
    q->gpu_device = -1;
    q->gpu_device_init = 0;
    q->gpu_number_of_curves = 0;
    q->gpu_checkpoint_interval_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    q->gpu_param = 3;   /* legacy batch parametrization unless gpu_param = 0 */
    q->gpu_mul_path[0] = '\0';
    q->gpu_sqr_path[0] = '\0';
    q->gpu_add_path[0] = '\0';
    q->gpu_sub_path[0] = '\0';
    q->gpu_special_mult_path[0] = '\0';
    q->gw_k = 0.0;
    q->gw_b = 0;
    q->gw_n = 0;
    q->gw_c = 0;
    q->gw_cl_flag = -1;
}

extern "C" void ecm_clear(ecm_params q)
{
    mpz_clear (q->x);
    mpz_clear (q->y);
    mpz_clear (q->sigma);
    mpz_clear (q->go);
    mpz_clear (q->B2min);
    mpz_clear (q->B2);
    gmp_randclear (q->rng);
    mpz_clear (q->batch_s);
    mpz_clear (q->E->a1);
    mpz_clear (q->E->a3);
    mpz_clear (q->E->a2);
    mpz_clear (q->E->a4);
    mpz_clear (q->E->a6);
    mpz_clear (q->E->sq[0]);
    free (q->E);
}
