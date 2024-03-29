import pprint

import numpy as np


def cme_simu_supply_lgt_dict(
        it_wkr=None, it_occ=None,
        fl_itc=None, fl_slp=None,
        fl_wge=None, fl_qty=None, fl_qtp=None):
    # if (it_wkr is not None):
    #     it_wkr = it_wkr + 100
    # if (it_occ is not None):
    #     it_occ = it_occ + 100

    dc_type_return = {
        'wkr': int, 'occ': int,
        'itc': float, 'slp': float,
        'wge': float, 'qty': float,
        'qtp': float,
        'lyr': int
    }
    dc_val_return = {
        # wkr type starts at 0
        'wkr': it_wkr,
        # 0 is not leisure, it's the first work category, leisure is not stored here
        # leisure can be found from: qtp - sum(qty) from the different occ conditional on wkr
        'occ': it_occ,
        # itc = intercept
        'itc': fl_itc,
        # slp = slope, the log wage effect term
        'slp': fl_slp,
        # Wage
        'wge': fl_wge,
        'qty': fl_qty,
        # potential number of workers
        'qtp': fl_qtp,
        # layer (only one layer, this is so that certain layer based function works
        'lyr': 0
    }

    return dc_val_return, dc_type_return


def cme_simu_supply_params_lgt(it_worker_types=2,
                               it_occ_types=2,
                               fl_itc_min=-5,
                               fl_itc_max=2,
                               fl_slp_min=0.5,
                               fl_slp_max=0.5,
                               fl_pop_min=0.5,
                               fl_pop_max=5,
                               it_seed=123,
                               bl_simu_params=True,
                               verbose=True):
    # it_worker_types = 2
    # it_occ_types = 2
    # fl_itc_min = -5
    # fl_itc_max = 2
    # fl_slp_min = 0.5
    # fl_slp_max = 0.5
    # it_seed = 123
    # verbose = True

    if bl_simu_params:
        np.random.seed(it_seed)
        mt_rand_coef_intcpt = np.random.rand(
            it_worker_types, it_occ_types
        ) * (fl_itc_max - fl_itc_min) + fl_itc_min

        # Slope coefficients must be homogeneous within worker-type, but can be different across workers
        ar_rand_coef_slope = np.random.rand(
            it_worker_types
        ) * (fl_slp_max - fl_slp_min) + fl_slp_min

        # ar_splv_totl_acrs_i: a 1 x (I) array of total potential labor available for each one of the I types of workesr
        ar_splv_totl_acrs_i = np.random.rand(
            it_worker_types
        ) * (fl_pop_max - fl_pop_min) + fl_pop_min

    else:
        mt_rand_coef_intcpt = np.reshape(
            [None] * (it_worker_types * it_occ_types),
            [it_worker_types, it_occ_types])
        ar_rand_coef_slope = np.array([None] * it_worker_types)
        ar_splv_totl_acrs_i = np.array([None] * it_worker_types)

    # Each input
    dc_lgt = {}
    # Start at 0 so that demand and supply keys are identical
    it_input_key_ctr = 0
    for it_occ_type_ctr in np.arange(it_occ_types):
        for it_worker_type_ctr in np.arange(it_worker_types):
            dc_cur_input, _ = cme_simu_supply_lgt_dict(
                it_wkr=it_worker_type_ctr, it_occ=it_occ_type_ctr,
                fl_itc=mt_rand_coef_intcpt[it_worker_type_ctr, it_occ_type_ctr],
                fl_slp=ar_rand_coef_slope[it_worker_type_ctr],
                fl_qtp=ar_splv_totl_acrs_i[it_worker_type_ctr])

            it_input_key_ctr = it_input_key_ctr + 1
            dc_lgt[it_input_key_ctr] = dc_cur_input

    # Print
    if (verbose):
        pprint.pprint(dc_lgt, width=2)

    return dc_lgt, ar_splv_totl_acrs_i


# Test
if __name__ == "__main__":
    # Simulate with randomly drawn parameters
    dc_lgt, ar_splv_totl_acrs_i = cme_simu_supply_params_lgt(
        it_worker_types=2,
        it_occ_types=2,
        fl_itc_min=-5,
        fl_itc_max=2,
        fl_slp_min=0.5,
        fl_slp_max=0.5,
        it_seed=123,
        bl_simu_params=True,
        verbose=True)

    # Generate skeleton without any values
    dc_lgt, ar_splv_totl_acrs_i = cme_simu_supply_params_lgt(
        it_worker_types=2,
        it_occ_types=2,
        bl_simu_params=False,
        verbose=True)
