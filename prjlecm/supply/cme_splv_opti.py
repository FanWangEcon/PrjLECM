# Optimal supply

import numpy as np
import pandas as pd

import prjlecm.equa.cme_supt_equa_supply as cme_supt_equa_supply
import prjlecm.equi.cme_equi_solve_gen_inputs as cme_equi_solve_gen_inputs
import prjlecm.input.cme_inpt_parse as cme_inpt_parse


def cme_supply_lgt_solver(dc_supply_lgt,
                          verbose=False, verbose_debug=False):
    """
    Solving for optimal supply problem, for any number of workers and any
    number of occuaptions. Assume multinomial-logit type structure.

    :param dc_supply_lgt: assume this dictionary contains prices via wge key
    for each occ and wkr combo, and also contains potential worker total count for
    each worker type
    :param verbose:
    :param verbose_debug:
    :return:
    """
    # 1. Get Parameters/inputs for solving optimal supply problems
    dc_parse_occ_wkr_lst = cme_inpt_parse.cme_parse_occ_wkr_lst(dc_supply_lgt)
    it_worker_types = dc_parse_occ_wkr_lst['it_wkr_cnt']
    it_occ_types = dc_parse_occ_wkr_lst['it_occ_cnt']

    dc_sprl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest(
        dc_supply_lgt, verbose=False)
    mt_sprl_intr = dc_sprl_intr_slpe["mt_sprl_intr"]
    mt_splv_wge = dc_sprl_intr_slpe["mt_splv_wge"]
    ar_sprl_slpe = dc_sprl_intr_slpe["ar_sprl_slpe"]
    ar_splv_qtp = dc_sprl_intr_slpe["ar_splv_qtp"]

    # 2. Solve for optimal supply worker by worker
    mt_splv_all = np.empty([it_worker_types, it_occ_types + 1], dtype=float)
    for it_wkr_ctr in np.arange(it_worker_types):
        ar_price_i = mt_splv_wge[it_wkr_ctr, :]
        ar_alpha_i = mt_sprl_intr[it_wkr_ctr, :]

        fl_beta_i = ar_sprl_slpe[it_wkr_ctr]
        fl_splv_totl_i = ar_splv_qtp[it_wkr_ctr]

        ar_splv_i = cme_supt_equa_supply.cme_splv_lgt_solver(
            ar_price_i, ar_alpha_i,
            fl_beta_i, fl_splv_totl_i)

        mt_splv_all[it_wkr_ctr, :] = ar_splv_i

    # 3. Update supply dictionary with quantities at each key
    for it_wkr_ctr in np.arange(it_worker_types):
        for it_occ_ctr in np.arange(it_occ_types):
            for it_key, dc_val in dc_supply_lgt.items():
                it_occ_key = dc_supply_lgt[it_key]["occ"]
                it_wkr_key = dc_supply_lgt[it_key]["wkr"]
                if (it_occ_key == it_occ_ctr) and (it_wkr_key == it_wkr_ctr):
                    dc_supply_lgt[it_key]['qty'] = mt_splv_all[it_wkr_ctr, it_occ_ctr]
    
    # 4. parse to table 
    pd_qtlv_all = pd.DataFrame(
        mt_splv_all,
        index=["i" + str(i + 1)
               for i in np.arange(np.shape(mt_splv_all)[0])],
        columns=["j" + str(j) for j in np.arange(np.shape(mt_splv_all)[1])]
    )

    return dc_supply_lgt, pd_qtlv_all
