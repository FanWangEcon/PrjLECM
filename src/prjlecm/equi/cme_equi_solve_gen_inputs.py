import pprint

import numpy as np

import prjlecm.equa.cme_supt_equa_demand as cme_supt_equa_demand
import prjlecm.equa.cme_supt_equa_supply as cme_supt_equa_supply
import prjlecm.input.cme_inpt_parse as cme_inpt_parse
import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply


# This file contains several support functions for CME equilibrium solvers


def cme_equi_supply_dict_converter_nonest(dc_supply_lgt, verbose=False):
    # count size
    dc_equi_sup_occ_wkr_lst = cme_inpt_parse.cme_parse_occ_wkr_lst(
        dc_supply_lgt, ls_it_idx=None, verbose=False
    )
    # note that for supply-side problem occ count does not consider leisure.
    it_occ_cnt = dc_equi_sup_occ_wkr_lst["it_occ_cnt"]
    it_wkr_cnt = dc_equi_sup_occ_wkr_lst["it_wkr_cnt"]
    ls_it_idx = dc_equi_sup_occ_wkr_lst["ls_it_idx"]

    # mt_sprl_wthn_i_acrs_jv1_intr : a I x (J-1) matrix of "relative-share" coefficients, Eq. 2 intercept (2nd line), compare j>1 to j=1
    mt_sprl_wthn_i_acrs_jv1_intr = np.empty([it_wkr_cnt, it_occ_cnt - 1], dtype=float)
    # mt_sprl_wthn_i_acrs_jv1_slpe : a I x (J-1) matrix of "elasticity" coefficients, Eq. 2 slopes (2nd line), compare j>1 to j=1
    mt_sprl_wthn_i_acrs_jv1_slpe = np.empty([it_wkr_cnt, it_occ_cnt - 1], dtype=float)

    # ar_sprl_wthn_i_acrs_j1v0_intr : a 1 x (I) array of "relative-share" coefficients, Eq. 6 intercept, compare j1 vs j0, across each i, wage level on RHS (j=0 does not have wage)
    ar_sprl_wthn_i_acrs_j1v0_intr = np.empty(
        [
            it_wkr_cnt,
        ],
        dtype=float,
    )
    # ar_sprl_wthn_i_acrs_j1v0_slpe : a I x (I) array of "elasticity" coefficients, Eq. 6 slopes, compre j1 vs j0, across each i, wage level on RHS (j=0 does not have wage)
    ar_sprl_wthn_i_acrs_j1v0_slpe = np.empty(
        [
            it_wkr_cnt,
        ],
        dtype=float,
    )

    # Convert share values to matrix
    mt_sprl_intr = np.empty([it_wkr_cnt, it_occ_cnt], dtype=float)
    # wages stored in the supply problem matrixes, might be all NONE
    # store so that demand and supply dcts are symmetric and can be grabbed out for solving
    # optimal supply problem given dc input.
    mt_splv_wge = np.empty([it_wkr_cnt, it_occ_cnt], dtype=float)
    ar_sprl_slpe = np.empty([it_wkr_cnt], dtype=float)
    # potential worker count, might be NONE
    ar_splv_qtp = np.empty([it_wkr_cnt], dtype=float)
    for it_idx in ls_it_idx:
        # Get worker and occupation index
        it_wkr = dc_supply_lgt[it_idx]["wkr"]
        it_occ = dc_supply_lgt[it_idx]["occ"]
        fl_itc = dc_supply_lgt[it_idx]["itc"]
        fl_slp = dc_supply_lgt[it_idx]["slp"]
        if ("wge" in dc_supply_lgt[it_idx]):
            fl_wge = dc_supply_lgt[it_idx]["wge"]
        else:
            fl_wge = float("nan")
        fl_qtp = dc_supply_lgt[it_idx]["qtp"]
        # Fill share matrix
        mt_sprl_intr[it_wkr, it_occ] = fl_itc
        mt_splv_wge[it_wkr, it_occ] = fl_wge
        # These must be worker type-specific, otherwise can not generate lin in log regression
        ar_sprl_slpe[it_wkr] = fl_slp
        ar_splv_qtp[it_wkr] = fl_qtp

    # Construct f
    for it_wkr_ctr in np.arange(it_wkr_cnt):
        fl_alpha_i_1 = mt_sprl_intr[it_wkr_ctr, 0]
        fl_beta_i = ar_sprl_slpe[it_wkr_ctr]

        # Fill in for Step 1
        for it_occ_ctr in np.arange(it_occ_cnt - 1):
            # Get psi and theta
            fl_alpha_i_j = mt_sprl_intr[it_wkr_ctr, it_occ_ctr + 1]

            # Compute
            fl_sprl_intr = cme_supt_equa_supply.cme_hh_lgt_sprl_intr(
                fl_alpha_i_j, fl_alpha_i_1
            )
            fl_sprl_slpe = cme_supt_equa_supply.cme_hh_lgt_sprl_slpe(fl_beta_i)

            # Fill in
            mt_sprl_wthn_i_acrs_jv1_intr[it_wkr_ctr, it_occ_ctr] = fl_sprl_intr
            mt_sprl_wthn_i_acrs_jv1_slpe[it_wkr_ctr, it_occ_ctr] = fl_sprl_slpe

        # Fill in for Step 3
        ar_sprl_wthn_i_acrs_j1v0_intr[it_wkr_ctr] = fl_alpha_i_1
        ar_sprl_wthn_i_acrs_j1v0_slpe[it_wkr_ctr] = fl_beta_i

    # See supply-side notations on https://github.com/FanWangEcon/PrjLECM/issues/4:
    # 
    # `I` is the number of individual types
    # `J` is the number of occupation types
    #
    # Below, the (1) of intercepts and slopes are data primitives, used togeher.
    # (2) of intercepts and slopes are used together. (3) of intercepts and
    # slopes are used together as well.
    #  
    # Supply intercepts, with (3) duplicating the first column of (1), and (2)
    # derived from differences of (1):
    # 
    # 1. Supply intercepts for i and j, `mt_sprl_intr`: :math:`\alpha_{i,j}`;
    # the I by J matrix of log(L_{i,j}/L_{i,0}) intercepts; this is the input
    # primitive based on which (2) and (3) below are derived; used in Step 4.
    # 
    # 2. Supply intercept difference within i across j,
    # `mt_sprl_wthn_i_acrs_jv1_intr`: :math:`\alpha_{i,j} - \alpha_{i,1}`; I by
    # J-1; derived form (1); used in Step 1.
    # 
    # 3. Supply intercept i and j=1, `ar_sprl_wthn_i_acrs_j1v0_intr`:
    # :math:`\alpha_{i,1}`, all $i$, j=1 vs j=0; I vector;
    # `ar_sprl_wthn_i_acrs_j1v0_intr` is the first column of `mt_sprl_intr`,
    # derived from (1); used in Step 2 and Step 3.
    #
    # Supply "slopes", with (2) and (3) simply duplicating (1): 
    # 
    # 1. Supply slope for i, `ar_sprl_slpe`: :math:`\beta_{i}`; the I vector of
    # log-linear wage coefficient with :math:`log(L_{i,j}/L_{i,0})` on the LHS;
    # Might be identical across individuals; This is the data primitive based on
    # which (2) and (3) are derived; used in Step 4.
    # 
    # 2. Supply slope for i and j>1, `mt_sprl_wthn_i_acrs_jv1_slpe`:
    # :math:`\beta_{i,j}`; this is the I by J-1 matrix of log-linear wage
    # coefficient with :math:`log(L(i,j)/L(i,1))` on the LHS, which is from
    # :math:`log(L_{i,j}/L_{i,0}) - log(L_{i,1}/L_{i,0})  `; This works because
    # :math:`\beta_{i}` is only i-specific; we are simply replicating
    # `ar_sprl_slpe`, treating it as a column and each of the J-1 columns are
    # identically duplicating `ar_sprl_slpe`; used in Step 1.
    # 
    # 3. Supply slope for i and j=1, `ar_sprl_wthn_i_acrs_j1v0_slpe`:
    # :math:`\beta`; This is dupliating `ar_sprl_slpe`, they are identical; used
    # in Step 2 and Step 3.
    # 

    dc_sprl_intr_slpe = {
        "mt_sprl_intr": mt_sprl_intr,
        "mt_splv_wge": mt_splv_wge,
        "ar_sprl_slpe": ar_sprl_slpe,
        "ar_splv_qtp": ar_splv_qtp,
        "mt_sprl_wthn_i_acrs_jv1_intr": mt_sprl_wthn_i_acrs_jv1_intr,
        "mt_sprl_wthn_i_acrs_jv1_slpe": mt_sprl_wthn_i_acrs_jv1_slpe,
        "ar_sprl_wthn_i_acrs_j1v0_intr": ar_sprl_wthn_i_acrs_j1v0_intr,
        "ar_sprl_wthn_i_acrs_j1v0_slpe": ar_sprl_wthn_i_acrs_j1v0_slpe,
    }

    if verbose:
        for st_key, dc_val in dc_sprl_intr_slpe.items():
            print("d-96215 key:" + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_sprl_intr_slpe


# Intercept and slope formula
def cme_equi_demand_dict_converter_nonest(dc_ces, verbose=False):
    """Demand input dictionary parser for non-nested CES"""
    ls_it_ipt = dc_ces[0]["ipt"]
    fl_elas = dc_ces[0]["pwr"]
    ls_res = cme_equi_demand_dict_converter_nest(
        dc_ces,
        st_rela_shr_key="shr",
        ls_it_ipt=ls_it_ipt,
        fl_elas=fl_elas,
        verbose=verbose,
    )

    return ls_res


def cme_equi_demand_dict_converter_nest(
    dc_ces, st_rela_shr_key="shc", ls_it_ipt=None, fl_elas=None, verbose=False
):
    """Generate inputs for equilibrium solution nested

    Work for nested or not nested

    Parameters
    ----------
    dc_ces : _type_
        _description_
    st_rela_shr_key : str, optional
        'shr' if single layer, 'shc' if nested 1st solution iteration, 'sni' if nested and >1
        solution iteration, by default 'shc'
    ls_it_ipt : _type_, optional
        _description_, by default None
    fl_elas : _type_, optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    # List of all keys to use
    if ls_it_ipt is None or fl_elas is None:
        it_max_layer, ls_maxlyr_key, it_lyr0_key = (
            cme_inpt_parse.cme_parse_demand_tbidx(dc_ces)
        )
        ls_it_ipt = ls_maxlyr_key
    if fl_elas is None:
        ar_pwr_across_layers = cme_inpt_parse.cme_parse_demand_lyrpwr(dc_ces)
        fl_elas = ar_pwr_across_layers[it_max_layer - 1]

    # create an empty matrixes with rows as indi as columns as occupations, to store the necessary information. Note that we consider RELATIVES!
    dc_equi_sup_occ_wkr_lst = cme_inpt_parse.cme_parse_occ_wkr_lst(
        dc_ces, ls_it_ipt, verbose=False
    )
    it_occ_cnt = dc_equi_sup_occ_wkr_lst["it_occ_cnt"]
    it_wkr_cnt = dc_equi_sup_occ_wkr_lst["it_wkr_cnt"]

    # mt_dmrl_wthn_i_acrs_jv1_intr : a I x (J-1) matrix of "relative-share" coefficients, Eq. 2 intercept (2nd line), compare j>1 to j=1
    mt_dmrl_wthn_i_acrs_jv1_intr = np.empty([it_wkr_cnt, it_occ_cnt - 1], dtype=float)
    # mt_dmrl_wthn_i_acrs_jv1_slpe : a I x (J-1) matrix of "elasticity" coefficients, Eq. 2 slopes (2nd line), compare j>1 to j=1
    mt_dmrl_wthn_i_acrs_jv1_slpe = np.empty([it_wkr_cnt, it_occ_cnt - 1], dtype=float)

    # ar_dmrl_acrs_iv1_cnd_j1_intr : a (I-1) x 1 array of "relative-share" coefficients, Eq. 5 intercept, compare I>1 to i=1, conditional on j=1
    ar_dmrl_acrs_iv1_cnd_j1_intr = np.empty(
        [
            it_wkr_cnt - 1,
        ],
        dtype=float,
    )
    # ar_dmrl_acrs_iv1_cnd_j1_slpe : a (I-1) x 1 array of "elasticity" coefficients, Eq. 5 slopes, compre i>1 to i=1
    ar_dmrl_acrs_iv1_cnd_j1_slpe = np.empty(
        [
            it_wkr_cnt - 1,
        ],
        dtype=float,
    )

    # Convert share values to matrix
    mt_dmrl_share = np.empty([it_wkr_cnt, it_occ_cnt], dtype=float)
    for it_ipt in ls_it_ipt:
        # Get worker and occupation index
        it_wkr = dc_ces[it_ipt]["wkr"]
        it_occ = dc_ces[it_ipt]["occ"]
        fl_demand_rela_share = dc_ces[it_ipt][st_rela_shr_key]
        # Fill share matrix
        mt_dmrl_share[it_wkr, it_occ] = fl_demand_rela_share

    if verbose:
        print(f"{np.sum(mt_dmrl_share)=}")

    # Fill in for step 1
    for it_wkr_ctr in np.arange(it_wkr_cnt):
        fl_theta_i_1 = mt_dmrl_share[it_wkr_ctr, 0]

        for it_occ_ctr in np.arange(it_occ_cnt - 1):
            # Get psi and theta
            fl_theta_i_j = mt_dmrl_share[it_wkr_ctr, it_occ_ctr + 1]

            # Compute
            fl_dmrl_intr = cme_supt_equa_demand.cme_prod_ces_dmrl_intr(
                fl_elas, fl_theta_i_j, fl_theta_i_1
            )
            fl_dmrl_slpe = cme_supt_equa_demand.cme_prod_ces_dmrl_slpe(fl_elas)

            # Fill in
            mt_dmrl_wthn_i_acrs_jv1_intr[it_wkr_ctr, it_occ_ctr] = fl_dmrl_intr
            mt_dmrl_wthn_i_acrs_jv1_slpe[it_wkr_ctr, it_occ_ctr] = fl_dmrl_slpe

    # Fill in for step three
    fl_theta_1_1 = mt_dmrl_share[0, 0]
    for it_wkr_ctr in np.arange(it_wkr_cnt - 1):
        fl_theta_i_1 = mt_dmrl_share[it_wkr_ctr + 1, 0]

        # Compute
        fl_dmrl_intr = cme_supt_equa_demand.cme_prod_ces_dmrl_intr(
            fl_elas, fl_theta_i_1, fl_theta_1_1
        )
        fl_dmrl_slpe = cme_supt_equa_demand.cme_prod_ces_dmrl_slpe(fl_elas)

        # Fill in
        ar_dmrl_acrs_iv1_cnd_j1_intr[it_wkr_ctr] = fl_dmrl_intr
        ar_dmrl_acrs_iv1_cnd_j1_slpe[it_wkr_ctr] = fl_dmrl_slpe

    # See demand-side notations on https://github.com/FanWangEcon/PrjLECM/issues/4:
    # 
    # `I` is the number of individual types
    # `J` is the number of occupation types
    # 
    # `mt_dmrl_share`: the I by J matrix of demand relative shares parameter
    # :math:`\theta_{i,j}`
    # 
    # `fl_elas`: the elasticity of substitution parameter :math:`\psi`
    # 
    # `mt_dmrl_wthn_i_acrs_jv1_intr` =
    # :math:`\frac{1}{1-\psi}\ln\left(\frac{\theta_{i,j}}{\theta_{i,1}}\right)`,
    # I by J-1 matrix, within $i$ and across $j$, each versus $j=1$
    # 
    # `mt_dmrl_wthn_i_acrs_jv1_slpe` = :math:`\frac{1}{1-\psi}`, I by J-1
    # matrix.
    # 
    # `ar_dmrl_acrs_iv1_cnd_j1_intr` =
    # :math:`\frac{1}{1-\psi}\ln\left(\frac{\theta_{i,1}}{\theta_{1,1}}\right)`,
    # (I-1) vector, across $I$, each versus $i=1$, conditional on $j=1$.
    # When $I=1$, this is `np.array([])`.
    # 
    # `ar_dmrl_acrs_iv1_cnd_j1_slpe` = :math:`\frac{1}{1-\psi}`, (I-1) vector. When
    # $I=1$, this is `np.array([])`.
    # 
    # `mt_dmrl_wthn_i_acrs_jv1_intr` and `mt_dmrl_acrs_iv1_cnd_j1_intr` are both
    # the intercept terms in log linear demand optimality condition where log
    # relative quantity is on the LHS and log relative price is on the RHS.

    dc_dmrl_intr_slpe = {
        "mt_dmrl_share": mt_dmrl_share,
        "fl_elas": fl_elas,
        "mt_dmrl_wthn_i_acrs_jv1_intr": mt_dmrl_wthn_i_acrs_jv1_intr,
        "mt_dmrl_wthn_i_acrs_jv1_slpe": mt_dmrl_wthn_i_acrs_jv1_slpe,
        "ar_dmrl_acrs_iv1_cnd_j1_intr": ar_dmrl_acrs_iv1_cnd_j1_intr,
        "ar_dmrl_acrs_iv1_cnd_j1_slpe": ar_dmrl_acrs_iv1_cnd_j1_slpe,
    }

    if verbose:
        for st_key, dc_val in dc_dmrl_intr_slpe.items():
            print("d-298608 key:" + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_dmrl_intr_slpe


if __name__ == "__main__":
    import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
    import prjlecm.demand.cme_dslv_eval as cme_dslv_eval

    bl_test_single_layer = False
    bl_test_nest_equi_init = False
    bl_test_nest_equi_post = True

    if bl_test_single_layer:
        # Supply testing
        dc_supply_lgt, _ = cme_inpt_simu_supply.cme_simu_supply_params_lgt(
            it_worker_types=3,
            it_occ_types=4,
            fl_itc_min=-5,
            fl_itc_max=2,
            fl_slp_min=0.4,
            fl_slp_max=0.6,
            it_seed=123,
            verbose=True,
        )
        dc_sprl_intr_slpe = cme_equi_supply_dict_converter_nonest(
            dc_supply_lgt, verbose=True
        )

        # Demand Testing
        dc_demand_ces = cme_inpt_simu_demand.cme_simu_demand_params_ces_single(
            it_worker_types=3,
            it_occ_types=4,
            fl_power_min=0.1,
            fl_power_max=0.8,
            it_seed=123,
            verbose=True,
        )
        dc_dmrl_intr_slpe = cme_equi_demand_dict_converter_nonest(
            dc_demand_ces, verbose=True
        )

    if bl_test_nest_equi_init:
        # B. Demand nested testing
        # B.1 first iteration, no q yet
        bl_simu_q = False
        st_rela_shr_key = "shc"
        dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
            ar_it_chd_tre=[3, 3],
            ar_it_occ_lyr=[2],
            fl_power_min=0.1,
            fl_power_max=0.8,
            it_seed=222,
            bl_simu_q=bl_simu_q,
            verbose=False,
            verbose_debug=False,
        )
        dc_ces_flat = dc_dc_ces_nested["dc_ces_flat"]
        dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_mpl(
            dc_ces_flat, verbose=False, verbose_debug=False
        )
        pprint.pprint(dc_ces_flat, width=10)
        dc_dmrl_intr_slpe_nested_1st = cme_equi_demand_dict_converter_nest(
            dc_ces_flat, st_rela_shr_key=st_rela_shr_key, verbose=True
        )

    if bl_test_nest_equi_post:
        # B. Demand nested testing
        # B.1 first iteration, no q yet
        bl_simu_q = True
        st_rela_shr_key = "sni"
        dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
            ar_it_chd_tre=[3, 3],
            ar_it_occ_lyr=[2],
            fl_power_min=0.1,
            fl_power_max=0.8,
            it_seed=222,
            bl_simu_q=bl_simu_q,
            verbose=False,
            verbose_debug=False,
        )
        dc_ces_flat = dc_dc_ces_nested["dc_ces_flat"]
        dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
            dc_ces_flat, verbose=False, verbose_debug=False
        )
        dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_mpl(
            dc_ces_flat, verbose=False, verbose_debug=False
        )
        pprint.pprint(dc_ces_flat, width=10)
        dc_dmrl_intr_slpe_nested_1st = cme_equi_demand_dict_converter_nest(
            dc_ces_flat, st_rela_shr_key=st_rela_shr_key, verbose=True
        )
