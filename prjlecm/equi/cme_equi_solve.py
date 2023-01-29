import pprint

import numpy as np
import pandas as pd

import prjlecm.equa.cme_supt_equa_demand as cme_supt_equa_demand
import prjlecm.equa.cme_supt_equa_supply as cme_supt_equa_supply
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply
import prjlecm.util.cme_supt_math as cme_supt_math
import prjlecm.util.cme_supt_opti as cme_supt_opti


# Get Sets of Linear lines and solve
def cme_equi_solve_sone(dc_sprl_intr_slpe, dc_dmrl_intr_slpe, verbose=False):
    # Loop over individuals
    # We have two equations:
    # ln(y) = a + b ln(x)
    # ln(y) = c + d ln(x)
    # which becomes
    # a + b ln(x) = c + d ln(x)
    # which means
    # ln(x) = (a - c)/(d - b)
    # and we also have
    # ln(y) = a + b ln(x)
    mt_sprl_wthn_i_acrs_jv1_intr = dc_sprl_intr_slpe["mt_sprl_wthn_i_acrs_jv1_intr"]
    mt_sprl_wthn_i_acrs_jv1_slpe = dc_sprl_intr_slpe["mt_sprl_wthn_i_acrs_jv1_slpe"]

    mt_dmrl_wthn_i_acrs_jv1_intr = dc_dmrl_intr_slpe["mt_dmrl_wthn_i_acrs_jv1_intr"]
    mt_dmrl_wthn_i_acrs_jv1_slpe = dc_dmrl_intr_slpe["mt_dmrl_wthn_i_acrs_jv1_slpe"]

    it_wkr_cnt, it_occ_cnt_minus_1 = np.shape(mt_sprl_wthn_i_acrs_jv1_slpe)
    # mt_eqrl_wage_wthn_i_acrs_jv1 : a I x (J-1) matrix of optimal relative wages, within i across j, compare j>1 to j=1
    mt_eqrl_wage_wthn_i_acrs_jv1 = np.empty(
        [it_wkr_cnt, it_occ_cnt_minus_1], dtype=float)
    # mt_eqrl_qnty_wthn_i_acrs_jv1 : a I x (J-1) matrix of optimal relative wages, within i across j, compare j>1 to j=1
    mt_eqrl_qnty_wthn_i_acrs_jv1 = np.empty(
        [it_wkr_cnt, it_occ_cnt_minus_1], dtype=float)

    for it_wkr_ctr in np.arange(it_wkr_cnt):
        for it_occ_ctr in np.arange(it_occ_cnt_minus_1):
            a = mt_sprl_wthn_i_acrs_jv1_intr[it_wkr_ctr, it_occ_ctr]
            b = mt_sprl_wthn_i_acrs_jv1_slpe[it_wkr_ctr, it_occ_ctr]

            c = mt_dmrl_wthn_i_acrs_jv1_intr[it_wkr_ctr, it_occ_ctr]
            d = mt_dmrl_wthn_i_acrs_jv1_slpe[it_wkr_ctr, it_occ_ctr]

            fl_ln_of_x, fl_ln_of_y = cme_supt_math.cme_math_lin2n(
                a, b, c, d)
            # fl_ln_of_x = (a-c)/(d-b)
            # fl_ln_of_y = a + b * fl_ln_of_x

            mt_eqrl_wage_wthn_i_acrs_jv1[it_wkr_ctr, it_occ_ctr] = fl_ln_of_x
            mt_eqrl_qnty_wthn_i_acrs_jv1[it_wkr_ctr, it_occ_ctr] = fl_ln_of_y

    dc_equi_solve_sone = {"mt_eqrl_wage_wthn_i_acrs_jv1": mt_eqrl_wage_wthn_i_acrs_jv1,
                          "mt_eqrl_qnty_wthn_i_acrs_jv1": mt_eqrl_qnty_wthn_i_acrs_jv1}

    if verbose:
        for st_key, dc_val in dc_equi_solve_sone.items():
            print('d-30900 key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_equi_solve_sone


def cme_equi_solve_stwo(dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
                        dc_equi_solve_sone,
                        fl_spsh_j0_i1=0.1,
                        verbose=False):
    # Step two solution

    # Get matrixes from supply
    # two things below should be identical
    # mt_sprl_intr = dc_sprl_intr_slpe["mt_sprl_intr"]
    # ar_sprl_slpe = dc_sprl_intr_slpe["ar_sprl_slpe"]

    ar_sprl_wthn_i_acrs_j1v0_intr = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_intr"]
    ar_sprl_wthn_i_acrs_j1v0_slpe = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_slpe"]

    # Non-leisure category j=1 share
    # mt_eqrl_wage_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_wage_wthn_i_acrs_jv1"]
    mt_eqrl_qnty_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_qnty_wthn_i_acrs_jv1"]

    ar_sum_lab_summ_j1_unit = (
            np.sum(np.exp(mt_eqrl_qnty_wthn_i_acrs_jv1), 1) + 1)
    # totw = total work
    ar_spsh_j1_of_totw = 1 / ar_sum_lab_summ_j1_unit

    # Solve for L_{1,1}
    fl_splv_i1_j1, _ = cme_supt_equa_supply.cme_agg_splv_j1(
        ar_splv_totl_acrs_i[0], fl_spsh_j0_i1, ar_spsh_j1_of_totw[0])

    # Solve for W_{1,1}
    fl_wglv_i1_j1_p1 = (
                               (1 - fl_spsh_j0_i1) * (1 / fl_spsh_j0_i1)) * ar_spsh_j1_of_totw[0]
    # fl_wglv_i1_j1_p2 = np.log(fl_wglv_i1_j1_p1) - mt_sprl_intr[0, 0]
    fl_wglv_i1_j1_p2 = np.log(fl_wglv_i1_j1_p1) - \
                       ar_sprl_wthn_i_acrs_j1v0_intr[0]
    # fl_wglv_i1_j1 = np.exp(fl_wglv_i1_j1_p2*(1/ar_sprl_slpe[0]))
    fl_wglv_i1_j1 = np.exp(
        fl_wglv_i1_j1_p2 * (1 / ar_sprl_wthn_i_acrs_j1v0_slpe[0]))

    return {"fl_splv_i1_j1": fl_splv_i1_j1,
            "fl_wglv_i1_j1": fl_wglv_i1_j1,
            "ar_spsh_j1_of_totw": ar_spsh_j1_of_totw}


def cme_equi_solve_sthr(dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
                        dc_dmrl_intr_slpe,
                        dc_equi_solve_stwo,
                        fl_spsh_j0_i1=0.1,
                        verbose=False):
    # sthr = step three function

    # Supply inputs
    ar_sprl_wthn_i_acrs_j1v0_intr = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_intr"]
    ar_sprl_wthn_i_acrs_j1v0_slpe = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_slpe"]

    # Demand Inputs
    ar_dmrl_acrs_iv1_cnd_j1_intr = dc_dmrl_intr_slpe["ar_dmrl_acrs_iv1_cnd_j1_intr"]
    ar_dmrl_acrs_iv1_cnd_j1_slpe = dc_dmrl_intr_slpe["ar_dmrl_acrs_iv1_cnd_j1_slpe"]

    # Step 2 results inputs
    # splv_i1_j1 = L_{1,1}(nu_1)
    # wglv_i1_j1 = W_{1,1}(nu_1)
    # ar_spsh_j1_of_totw  = \hat{chi}^{\ast}_i
    fl_splv_i1_j1 = dc_equi_solve_stwo["fl_splv_i1_j1"]
    fl_wglv_i1_j1 = dc_equi_solve_stwo["fl_wglv_i1_j1"]
    ar_spsh_j1_of_totw = dc_equi_solve_stwo["ar_spsh_j1_of_totw"]

    it_wkr_cnt = np.shape(ar_sprl_wthn_i_acrs_j1v0_intr)[0]
    ar_it_wkr_i2tI_cnt = np.arange(it_wkr_cnt - 1) + 1
    # Obtain inputs for nonlinear solver to find shares
    # for supply structure: Loop over i from 2 to last, already solved for i = 1
    # demand matrix already I-1, excluding the first
    ar_lambda = np.empty([it_wkr_cnt - 1, ], dtype=float)
    ar_omega = np.empty([it_wkr_cnt - 1, ], dtype=float)
    ar_coef_ln1misnui = np.empty([it_wkr_cnt - 1, ], dtype=float)
    for it_wkr_ctr in ar_it_wkr_i2tI_cnt:
        # intr = alpha_{i,1}, slpe = beta_{i}
        fl_sprl_i_intr = ar_sprl_wthn_i_acrs_j1v0_intr[it_wkr_ctr]
        fl_sprl_i_slpe = ar_sprl_wthn_i_acrs_j1v0_slpe[it_wkr_ctr]

        # Invert demand-side intercept and slope
        fl_dmrl_i_intr = ar_dmrl_acrs_iv1_cnd_j1_intr[it_wkr_ctr - 1]
        fl_dmrl_i_slpe = ar_dmrl_acrs_iv1_cnd_j1_slpe[it_wkr_ctr - 1]
        # intr_inv = ln(theta_{i,1}/theta_{1,1})
        # slpe_inv = (psi-1)
        fl_dmrl_i_intr_inv, fl_dmrl_i_slpe_inv = cme_supt_math.cme_math_lininv(
            fl_dmrl_i_intr, fl_dmrl_i_slpe)

        # Lambda: lambda is a part of ln(W_{i,1})
        fl_lambda_p1 = np.log(fl_wglv_i1_j1) - fl_dmrl_i_slpe_inv * \
                       np.log(fl_splv_i1_j1) + fl_dmrl_i_intr_inv
        fl_lambda = fl_sprl_i_intr + fl_sprl_i_slpe * fl_lambda_p1

        # Omega: Omega has all the non nu_i i > 1 components
        fl_omega_p1 = ar_splv_totl_acrs_i[it_wkr_ctr] * \
                      ar_spsh_j1_of_totw[it_wkr_ctr]
        fl_omega_p2 = fl_sprl_i_slpe * fl_dmrl_i_slpe_inv * np.log(fl_omega_p1)
        fl_omega = fl_lambda + fl_omega_p2 - \
                   np.log(ar_spsh_j1_of_totw[it_wkr_ctr])

        # coefficient (1-beta*(psi-1)) before ln(1-nu_i)
        fl_coef_ln1misnui = 1 - fl_sprl_i_slpe * fl_dmrl_i_slpe_inv

        # store
        ar_lambda[it_wkr_ctr - 1] = fl_lambda
        ar_omega[it_wkr_ctr - 1] = fl_omega
        ar_coef_ln1misnui[it_wkr_ctr - 1] = fl_coef_ln1misnui

    # Apply non-linear solver to find nu_i(nu_1)
    ar_nu_solved = np.empty([it_wkr_cnt, ], dtype=float)
    ar_nu_solved[0] = fl_spsh_j0_i1
    for it_wkr_ctr in ar_it_wkr_i2tI_cnt:
        # store
        fl_omega = ar_omega[it_wkr_ctr - 1]
        fl_coef_ln1misnui = ar_coef_ln1misnui[it_wkr_ctr - 1]

        fl_nu_solved, fl_obj = cme_supt_opti.cme_opti_nu_bisect(
            fl_coef_ln1misnui, fl_omega)
        ar_nu_solved[it_wkr_ctr] = fl_nu_solved

    # Results matrix collection
    pd_nu_solu = pd.DataFrame(
        {"wkr_i": np.insert(ar_it_wkr_i2tI_cnt, 0, 0) + 1,
         "nu_i": ar_nu_solved,
         "coef_ln1misnui_i": np.insert(ar_coef_ln1misnui, 0, None),
         "omega_i": np.insert(ar_omega, 0, None),
         "lambda_i": np.insert(ar_lambda, 0, None)
         })

    # Print results
    if (verbose):
        print(pd_nu_solu.round(decimals=3))

    # return
    return {"ar_nu_solved": ar_nu_solved,
            "pd_nu_solu": pd_nu_solu}


def cme_equi_solve_sfur(dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
                        dc_dmrl_intr_slpe,
                        dc_equi_solve_sone, dc_equi_solve_stwo, dc_equi_solve_sthr,
                        verbose=False):
    # Supply levels, all workers, j=1
    ar_spsh_j1_of_totw = dc_equi_solve_stwo["ar_spsh_j1_of_totw"]
    ar_nu_solved = dc_equi_solve_sthr["ar_nu_solved"]
    ar_splv_j1, ar_splv_j0 = cme_supt_equa_supply.cme_agg_splv_j1(
        ar_splv_totl_acrs_i, ar_nu_solved, ar_spsh_j1_of_totw)

    # Wage levels, all workers, j=1
    mt_sprl_intr = dc_sprl_intr_slpe["mt_sprl_intr"]
    ar_sprl_slpe = dc_sprl_intr_slpe["ar_sprl_slpe"]
    ar_wglv_j1 = cme_supt_equa_supply.cme_hh_lgt_wglv_j1(
        ar_splv_j1, ar_splv_j0,
        mt_sprl_intr[:, 0], ar_sprl_slpe)

    # Print results
    pd_splv_j1 = pd.DataFrame(
        {"nu_i": ar_nu_solved,
         "ar_splv_totl_acrs_i": ar_splv_totl_acrs_i,
         "ar_spsh_j1_of_totw": ar_spsh_j1_of_totw,
         "ar_splv_j1": ar_splv_j1,
         "ar_splv_j0": ar_splv_j0,
         "ar_wglv_j1": ar_wglv_j1
         })

    # Print results
    if (verbose):
        print(pd_splv_j1.round(decimals=4))

    # Supply levels, all workers, all j
    mt_eqrl_qnty_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_qnty_wthn_i_acrs_jv1"]
    mt_qtlv_j2tJ = np.transpose(np.transpose(
        np.exp(mt_eqrl_qnty_wthn_i_acrs_jv1)) * ar_splv_j1)
    # mt_splv_all is I by J+1 matrix, all labor supply quantities, across all occupations
    mt_qtlv_all = np.column_stack([ar_splv_j0, ar_splv_j1, mt_qtlv_j2tJ])

    # Wage levels, all workers, all j
    mt_eqrl_wage_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_wage_wthn_i_acrs_jv1"]
    mt_wglv_j2tJ = np.transpose(np.transpose(
        np.exp(mt_eqrl_wage_wthn_i_acrs_jv1)) * ar_wglv_j1)
    # mt_splv_all is I by J matrix, all labor supply quantities, across all occupations
    mt_wglv_all = np.column_stack([ar_wglv_j1, mt_wglv_j2tJ])

    # Outputs as panda
    pd_qtlv_all = pd.DataFrame(
        mt_qtlv_all,
        index=["i" + str(i + 1)
               for i in np.arange(np.shape(mt_qtlv_all)[0])],
        columns=["j" + str(j) for j in np.arange(np.shape(mt_qtlv_all)[1])]
    )

    pd_wglv_all = pd.DataFrame(
        mt_wglv_all,
        index=["i" + str(i + 1)
               for i in np.arange(np.shape(mt_wglv_all)[0])],
        columns=["j" + str(j + 1)
                 for j in np.arange(np.shape(mt_wglv_all)[1])]
    )

    if (verbose):
        ar_splv_totw = np.sum(mt_qtlv_all, 1)
        pd_qtwg_all = pd.DataFrame(
            {"nu_i": ar_nu_solved,
             "ar_splv_totl_acrs_i": ar_splv_totl_acrs_i,
             "ar_spsh_j1_of_totw": ar_spsh_j1_of_totw,
             "ar_splv_totw": ar_splv_totw,
             "ar_splv_j1": ar_splv_j1,
             "ar_wglv_j1": ar_wglv_j1
             })
        print(pd_qtwg_all.round(decimals=3))
        print(pd_qtlv_all.round(decimals=3))
        print(pd_wglv_all.round(decimals=3))

    # Generate total output
    # Get parameters
    mt_dmrl_share = dc_dmrl_intr_slpe["mt_dmrl_share"]
    ar_dmrl_share_flat = np.ravel(mt_dmrl_share)
    # Get currently solved quantities
    ar_splv_all_flat = np.ravel(mt_qtlv_all[:, 1::])
    fl_elas = dc_dmrl_intr_slpe["fl_elas"]

    # this is general except for this last spot which does not work with nested problem
    fl_ces_output = cme_supt_equa_demand.cme_prod_ces(
        fl_elas, ar_dmrl_share_flat, ar_splv_all_flat)

    if verbose:
        print(f'{fl_ces_output=}')

    return {"fl_ces_output": fl_ces_output,
            "pd_qtlv_all": pd_qtlv_all,
            "pd_wglv_all": pd_wglv_all}


# steps 2 3 and 4 together


def cme_equi_solve_stwthfu(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_sone,
        fl_spsh_j0_i1=0.1, verbose=False):
    # D.2 Solve second step
    dc_equi_solve_stwo = cme_equi_solve_stwo(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_equi_solve_sone,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=verbose)

    # D.3 Solve third step
    dc_equi_solve_sthr = cme_equi_solve_sthr(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_stwo,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=verbose)

    # D.4 Generate all Levels
    dc_equi_solv_sfur = cme_equi_solve_sfur(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_sone, dc_equi_solve_stwo, dc_equi_solve_sthr,
        verbose=verbose)

    return dc_equi_solv_sfur


def cme_equi_solve(dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
                   dc_dmrl_intr_slpe,
                   dc_equi_solve_sone,
                   fl_output_target=0.3,
                   verbose_slve=False,
                   verbose=False):
    def cmei_opti_nu1_bisect_solver(fl_spsh_j0_i1):
        return cme_equi_solve_stwthfu(
            dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
            dc_dmrl_intr_slpe,
            dc_equi_solve_sone,
            fl_spsh_j0_i1=fl_spsh_j0_i1,
            verbose=verbose_slve)

    # minimal leisure time, max output
    fl_nu1_min = 1e-5
    fl_ces_output_max = cmei_opti_nu1_bisect_solver(
        fl_spsh_j0_i1=fl_nu1_min)["fl_ces_output"]
    if fl_output_target >= fl_ces_output_max:
        fl_nu1_solved = None
        fl_obj = None
        dc_equi_solv_sfur = None
    else:
        fl_nu1_solved, fl_obj, dc_equi_solv_sfur = cme_supt_opti.cme_opti_nu1_bisect(
            cmei_opti_nu1_bisect_solver,
            fl_output_target,
            fl_nu1_min=fl_nu1_min)

    if verbose:
        print(f'd-257707: {fl_nu1_solved=} and {fl_obj=}')
        if (dc_equi_solv_sfur is not None):
            for st_key, dc_val in dc_equi_solv_sfur.items():
                print('d-257707 key:' + str(st_key))
                print(dc_val)

    return fl_nu1_solved, dc_equi_solv_sfur


if __name__ == "__main__":
    import cme_equi_solve_gen_inputs

    # Parameters set up
    # Simulation parameter
    it_worker_types = 10
    it_occ_types = 40
    # it_seed_supply = 123
    # it_seed_demand = 456

    it_worker_types = 6
    it_occ_types = 4
    it_seed_supply = np.random.randint(1, 100)
    it_seed_demand = np.random.randint(1, 100)
    # it_seed_supply = 59
    # it_seed_demand = 65
    it_seed_supply = 71
    it_seed_demand = 71

    fl_itc_min = -2
    fl_itc_max = 1
    fl_slp_min = 0.4
    fl_slp_max = 0.6

    fl_power_min = 0.20
    fl_power_max = 0.20

    # Print control
    bl_verbose_simu = False
    bl_verbose_prep = False
    bl_verbose_slve = True

    # A. Simulate
    # A.1 Simulate supply parameters
    dc_supply_lgt, ar_splv_totl_acrs_i = cme_inpt_simu_supply.cme_simu_supply_params_lgt(
        it_worker_types=it_worker_types,
        it_occ_types=it_occ_types,
        fl_itc_min=fl_itc_min,
        fl_itc_max=fl_itc_max,
        fl_slp_min=fl_slp_min,
        fl_slp_max=fl_slp_max,
        it_seed=it_seed_supply,
        verbose=bl_verbose_simu)

    # A.2 Simulate demand parameters
    dc_demand_ces = cme_inpt_simu_demand.cme_simu_demand_params_ces_single(
        it_worker_types=it_worker_types,
        it_occ_types=it_occ_types,
        fl_power_min=fl_power_min,
        fl_power_max=fl_power_max,
        it_seed=it_seed_demand,
        verbose=bl_verbose_simu)

    # B. Supply and Demand input structures
    dc_sprl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest(
        dc_supply_lgt, verbose=bl_verbose_prep)
    dc_dmrl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_demand_dict_converter_nonest(
        dc_demand_ces, verbose=bl_verbose_prep)

    # C. Solve first step
    dc_equi_solve_sone = cme_equi_solve_sone(
        dc_sprl_intr_slpe, dc_dmrl_intr_slpe,
        verbose=bl_verbose_slve)

    # D.1 Solution parameter
    fl_spsh_j0_i1 = 0.1
    # D.2 Solve second step
    dc_equi_solve_stwo = cme_equi_solve_stwo(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_equi_solve_sone,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=bl_verbose_slve)

    # D.3 Solve third step
    dc_equi_solve_sthr = cme_equi_solve_sthr(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_stwo,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=bl_verbose_slve)

    # D.4 Generate all Levels
    cd_equi_solv_sfur = cme_equi_solve_sfur(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_sone, dc_equi_solve_stwo, dc_equi_solve_sthr,
        verbose=bl_verbose_slve)

    # solve for nu_1
    fl_output_target = 0.3
    fl_nu1_solved, dc_equi_solv_sfur = cme_equi_solve(
        dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_sone,
        fl_output_target=fl_output_target,
        verbose_slve=False,
        verbose=True)

    print(f'{it_seed_supply=}, {it_seed_demand=}')
