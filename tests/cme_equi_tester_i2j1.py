# Test the first step function
# So in the first step, use the relative optimality conditions across occupations
# within worker type.
# Test the solution for when there

import numpy as np

import prjlecm.demand.cme_dslv_opti as cme_dslv_opti
import prjlecm.equa.cme_supt_equa_supply as cme_supt_equa_supply
import prjlecm.equi.cme_equi_solve as cme_equi_solve
import prjlecm.equi.cme_equi_solve_gen_inputs as cme_equi_solve_gen_inputs
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply

# Note:
# Testing and solving with I=2 and J=1
shr_i1_j1 = 0.32433331
shr_i2_j1 = 1 - shr_i1_j1
elas_pwr = 0.2
dmrl_acrs_i2v1_cnd_j1_intr = np.log(shr_i2_j1 / shr_i1_j1) * (1 / (1 - elas_pwr))
dmrl_acrs_i1v1_cnd_j1_slpe = 1 / (1 - elas_pwr)
print(f'{dmrl_acrs_i2v1_cnd_j1_intr=} and {dmrl_acrs_i1v1_cnd_j1_slpe=}')

# Step 1 does not have results
# check step 2 results
# suppose fl_spsh_j0_i1 = some value, implment step 2 manually
# the first step here is purely from the supply side
li1_total = 4.62451085
li2_total = 4.04739815
alpha = -1
beta = 0.5
spsh_j0_i1 = 0.40  # This is the leisure share for type 1
spsh_j0_i2 = 0.52758257  # Solution from dc_equi_solve_sthr
# This would approximately be the correct solution
# wglv_i2_j1=34.11739532602192 and wglv_i2_j1_alt=34.762039387307794
spsh_j0_i2 = 0.31758257


def ffit_i2j1_wglv_given_spshj0(spsh_j0_i, li_total, verbose=True):
    """Internal function to go from share to wage supply-side
    """
    splv_i_j1 = li_total * (1 - spsh_j0_i)

    wglv_i_j1_p1 = np.exp(
        (np.log((1 - spsh_j0_i) * (1 / spsh_j0_i)) - alpha) / beta)
    wglv_i_j1_p1_alt = np.exp(
        ((np.log(splv_i_j1 / (li1_total - splv_i_j1)) - alpha) / beta))

    if np.abs(wglv_i_j1_p1 - wglv_i_j1_p1_alt) < 1e-10:
        pass
        if verbose:
            print(f'{splv_i_j1=} and {wglv_i_j1_p1=}')
    else:
        print(f'{wglv_i_j1_p1=} != {wglv_i_j1_p1_alt=}')

    return splv_i_j1, wglv_i_j1_p1


# test
splv_i1_j1, wglv_i1_j1 = ffit_i2j1_wglv_given_spshj0(spsh_j0_i1, li1_total)
splv_i2_j1, wglv_i2_j1 = ffit_i2j1_wglv_given_spshj0(spsh_j0_i2, li2_total)

# step 3 results check,
# Given the outputs from step three, are we satisfying the two equations for step 3?
wglv_i2_j1_alt = np.exp(np.log(shr_i2_j1 / shr_i1_j1) +
                        (elas_pwr - 1) * np.log(splv_i2_j1 / splv_i1_j1)) * wglv_i1_j1
print(f'{wglv_i2_j1=} and {wglv_i2_j1_alt=}')

# This tests all steps at the same time
# This provides a tester function, we have a single layer CES with some parameters
# Testing all steps of the problem at the same time.
# 1. Solve for the equilibrium prices and quantities
# 2. Given parameters and prices, solve for optimal demand decisions
# 3. Given parameters and prices, solve for optimal supply decisions
# 4. Check that (2) and (3) are equivalent


# STEP 0: Print controls
# Print control
bl_verbose_simu = False
bl_verbose_prep = False
bl_verbose_step = False
bl_verbose_slve = True

bl_verbose_rela = False

# STEP 1: Simulating parameters
# Parameters set up
# Simulation parameter

it_worker_types = 2
it_occ_types = 1

# it_seed_supply = 123
# it_seed_demand = 456
# it_seed_supply = np.random.randint(1, 100)
# it_seed_demand = np.random.randint(1, 100)
it_seed_supply = 71
it_seed_demand = 71

fl_itc_min = -2
fl_itc_max = 1
fl_slp_min = 0.4
fl_slp_max = 0.6

fl_itc_min = -1
fl_itc_max = -1
fl_slp_min = 0.5
fl_slp_max = 0.5

fl_power_min = -0.20
fl_power_max = -0.20

# 1.A Simulate
# Simulate supply parameters
dc_supply_lgt, ar_splv_totl_acrs_i = cme_inpt_simu_supply.cme_simu_supply_params_lgt(
    it_worker_types=it_worker_types,
    it_occ_types=it_occ_types,
    fl_itc_min=fl_itc_min,
    fl_itc_max=fl_itc_max,
    fl_slp_min=fl_slp_min,
    fl_slp_max=fl_slp_max,
    it_seed=it_seed_supply,
    verbose=bl_verbose_simu)

# 1.B Simulate demand parameters
dc_demand_ces = cme_inpt_simu_demand.cme_simu_demand_params_ces_single(
    it_worker_types=it_worker_types,
    it_occ_types=it_occ_types,
    fl_power_min=fl_power_min,
    fl_power_max=fl_power_max,
    it_seed=it_seed_demand,
    verbose=bl_verbose_simu)

# STEP 2, SOLVE for Equilibrium Wages and Quantities

# 2.A Supply and Demand input structures
dc_sprl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest(
    dc_supply_lgt, verbose=bl_verbose_prep)
dc_dmrl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_demand_dict_converter_nonest(
    dc_demand_ces, verbose=bl_verbose_prep)

# 2.B Solve first step
dc_equi_solve_sone = cme_equi_solve.cme_equi_solve_sone(
    dc_sprl_intr_slpe, dc_dmrl_intr_slpe,
    verbose=bl_verbose_slve)
mt_eqrl_wage_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_wage_wthn_i_acrs_jv1"]
mt_eqrl_qnty_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_qnty_wthn_i_acrs_jv1"]

# 2.C, Show steps two, three, four details for debugging.
fl_spsh_j0_i1 = spsh_j0_i1
# D.2 Solve second step
dc_equi_solve_stwo = cme_equi_solve.cme_equi_solve_stwo(
    dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
    dc_equi_solve_sone,
    fl_spsh_j0_i1=fl_spsh_j0_i1,
    verbose=bl_verbose_step)

# D.3 Solve third step
dc_equi_solve_sthr = cme_equi_solve.cme_equi_solve_sthr(
    dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_stwo,
    fl_spsh_j0_i1=fl_spsh_j0_i1,
    verbose=bl_verbose_step)

# D.4 Generate all Levels
dc_equi_solv_sfur = cme_equi_solve.cme_equi_solve_sfur(
    dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone, dc_equi_solve_stwo, dc_equi_solve_sthr,
    dc_demand_ces,
    verbose=bl_verbose_step)

# 2.D Solve for nu_1 (steps 2 + 3 + 4 nested)
fl_output_target = 0.5
fl_nu1_solved, dc_equi_solv_sfur = cme_equi_solve.cme_equi_solve(
    dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_demand_ces,
    fl_output_target=fl_output_target,
    verbose_slve=bl_verbose_step,
    verbose=bl_verbose_slve)

pd_qtlv_all = dc_equi_solv_sfur['pd_qtlv_all']
pd_wglv_all = dc_equi_solv_sfur['pd_wglv_all']

mt_qtlv_all = pd_qtlv_all.to_numpy()
mt_qtrl_all = (mt_qtlv_all.T / mt_qtlv_all[:, 1]).T
ar_qtlv_all_flat = np.ravel(mt_qtlv_all)
mt_wglv_all = pd_wglv_all.to_numpy()
mt_wgrl_all = (mt_wglv_all.T / mt_wglv_all[:, 0]).T
ar_wglv_all_flat = np.ravel(mt_wglv_all)

if bl_verbose_rela:
    print(f'{mt_qtrl_all=}')
    print(f'{mt_wgrl_all=}')

# STEP 3, Check equilibrium solutions with optimal demands
# Generate total output
# Get parameters
mt_dmrl_share = dc_dmrl_intr_slpe["mt_dmrl_share"]
ar_dmrl_share_flat = np.ravel(mt_dmrl_share)
fl_elas = dc_dmrl_intr_slpe["fl_elas"]

# this is general except for this last spot which does not work with nested problem
fl_Q = fl_output_target
fl_A = 1
ar_opti_costmin_x_flat, fl_mc_aggprice = cme_dslv_opti.cme_prod_ces_solver(
    ar_wglv_all_flat,
    ar_dmrl_share_flat,
    fl_elas,
    fl_Q, fl_A)
mt_opti_costmin_x = np.reshape(ar_opti_costmin_x_flat, np.shape(mt_wglv_all))
print(f'{mt_opti_costmin_x=}')

# check relative optimality within i across j
# note 0th col here is 1th col for mt_qtlv_all
mt_ocmx_rela = (mt_opti_costmin_x.T / mt_opti_costmin_x[:, 0]).T
if bl_verbose_rela:
    print(f'{mt_ocmx_rela=}')

# STEP 4, Check equilibrium solutions with optimal supply
mt_sprl_intr = dc_sprl_intr_slpe["mt_sprl_intr"]
ar_sprl_slpe = dc_sprl_intr_slpe["ar_sprl_slpe"]
mt_splv_all = np.empty([it_worker_types, it_occ_types + 1], dtype=float)
for it_wkr_ctr in np.arange(it_worker_types):
    ar_price_i = mt_wglv_all[it_wkr_ctr, :]
    ar_alpha_i = mt_sprl_intr[it_wkr_ctr, :]
    fl_beta_i = ar_sprl_slpe[it_wkr_ctr]
    fl_splv_totl_i = ar_splv_totl_acrs_i[it_wkr_ctr]

    ar_splv_i = cme_supt_equa_supply.cme_splv_lgt_solver(
        ar_price_i, ar_alpha_i,
        fl_beta_i, fl_splv_totl_i)

    mt_splv_all[it_wkr_ctr, :] = ar_splv_i

mt_splv_all / mt_qtlv_all
print(f'{mt_splv_all=}')

# check relative optimality within i across j
mt_sprl_rela = (mt_splv_all.T / mt_splv_all[:, 1]).T
if bl_verbose_rela:
    print(f'{mt_sprl_rela=}')
