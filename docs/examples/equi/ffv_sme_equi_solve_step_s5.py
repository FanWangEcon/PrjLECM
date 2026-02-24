# -*- coding: utf-8 -*-
"""
Testing solving for relative optimal equilibrium prices and quantities in labor market model
======================================================================

Part 5 of https://github.com/FanWangEcon/PrjLECM/issues/4

In contrasts to steps 1, 2, and 3. Solving with just 1 individual and solving with 3 individuals generate different results for equilibrium quantities and wages relevant for the 1 individual.

This step almost completes the full equilibrium solution, except that the share of individuals of type i=1 choosing leisure is still an exogenous input.

Additionally, in the
"""

import numpy as np
import prjlecm.equi.cme_equi_solve as cme_equi_solve
import prjlecm.util.cme_supt_misc as cme_supt_misc

np.set_printoptions(precision=2, suppress=True)
bl_nested_ces = False

# %%
# Solve for I=1 worker types, and J=4 occupations
# ======================================================================
# These correspond to what is written in Part 4 of https://github.com/FanWangEcon/PrjLECM/issues/4:
#
# In Step 1, we solve for relative equilibrium quantitles and wages within
# individual type i, of each occupation j vs occupation 1.
#
# In Step 2, we focus on individual type 1. We make a guess for share of
# individual of type 1 who choose leisure, and find the number of individual of
# type 1 participating in occupation type 1 given this, and then find the level
# of wage for occupation type 1 and individual type 1.
#
# In Step 3, given the level of individual type 1 and occupation j=1 worker and
# wage, we find the number of individuals of all types i > 1 working in
# occupation j=1. This uses within occupation j=1, relative demand optimality
# conditions across worker i>1 vs i=1.  The solution is based on solving I-1
# nonlinear equations of each of which is one equation with one unknown.
#
# In Step 4, given the number of individuals (and their wage) of all types i
# working in occupation j=1, since we also have the relative equilibrium
# quantities and wages from Step (1), we now have the equilibrium quantities and
# wages for all worker types i and all occupations j.
#
# Note that Steps 2, 3, and 4 are all dependent on the share of individuals of
# type i=1 choosing leisure. After step 4, all equilibrium quantity and wage
# levels are determined by this single unknown.
#
# In Step 5, we solve for the share of individuals of type i=1 choosing leisure,
# so that aggregate output matches the aggregate output target, this is an
# aggregate one equation with one unknown, and we solve through bisection. Step
# 1 is not dependent on the share of individuals of type i=1 choosing leisure,
# so we can run step 1 first, and use its outputs for step 5 here.
#
# Inputs for this is a subset of inputs for Step 4, because in step 4, we had to
# also provide as inputs the outputs from steps 2 and 3, but now these are generated
# inside step 5.
#
# We have one new input:
#
# - `fl_output_target`: the aggregate output target, which is the target for the
# bisection solver to solve for the share of individuals of type i=1 choosing
# leisure, this is the output of the aggregate production function.
#
# Expected output example when I=1:
# 0.5527037569999697
# 
# 0.6706902735089101
# 
# {
#     'fl_ces_output': 0.30000023955959393,
#     'pd_qtlv_all':           j0        j1      j2        j3        j4
# i1  1.326489  0.695162  0.0535  0.035514  0.289336,
#     'pd_wglv_all':           j1       j2        j3        j4
# i1  0.343307  0.64536  0.652366  0.457009
# }


fl_output_target = 0.3
dc_ces = {}

# From step 1: Relative equilibrium quantities and wages
dc_equi_solve_sone = {
    "mt_eqrl_wage_wthn_i_acrs_jv1": np.array([[0.63118508, 0.64198258, 0.28607947]]),
    "mt_eqrl_qnty_wthn_i_acrs_jv1": np.array([[-2.56447086, -2.97421994, -0.87655583]]),
}

# From step 2: Share of occupation j=1 among workers of each type i, this was an intermediary output in step 2, not the core output. `ar_spsh_j1_of_totw` could have been generated in step 1 and included as step 1 output as well.
dc_equi_solve_stwo = {
    "ar_spsh_j1_of_totw": np.array([0.64755888]),
}

# From step 3: the share of individuals of type i choosing leisure (i=1 is assumed, i>1 is solved)
dc_equi_solve_sthr = {"ar_nu_solved": np.array([0.1])}

#
ar_splv_totl_acrs_i = np.array([2.40])
dc_sprl_intr_slpe = {
    "mt_sprl_intr": np.array(
        [
            [-0.1247157, -2.99702466, -3.41203983, -1.14079662],
        ]
    ),
    "ar_sprl_slpe": np.array([0.48771445]),
    "ar_sprl_wthn_i_acrs_j1v0_intr": np.array([-0.1247157]),
    "ar_sprl_wthn_i_acrs_j1v0_slpe": np.array([0.48771445]),
}

# `mt_dmrl_share` was not used in prior steps, relative demand shares
# We have two alternative ways:
# 1. for single layer problem, if we know elasticity of substitution, and the "share" parameters, and we know levels of inputs, then we can find the output
# 2. for nested problems, this becomes trickier. And we have to solve through nested structure. And specify through `dc_ces` structure, and `mt_dmrl_share` and `fl_elas` are not used.
# Here, we call
dc_dmrl_intr_slpe = {
    "mt_dmrl_share": np.array(
        [
            [0.10691519, 0.04392533, 0.03482403, 0.08463249]
            / np.sum([0.10691519, 0.04392533, 0.03482403, 0.08463249]),
        ]
    ),
    "ar_dmrl_acrs_iv1_cnd_j1_intr": np.array([]),
    "ar_dmrl_acrs_iv1_cnd_j1_slpe": np.array([]),
    "fl_elas": 0.4070005712757371,
}

# D.2 Solve second step
fl_nu1_solved, dc_equi_solv_sfur, fl_ces_output_max = cme_equi_solve.cme_equi_solve(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_ces,
    bl_nested_ces=bl_nested_ces,
    fl_output_target=fl_output_target,
    verbose_slve=False,
    verbose=False,
)

cme_supt_misc.print_dict_aligned(fl_nu1_solved)
cme_supt_misc.print_dict_aligned(fl_ces_output_max)
cme_supt_misc.print_dict_aligned(dc_equi_solv_sfur)


# %%
# Solve for I=3 worker types, and J=4 occupations
# ======================================================================
# Results are independent across worker types, so we get the same result from the first
# row below as we did in the prior call with only first row (worker type 1) information
#
# Expected Output When I=3:
# 
# 0.3773685186576843
# 
# 0.3983034502972507
# 
# {
#     'fl_ces_output': 0.30000003439194534,
#     'pd_qtlv_all':           j0        j1        j2        j3        j4
# i1  0.905684  0.967657  0.074471  0.049435  0.402752
# i2  0.096813  0.180228  0.028537  0.868508  0.145914
# i3  0.398624  0.137072  0.078275  0.057039  0.618990,
#     'pd_wglv_all':           j1        j2        j3        j4
# i1  1.479084  2.780437  2.810622  1.968954
# i2  4.139238  7.261109  2.220667  4.465649
# i3  3.254564  3.699272  3.905976  2.017945
# }


fl_output_target = 0.3
dc_ces = {}

# From step 1: Relative equilibrium quantities and wages
dc_equi_solve_sone = {
    "mt_eqrl_wage_wthn_i_acrs_jv1": np.array(
        [
            [0.63118508, 0.64198258, 0.28607947],
            [0.56202092, -0.62270413, 0.07590281],
            [0.12807774, 0.18244942, -0.47797888],
        ]
    ),
    "mt_eqrl_qnty_wthn_i_acrs_jv1": np.array(
        [
            [-2.56447086, -2.97421994, -0.87655583],
            [-1.84302116, 1.57255261, -0.21120755],
            [-0.56027346, -0.87677284, 1.50758174],
        ]
    ),
}


# From step 2: Share of occupation j=1 among workers of each type i, this was an intermediary output in step 2, not the core output. `ar_spsh_j1_of_totw` could have been generated in step 1 and included as step 1 output as well.
dc_equi_solve_stwo = {
    "ar_spsh_j1_of_totw": np.array([0.64755888, 0.14734316, 0.1537758])
}

# From step 3: the share of individuals of type i choosing leisure (i=1 is assumed, i>1 is solved)
dc_equi_solve_sthr = {
    "ar_nu_solved": np.array([0.1, 0.01720917, 0.07619084]),
}

#
ar_splv_totl_acrs_i = np.array([2.40, 1.32, 1.29])
dc_sprl_intr_slpe = { 
    "mt_sprl_intr": np.array(
        [
            [-0.1247157, -2.99702466, -3.41203983, -1.14079662],
            [0.03628279, -2.03825478, 1.86534939, -0.20619183],
            [-1.63347669, -2.25517737, -2.59775389, 0.10334795],
        ]
    ),
    "ar_sprl_slpe": np.array([0.48771445, 0.41193558, 0.47960885]),
    "ar_sprl_wthn_i_acrs_j1v0_intr": np.array([-0.1247157, 0.03628279, -1.63347669]),
    "ar_sprl_wthn_i_acrs_j1v0_slpe": np.array([0.48771445, 0.41193558, 0.47960885]),
}

# `mt_dmrl_share` was not used in prior steps, relative demand shares
# We have two alternative ways:
# 1. for single layer problem, if we know elasticity of substitution, and the "share" parameters, and we know levels of inputs, then we can find the output
# 2. for nested problems, this becomes trickier. And we have to solve through nested structure. And specify through `dc_ces` structure, and `mt_dmrl_share` and `fl_elas` are not used.
# Here, we call
dc_dmrl_intr_slpe = {
    "mt_dmrl_share": np.array(
        [
            [0.10691519, 0.04392533, 0.03482403, 0.08463249],
            [0.11044589, 0.0649512, 0.1505574, 0.10512841],
            [0.073828, 0.06019407, 0.05268136, 0.11191663],
        ]
    ),
    "ar_dmrl_acrs_iv1_cnd_j1_intr": np.array([0.054789, -0.62444894]),
    "ar_dmrl_acrs_iv1_cnd_j1_slpe": np.array([-1.68634227, -1.68634227]),
    "fl_elas": 0.4070005712757371,
}

# D.2 Solve second step
fl_nu1_solved, dc_equi_solv_sfur, fl_ces_output_max = cme_equi_solve.cme_equi_solve(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_ces,
    bl_nested_ces=bl_nested_ces,
    fl_output_target=fl_output_target,
    verbose_slve=False,
    verbose=False,
)

cme_supt_misc.print_dict_aligned(fl_nu1_solved)
cme_supt_misc.print_dict_aligned(fl_ces_output_max)
cme_supt_misc.print_dict_aligned(dc_equi_solv_sfur)

# %%
