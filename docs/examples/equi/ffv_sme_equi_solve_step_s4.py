# -*- coding: utf-8 -*-
"""
Testing solving for relative optimal equilibrium prices and quantities in labor market model
======================================================================

Part 4 of https://github.com/FanWangEcon/PrjLECM/issues/4

This step almost completes the full equilibrium solution, except that the share
of individuals of type i=1 choosing leisure is still an exogenous input.
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
# the current step, we now use the relatively quantity and wage
# relations across individuals for the same occupation j=1. All ratios are
# against individual 1's wages and quantity, which we found in Step 2. So the
# unknown are no longer relative wages and quantites, but levels of wages and
# quantities in occupation j. We arrive at $I-1$ set of equilibrium conditions
# for $I-1$ equilibrium wage levels for $I-1$ individual types, all in
# occupation j=1.
#
# We provide as input the share of worker type 1 choosing leisure, and solve for
# the the share of workers of type 2, 3, ..., I choosing leisure. Each one of
# these $I-1$ result is solved via bisetion from
# :function:`prjlecm.equi.cme_supt_opti.cme_opti_nu1_bisect()`.  When we know
# the share of leisure for all individuals, given the relative equilibrium
# quantities we had solved, we have the equilibrium quantities of workers for
# all $I$ and all $J$. And we also have, via supply-side equations with relative
# log quantity of worker in occupation j>1 with respect to individuals in j=1,
# the wage levels for all $I$ and all $J$.
#
# Inputs here not used in step one or two, random values for these could be generated
# by `cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest()`:
#
# - Demand intercept, `ar_dmrl_acrs_iv1_cnd_j1_intr` =
# :math:`\frac{1}{1-\psi}\ln\left(\frac{\theta_{i,1}}{\theta_{1,1}}\right)`, 1
# by (I-1) array, across $I$, each versus $i=1$, conditional on $j=1$. When
# $I=1$, this is `np.array([])`.
# - Demand slope, `ar_dmrl_acrs_iv1_cnd_j1_slpe` = :math:`\frac{1}{1-\psi}`, 1
# by (I-1) array, can be I-1 specific, but in practice is same for all $i$. When
# $I=1$, this is `np.array([])`.
#
# In calling `cme_equi_solve.cme_equi_solve_stwo()`, we generate two two putputs:
#
# - `ar_nu_solved`: the equilibrium value of nu for each individual type i,
# :math:`\nu_{i}`, but when I=1, this is just our input value `fl_spsh_j0_i1`
# - `pd_nu_solu`: a dataframe of solution details for each individual type i
#
# Expected output example when I=1:
# {
#     'fl_ces_output': 0.6036272833447756,
#     'pd_qtlv_all':       j0        j1        j2        j3       j4
# i1  0.24  1.398727  0.107646  0.071457  0.58217,
#     'pd_wglv_all':            j1         j2         j3         j4
# i1  47.936469  90.112749  91.091013  63.812943
# }


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
    "mt_sprl_intr": np.array([
            [-0.1247157, -2.99702466, -3.41203983, -1.14079662],
        ]),
    "ar_sprl_slpe": np.array([0.48771445]),
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
    "fl_elas": 0.4070005712757371,
}

# D.2 Solve second step
dc_equi_solve_sthr = cme_equi_solve.cme_equi_solve_sfur(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_equi_solve_stwo,
    dc_equi_solve_sthr,
    dc_ces,
    bl_nested_ces=bl_nested_ces,
    verbose=False,
)


cme_supt_misc.print_dict_aligned(dc_equi_solve_sthr)

# %%
# Solve for I=3 worker types, and J=4 occupations
# ======================================================================
# Results are independent across worker types, so we get the same result from the first
# row below as we did in the prior call with only first row (worker type 1) information
#
# Expected Output When I=3:
# {
#     'fl_ces_output': 0.37380594390209043,
#     'pd_qtlv_all':           j0        j1        j2        j3        j4
# i1  0.240000  1.398727  0.107646  0.071457  0.582170
# i2  0.022716  0.191146  0.030266  0.921119  0.154753
# i3  0.098286  0.183257  0.104649  0.076257  0.827550,
#     'pd_wglv_all':             j1          j2          j3          j4
# i1   47.936469   90.112749   91.091013   63.812943
# i2  161.183366  282.750124   86.473543  173.893918
# i3  110.480255  125.576428  132.593259   68.501657
# }


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
    "fl_elas": 0.4070005712757371,
}

# D.2 Solve second step
dc_equi_solve_sthr = cme_equi_solve.cme_equi_solve_sfur(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_equi_solve_stwo,
    dc_equi_solve_sthr,
    dc_ces,
    bl_nested_ces=bl_nested_ces,
    verbose=False,
)

cme_supt_misc.print_dict_aligned(dc_equi_solve_sthr)

# %%
