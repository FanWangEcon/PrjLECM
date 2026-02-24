# -*- coding: utf-8 -*-
"""
Testing solving for relative optimal equilibrium prices and quantities in labor market model
======================================================================

Part 3 of https://github.com/FanWangEcon/PrjLECM/issues/4

"""

import numpy as np
import prjlecm.equi.cme_equi_solve as cme_equi_solve
import prjlecm.util.cme_supt_misc as cme_supt_misc

np.set_printoptions(precision=8, suppress=True)
bl_call_solve_sone = False

# %%
# Solve for I=1 worker types, and J=4 occupations
# ======================================================================
# These correspond to what is written in Part 3 of https://github.com/FanWangEcon/PrjLECM/issues/4:
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
# Specifically, in the current step, we now use the relatively quantity and wage
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
#     'ar_nu_solved':
#         np.array([0.1]),
#     'pd_nu_solu':    wkr_i  nu_i  coef_ln1misnui_i  omega_i  lambda_i
#               0      1   0.1               NaN      NaN       NaN
# }

dc_equi_solve_stwo = {
    "fl_splv_i1_j1": 1.3987271786822328,
    "fl_wglv_i1_j1": 47.93646869485771,
    "ar_spsh_j1_of_totw": np.array([0.64755888]),
}

ar_splv_totl_acrs_i = np.array([2.40])
dc_sprl_intr_slpe = {
    "ar_sprl_wthn_i_acrs_j1v0_intr": np.array([-0.1247157]),
    "ar_sprl_wthn_i_acrs_j1v0_slpe": np.array([0.48771445]),
}
dc_dmrl_intr_slpe = {
    "ar_dmrl_acrs_iv1_cnd_j1_intr": np.array([]),
    "ar_dmrl_acrs_iv1_cnd_j1_slpe": np.array([]),
}

# D.1 Solution parameter
fl_spsh_j0_i1 = 0.1
# D.2 Solve second step
dc_equi_solve_sthr = cme_equi_solve.cme_equi_solve_sthr(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_stwo,
    fl_spsh_j0_i1,
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
#     'ar_nu_solved':
#         np.array([0.1       , 0.01720917, 0.07619084]),
#     'pd_nu_solu':    wkr_i      nu_i  coef_ln1misnui_i   omega_i  lambda_i
# 0      1  0.100000               NaN       NaN       NaN
# 1      2  0.017209          1.244278  4.040738  1.725777
# 2      3  0.076191          1.284408  2.472711  0.140389
# }

dc_equi_solve_stwo = {
    "fl_splv_i1_j1": 1.3987271786822328,
    "fl_wglv_i1_j1": 47.93646869485771,
    "ar_spsh_j1_of_totw": np.array([0.64755888, 0.14734316, 0.1537758]),
}

ar_splv_totl_acrs_i = np.array([2.40, 1.32, 1.29])
dc_sprl_intr_slpe = {
    "ar_sprl_wthn_i_acrs_j1v0_intr": np.array([-0.1247157, 0.03628279, -1.63347669]),
    "ar_sprl_wthn_i_acrs_j1v0_slpe": np.array([0.48771445, 0.41193558, 0.47960885]),
}

# Need the relative quantity output only for step two
dc_dmrl_intr_slpe = {
    "ar_dmrl_acrs_iv1_cnd_j1_intr": np.array([0.054789, -0.62444894]),
    "ar_dmrl_acrs_iv1_cnd_j1_slpe": np.array([-1.68634227, -1.68634227]),
}

# D.1 Solution parameter
fl_spsh_j0_i1 = 0.1
# D.2 Solve second step
dc_equi_solve_sthr = cme_equi_solve.cme_equi_solve_sthr(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_stwo,
    fl_spsh_j0_i1,
    verbose=False,
)

cme_supt_misc.print_dict_aligned(dc_equi_solve_sthr)

# %%
