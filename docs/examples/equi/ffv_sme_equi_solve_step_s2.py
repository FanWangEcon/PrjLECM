# -*- coding: utf-8 -*-
"""
Testing solving for relative optimal equilibrium prices and quantities in labor market model
============================================================================================

Part 2 of https://github.com/FanWangEcon/PrjLECM/issues/4

"""

import numpy as np
import prjlecm.equi.cme_equi_solve as cme_equi_solve
import prjlecm.util.cme_supt_misc as cme_supt_misc

np.set_printoptions(precision=8, suppress=True)
bl_call_solve_sone = False

# %%
# Solve for I=1 worker types, and J=4 occupations
# ======================================================================
# These correspond to what is written in Part 2 of https://github.com/FanWangEcon/PrjLECM/issues/4:
#
# In Step 1, we solved for equilibrium relative quantities and prices, all within-worker-type relative
# optimality of the jth occupation to the 1st occupation for the ith worker. But we are interested
# in equilibrium levels, not relative levels.
#
# In Step 2, First, we find the level of wage and labor quantity for occupation i=1 and worker type j=1.
# We do this by also providing `fl_spsh_j0_i1`, the share of type 1 individuals choosing
# leisure (occupation 0) as well as the total number of type 1 individuals,
# from the first element of `ar_splv_totl_acrs_i`. We multiply potential worker by the share
# not in leisure, and multiply this by share working in occupation 1, among those working.
# The share in occupation 1 among those working is equal to the inverse of the sum of
# the relative quantities of all occupations (including occupation 1) relative to
# occupation 1 (among worker of type 1).
#
# In the example here, there are 0, 1, 2, 3, 4 occupations:
#
# - `ar_splv_totl_acrs_i[0]` = 2.4, means there are 2.4 total type 1 individuals
# - `fl_spsh_j0_i1` = 0.1, means 10% of type 1 individuals choose leisure (occupation 0)
# - Therefore, 90% of type 1 individuals work, i.e., 2.4 * (1 - 0.1) = 2.16 individuals work
# - From Step 1, log relative quantities for occupation 2, 3, and 4 vs occupation 1 are: `mt_eqrl_qnty_wthn_i_acrs_jv1` = np.array([[-2.58, -2.98, -0.86]])
# - Exponentiating, np.exp(np.array([[-2.58, -2.98, -0.86]])) = np.array([[0.075774, 0.05079283, 0.42316208]])
# - The relative quantity of occupation 1 vs occupation 1 is 1, so total relative quantities across all occupations is: 1 + 0.075774 + 0.05079283 + 0.42316208 = 1.54972891, inverting this we have 1 / 1.54972891 = 0.645274146, which is the share of occupation 1 among those working, which is also stored in output `ar_spsh_j1_of_totw` below.
# - Therefore, the number of type 1 individuals working in occupation 1 is: 2.4 * (1 - 0.1) * 0.645274146 = 1.3937921, which is what we store in output `fl_splv_i1_j1` below.
#
# In Step 2, Second, we now use the equilibrium quantity of individuals of type 1
# choosing occupation 1 as well as the number of individuals choosing leisure,
# to arrive at the equilibrium level of occupation 1 wage for individual of type 1.
# We can do this because from demand side, as shown here, https://fanwangecon.github.io/R4Econ/regnonlin/logit/htmlpdfr/fs_logit_aggregate_shares.html.
# The log relative quantity of occupation 1 vs leisure is equal to the type 1 individual occupation 1 indirect utility intercept plus beta times log wage of occupation  for type 1 individual.
#
# In the example here:
#
# - The log of relative quantity of occupation 1 vs leisure is: np.log(1.3937921 / (2.4 * 0.1)) = 1.759145
# - From the first element of `ar_sprl_wthn_i_acrs_j1v0_intr`, we have the supply intercept for occupation 1 and type 1 individual is -0.1247157
# - From the first element of `ar_sprl_wthn_i_acrs_j1v0_slpe`, we have the supply slope for occupation 1 and type 1 individual is 0.48771445, (this is the coefficient in front of log wage)
# - Therefore, we can solve for the equilibrium wage for occupation 1 and type 1 individual as: (1.759145 + 0.1247157) / 0.48771445 = 3.86263, then exponentiating, np.exp(3.86263) = 47.5903, which is what we get from `fl_wglv_i1_j1` below.
#
# Now we document the inputs for the function. First type of inputs are inputs
# that are specified here and to be solved in later steps:
#
# - Share of worker of type 1 choosing leisure (occupation 0), `fl_spsh_j0_i1`: :math:`v_i`
#
# Second inputs types here not used in step one, random values for these could be generated
# by `cme_inpt_simu_supply.cme_simu_supply_params_lgt()`:
#
# - Total type i population, `ar_splv_totl_acrs_i`: :math:`\hat{L}_{i}`
#
# Third inputs types here not used in step one, random values for these could be generated
# by `cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest()`:
#
# - Supply intercept, `ar_sprl_wthn_i_acrs_j1v0_intr`: :math:`\alpha_{i,1}`, all
# $i$, j=1 vs j=0
# - Supply slope, `ar_sprl_wthn_i_acrs_j1v0_slpe`: :math:`\beta`
#
# Fourth input here is one of the outputs from step one:
#
# - Relative equilibrium quantities, `mt_eqrl_qnty_wthn_i_acrs_jv1`: :math:`\ln\left(\frac{L_{i,j}}{L_{i,1}}\right)`
#
# There is one worker type (I=1) and four occupations (J=4). There are
# 3 columns, because occupation 1 is the base occupation, and we are solving
# for relative prices (wages) and quantities for occupations 2, 3, and 4 relative to
# occupation 1.
#
# In calling `cme_equi_solve.cme_equi_solve_stwo()`, we generate equilibrium
# quantity and wage levels condition on unknown `v_1`:
#
# - `fl_splv_i1_j1`: equilibrium level for occupation 1 of worker type 1, :math:`L_{1,1}(v_1)`
# - `fl_wglv_i1_j1`: relative equilibrium quantities, :math:`\ln\left(\frac{L_{i,j}}{L_{i,1}}\right)`
# - `ar_spsh_j1_of_totw`: share of occupation 1 among all working individuals of type 1, :math:`\frac{L_{1,1}}{\sum_{j=1}^{J}L_{1,j}}`
#
# Expected output:
# {
#     'fl_splv_i1_j1': 1.3987271786822328,
#     'fl_wglv_i1_j1': 47.93646869485771,
#     'ar_spsh_j1_of_totw':
#         np.array([0.64755888])
# }

ar_splv_totl_acrs_i = np.array([2.40])
dc_sprl_intr_slpe = {
    "ar_sprl_wthn_i_acrs_j1v0_intr": np.array([-0.1247157]),
    "ar_sprl_wthn_i_acrs_j1v0_slpe": np.array([0.48771445]),
}

# Need the relative quantity output only for step two
dc_equi_solve_sone = {
    "mt_eqrl_qnty_wthn_i_acrs_jv1": np.array([[-2.56447086, -2.97421994, -0.87655583]])
}

# D.1 Solution parameter
fl_spsh_j0_i1 = 0.1
# D.2 Solve second step
dc_equi_solve_stwo = cme_equi_solve.cme_equi_solve_stwo(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_equi_solve_sone,
    fl_spsh_j0_i1=fl_spsh_j0_i1,
    verbose=False,
)

# Usage
cme_supt_misc.print_dict_aligned(dc_equi_solve_stwo)

# %%
# Solve for I=3 worker types, and J=4 occupations
# ======================================================================
#
# We now provide information across three worker types.
# However, the function `cme_equi_solve.cme_equi_solve_stwo()` is still solving
# The wage level and quantity level for occupation 1 and worker type 1 only.

# - `fl_splv_i1_j1` and `fl_wglv_i1_j1` are the same as calling with I=1.
# - `ar_spsh_j1_of_totw`: share of occupation 1 among all working individuals of
# type 1, 2, and 3, :math:`\frac{L_{i,1}}{\sum_{j=1}^{J}L_{i,j}}`
#
# Expected Output:
# {
#     'fl_splv_i1_j1': 1.3987271786822328,
#     'fl_wglv_i1_j1': 47.93646869485771,
#     'ar_spsh_j1_of_totw':
#         np.array([0.64755888, 0.14734316, 0.1537758 ])
# }

ar_splv_totl_acrs_i = np.array([2.40, 1.32, 1.29])
dc_sprl_intr_slpe = {
    "ar_sprl_wthn_i_acrs_j1v0_intr": np.array([-0.1247157, 0.03628279, -1.63347669]),
    "ar_sprl_wthn_i_acrs_j1v0_slpe": np.array([0.48771445, 0.41193558, 0.47960885]),
}

# Need the relative quantity output only for step two
dc_equi_solve_sone = {
    "mt_eqrl_qnty_wthn_i_acrs_jv1": np.array(
        [
            [-2.56447086, -2.97421994, -0.87655583],
            [-1.84302116, 1.57255261, -0.21120755],
            [-0.56027346, -0.87677284, 1.50758174],
        ]
    )
}

# D.1 Solution parameter
fl_spsh_j0_i1 = 0.1
# D.2 Solve second step
dc_equi_solve_stwo = cme_equi_solve.cme_equi_solve_stwo(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_equi_solve_sone,
    fl_spsh_j0_i1=fl_spsh_j0_i1,
    verbose=False,
)

# Usage
cme_supt_misc.print_dict_aligned(dc_equi_solve_stwo)

# %%
