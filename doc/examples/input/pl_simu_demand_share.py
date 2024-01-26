# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Numeric Rounding Function
========================================================================
Given an array of numbers round it with conditioning formattings.
"""

# Author: Fan Wang (fanwangecon.github.io)
import prjlecm.input.cme_inpt_simu_demand_share as cisds
# %%
# Case one, baseline call
# --------------------------------------------------------------

verbose = True

dc_ar_poly_coef = cisds.cme_simu_ces_shrparam_poly_coef(verbose=True)
dc_mt_poly_x = cisds.cme_simu_ces_shrparam_poly_obs(verbose=True)
dc_sr_shares = cisds.cme_simu_ces_shrparam_param_gen(dc_mt_poly_x, dc_ar_poly_coef, True)

# %%
# Case two, some nest share poly-coef across prod, differ in poly-obs across prod
# 
# Two layer system, top layer a function of "year", bottom a function of "age"
# year varies across "production functions"
# age across share within nest.
# 
# so we need to generate 4 nests, bottom two shared across production functions, top 1 differs across prod func
# bottom layer is 3rd order polynomial or age, top layer is 3rd order polynomial of years
# coefficients are the same across produnction function, so poly seeds are identical for the 3rd and 4th nests below 
# which are the top layers for two different production functions. Note that share parameters also share the same 
# polynomial weights across polynomial coefficients.
# --------------------------------------------------------------

# Across production function, shared polynomial parameters
it_poly_order_top = 3
it_seed_top = 445
ar_fl_poly_scale_top = [1e0, 1e-1, 1e-2, 1e-3]
ar_bl_poly_pos_top = [True, True, True, True]

# Poly coef parameters
ar_it_poly_order = [3, 3, it_poly_order_top, it_poly_order_top]
ar_it_poly_seed = [123, 456, it_seed_top, it_seed_top]
dc_ar_fl_poly_scale = {0:[5e0, 1e-1, 1e-2, 1e-3], 1:[3e0, 1e0, 1e-2, 1e-2], 
                        2:ar_fl_poly_scale_top, 3:ar_fl_poly_scale_top}
dc_ar_bl_poly_pos = {0:[True, True, False, True], 1:[True, True, False, True], 
                        2:ar_bl_poly_pos_top, 3:ar_bl_poly_pos_top}
# Run poly coef function
dc_ar_poly_coef = cisds.cme_simu_ces_shrparam_poly_coef(
            ar_it_poly_order = ar_it_poly_order,
            ar_it_poly_seed = ar_it_poly_seed,
            dc_ar_fl_poly_scale = dc_ar_fl_poly_scale,
            dc_ar_bl_poly_pos = dc_ar_bl_poly_pos,
            verbose = False)

# Poly obs parameters
# note that the normalization here for the across prod
# parameter is done by changing obs for one share but not the other
# so the other is always the same in value, auto-normalizer.
dc_ar_fl_x_obs = {0:[25, 35, 45], 1:[25,35,45], 
                    2:[3, -10],
                    3:[10, -10]}
# Run poly obs function
dc_mt_poly_x = cisds.cme_simu_ces_shrparam_poly_obs(
    ar_it_poly_order = ar_it_poly_order,
    dc_ar_fl_x_obs = dc_ar_fl_x_obs, 
    verbose=True)

# Generate share parameters
dc_ar_shares = cisds.cme_simu_ces_shrparam_param_gen(
    dc_mt_poly_x, dc_ar_poly_coef, True)

# Normalize 
dc_ar_shares_normalized, dc_fl_normalize = cisds.cme_simu_ces_shrparam_normalize(
    dc_ar_shares, fl_share_sum = 1, verbose =True)


# %%
# Case three, some nest differ poly-coef across prod, same poly-obs across prod
# --------------------------------------------------------------
pass

# %%
# Case four, some nest same poly-coef across prod, same poly-obs across prod
# --------------------------------------------------------------
pass