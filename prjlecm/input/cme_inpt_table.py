# Single tabular version of input structure
# To achieve the goal of working together with hand-provided input, create the following input structure:
# 
# Each node in tree is a different row
# Columns are:
# 1. share parameter, unique to each row
# 2. elasticity of substitution parameter, shared across rows within each nest
# 3. indicator of whether this layer is an occupation layer
# 4. columns for each layer as well, to indicate, map out tree structure
#
# 
# 1. Table with layers keys as columns, from top to bottom of CES nest from first to later columns
# 2. Store in a single column share parameter values
# 3. For higher level nest, where lower nests don't exist, leave unspecified columns as NA
# 4. Have an additional column specified ces-layer, is this for the 1st, 2nd, 3rd, 4th layer. 
# 
# how to construct a table in panda 

# There are two separate things, one is to convert from dc_ces_flat to the table format, 
# One is to translate from table format to dc_ces_flat, we should have both types of functions here
# but first, do the latter. 

import pandas as pd 

st_columns = ["lyr", "isocc", "c-flex", "c-gender-age", "shr", "pwr"]
pd_inputs = pd.DataFrame(columns=st_columns)

# row by row as dictionary
row_1 = {'lyr':1, 'isocc':True, "c-flex": "flexible", "shr":}

# import numpy as np
# import cme_inpt_simu_demand

# dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
#     ar_it_chd_tre=[2, 2, 3], ar_it_occ_lyr=[2],
#     fl_power_min=0.1,
#     fl_power_max=0.8,
#     it_seed=123,
#     bl_simu_q=True,
#     verbose=True, verbose_debug=True)

# ls_chd_uqe_idx_all = [np.arange(it_chd_tre)
#                         for it_chd_tre in ar_it_chd_tre]
# mt_occ_wkr_mesh = np.array(np.meshgrid(
#     *ls_chd_uqe_idx_all)).T.reshape(-1, len(ar_it_chd_tre))

# # which columns to select
# ar_it_col_sel_occ = np.array(ar_it_occ_lyr) - 1
# ar_it_col_sel_wkr = [idx for idx in np.arange(
#     len(ar_it_chd_tre)) if idx not in ar_it_col_sel_occ]

