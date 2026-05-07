# -*- coding: utf-8 -*-
r"""
Equilibrium solutions, nested 2-layer CES, multinomial labor supply.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Part 2 of https://github.com/FanWangEcon/PrjLECM/issues/5

Here, we build on :doc:`ffv_sme_equi_solve_gen_inputs` results, but now have a nested-CES demand system. With two layers of nesting. The nested CES demand system is the same as the one used in :doc:`../demand/ffv_cme_dslv_opti_nested`, with the same parameters. The supply side here is identical to what is in :doc:`ffv_sme_equi_solve_gen_inputs`.

On this page, we:

1. Provide supply side equations and parameters.
2. Provide nested-demand side parameters in dataframe.
3. Convert these dataframes to dictionaries: `dc_supply_lgt` and `dc_demand_ces`, and get the array of total potential workers: `ar_splv_totl_acrs_i`. Note that in :doc:`ffv_sme_equi_solve_gen_inputs`, the next step is to create the dictionary of demand and supply arrays from, including `dc_sprl_intr_slpe` and `dc_dmrl_intr_slpe`. But here, for nested solution, we do not need those, they will be generated within the equilibrium solver with nested-demand.
4. Solve the equilibrium labor quantity and prices (wages) solutions, via :func:`prjlecm.equi.cme_equi_solve_nest.cme_equi_solve_nest`, given quantity output target `fl_output_target=1`
5. Given equilibrium wages we found, call the demand solver and call the supply solver, solve for optimal demand and supply given wages, to see if the equilibrium quanities match up to partial equilibrium results.

Suppose we have the following production function

Demand side
===========

Suppose we have the following production function:

.. math::

   \begin{align}
   \begin{split}
   Y(x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}) &= 
   \left(
      \left(
          \theta_{1,1} \cdot x_{1,1}^{\psi_1} + 
          \theta_{1,2} \cdot x_{1,2}^{\psi_1}
      \right)^{\frac{\psi}{\psi_1}} + 
      \left(
        \theta_{2,1} \cdot x_{2,1}^{\psi_2} + 
        \theta_{2,2} \cdot x_{2,2}^{\psi_2}
      \right)^{\frac{\psi}{\psi_2}}
   \right)^{\frac{1}{\psi}}\\
   &=
   \left(
      0.50 \cdot
      \left(
          0.30 \cdot x_{1,1}^{0.20} + 
          0.70 \cdot x_{1,2}^{0.20}
      \right)^{\frac{0.70}{0.20}} + 
      0.50 \cdot
      \left(
          0.10 \cdot x_{2,1}^{0.20} + 
          0.90 \cdot x_{2,2}^{0.20}
      \right)^{\frac{0.70}{0.20}}
   \right)^{\frac{1}{0.70}}\\
   \end{split}
   \end{align}


Note that the :math:`\theta_{i,j}` values sum up to one, and the :math:`{i,j}` subscript denote individual :math:`i` and occupation :math:`j`.

Supply side
===========

And on the supply side we have:

.. math::

   \begin{align}
   \begin{gathered}
   V_{i, 1,0} = 0 + u_{i, 1, 0}\\
   V_{i, 1,1} = \alpha_{1,1} + \beta \cdot \ln\left( W_{1,1} \right) + u_{i, 1,1}\\
   V_{i, 1,2} = \alpha_{1,2} + \beta \cdot \ln\left( W_{1,2} \right) + u_{i, 1,2}\\
   V_{i, 2,0} = 0 + u_{i, 2, 0}\\
   V_{i, 2,1} = \alpha_{2,1} + \beta \cdot \ln\left( W_{2,1} \right) + u_{i, 2,1}\\
   V_{i, 2,2} = \alpha_{2,2} + \beta \cdot \ln\left( W_{2,2} \right) + u_{i, 2,2}\\
   \end{gathered}
   \end{align}

With the parameter values:

.. math::

   \begin{align}
   \begin{gathered}
   V_{i, 1,1} = -0.12 + 0.5 \cdot \ln\left( W_{1,1} \right) + u_{i, 1,1}\\
   V_{i, 1,2} = -3.41 + 0.5 \cdot \ln\left( W_{1,2} \right) + u_{i, 1,2}\\
   V_{i, 2,1} = -3.00 + 0.5 \cdot \ln\left( W_{2,1} \right) + u_{i, 2,1}\\
   V_{i, 2,2} = -1.14 + 0.5 \cdot \ln\left( W_{2,2} \right) + u_{i, 2,2}\\
   \end{gathered}
   \end{align}

where :math:`V_{i, 1, 1}` is the indirect utility for an individual :math:`i`, that is of type :math:`i=1`, and if chooses occupation :math:`j=1`. We normalize the non-error component of :math:`j=0` (leisure) to :math:`0`. The :math:`u` are i.i.d. random Extreme Value Type I errors.

Supply side total potential worker counts
-----------------------------------------

Let the total population be:

.. math::

   \begin{align}
   \begin{split}
   L_1 = 4.91, L_2 = 3.58
   \end{split}
   \end{align}

Note that the probability ratio is the same as the quantity ratio because both are multiplied by the total potential worker pool, :math:`L_i` (the total potential workers).

"""

import prjlecm.demand.cme_dslv_opti as cme_dslv_opti
import prjlecm.supply.cme_splv_opti as cme_splv_opti

import prjlecm.input.cme_inpt_parse_wgqt as cme_inpt_parse_wgqt

import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.equi.cme_equi_solve_nest as cme_equi_solve_nest
import prjlecm.equi.cme_equi_solve_gen_inputs as cme_equi_solve_gen_inputs
import prjlecm.util.cme_supt_misc as cme_supt_misc
import numpy as np
import pandas as pd

# %%
# Implementation
# ================
# Our implementation involves three steps. First, we create a dataframe storing
# all CES production function information, along with wages. Second, we convert the
# dataframe to a demand dictionary. Third, we call the demand cost minimization solver to solve for optimal demand.
#
# Step 1, Supply side input data frame
# ------------------------------------
#
# Create a pandas dataframe with the information below, exclude the wge and qty columns
# Enter data row by row, after first creating the dataframe's column names::
#
#       key_node  wkr  occ       itc  slp   wge   qty       qtp  lyr
#    0         1    0    0 -0.124716  0.5  None  None  4.913439    0
#    1         2    1    0 -3.412040  0.5  None  None  3.581734    0
#    2         3    0    1 -2.997025  0.5  None  None  4.913439    0
#    3         4    1    1 -1.140797  0.5  None  None  3.581734    0

# Define the column names, excluding 'wge' and 'qty'
columns = ["key_node", "wkr", "occ", "itc", "slp", "qtp", "lyr"]
verbose = True

# Create the DataFrame row by row
data = [
    [1, 0, 0, -0.124716, 0.5, 4.913439, 0],
    [2, 1, 0, -3.412040, 0.5, 3.581734, 0],
    [3, 0, 1, -2.997025, 0.5, 4.913439, 0],
    [4, 1, 1, -1.140797, 0.5, 3.581734, 0],
]

df_supply_params = pd.DataFrame(data, columns=columns)
print(df_supply_params)

# Get the total number of potential workers for each worker type.
ar_splv_totl_acrs_i = (
    df_supply_params.drop_duplicates(subset=["wkr", "qtp"])
    .sort_values(by="wkr")["qtp"]
    .values
)

# %%
# Step 2, Demand side input data frame
# ------------------------------------
# Enter data row by row, after first creating the dataframe's column names.
# See :doc:`../demand/ffv_cme_dslv_opti_nested` for more details::

# Define the column names that are NOT always None across all rows
columns = ["key_node", "lyr", "prt", "wkr", "occ", "shr", "pwr", "ipt"]

# Build the data rows, omitting columns where all values are None
# For missing values, use pd.NA, None, or float('nan') as appropriate

data = [
    [1, 2, 5, 0, 0, 0.30, float("nan"), None],
    [2, 2, 5, 0, 1, 0.70, float("nan"), None],
    [3, 2, 6, 1, 0, 0.10, float("nan"), None],
    [4, 2, 6, 1, 1, 0.90, float("nan"), None],
    [5, 1, 7, pd.NA, pd.NA, 0.50, 0.20, [1, 2]],
    [6, 1, 7, pd.NA, pd.NA, 0.50, 0.20, [3, 4]],
    [7, 0, pd.NA, pd.NA, pd.NA, float("nan"), 0.70, [5, 6]],
]

df_demand_params = pd.DataFrame(data, columns=columns)
print(df_demand_params)

# %%
# Step 3, Generate supply and demand side dictionaries.
# -----------------------------------------------------
# Given demand and supply side input dataframes, we generated corresponding dictionaries.

dc_supply_lgt = cme_inpt_convert.cme_convert_pd2dc(
    df_supply_params, input_type="supply", verbose=True
)
dc_demand_ces = cme_inpt_convert.cme_convert_pd2dc(
    df_demand_params, input_type="demand", verbose=True
)

# %%
# Step 4, Solving the equilibrium problem
# ------------------------------------
# We call our equilibrium solution function, and solve for the equilibrium wages and quantities.

fl_output_target = 1

dc_ces_solu, dc_supply_lgt, dc_equi_solv_sfur, dc_equi_solve_nest_info = (
    cme_equi_solve_nest.cme_equi_solve_nest(
        dc_demand_ces,
        dc_supply_lgt,
        ar_splv_totl_acrs_i,
        fl_output_target,
        it_iter_max=1e2,
        fl_iter_tol=1e-5,
        fl_solu_tol=1e-3,
        verbose=False,
        verbose_debug=False,
    )
)

cme_supt_misc.print_dict_aligned(dc_ces_solu)
cme_supt_misc.print_dict_aligned(dc_equi_solve_nest_info)
cme_supt_misc.print_dict_aligned(dc_equi_solv_sfur)

# %%
# Step 5, Solve for demand and supply quantities given prices, check if quantities match equilibrium quantities
# =====================================================================================================
# Above, we have just solved for the equilibrium wages and quantities jointly,
# given demand and supply parameters.  Here we solve for the demand and supply
# problems separately, using the wages we just found, to see if he optimal
# quantities supplied and demand match up with the equilibrium quantities. This
# is a check to see if the equilibrium solution is consistent with the demand
# and supply solutions.


pd_wglv_all = dc_equi_solv_sfur["pd_wglv_all"]
pd_qtlv_all = dc_equi_solv_sfur["pd_qtlv_all"]
dc_demand_ces, dc_supply_lgt = cme_inpt_parse_wgqt.cme_parse_wgqt_pd2dc(
    fl_output_target, dc_demand_ces, dc_supply_lgt, pd_wglv_all, pd_qtlv_all
)

# 2. Given wages, solve optimal labor demands
dc_demand_ces = cme_dslv_opti.cme_prod_ces_nested_solver(
    dc_demand_ces, fl_Q_agg=None, verbose=False, verbose_debug=False
)
# Get optimal demand quantites
pd_qtlv_all_demand, __ = cme_inpt_parse_wgqt.cme_parse_qtwg_dc2pd_demand(
    dc_demand_ces, ar_splv_totl_acrs_i
)
# difference between optimal demand given prices and equilibrium quantities
pd_qtlv_equi_vs_demand = pd_qtlv_all - pd_qtlv_all_demand
fl_diff_equi_demand = np.sum(np.sum(np.abs(pd_qtlv_equi_vs_demand)))
if verbose:
    print(f"{pd_qtlv_all_demand=}")
    print(f"{pd_qtlv_equi_vs_demand=}")
    print(f"{fl_diff_equi_demand=}")

# 3. Check on supply decisions
dc_supply_lgt, pd_qtlv_all_supply = cme_splv_opti.cme_supply_lgt_solver(dc_supply_lgt)
# difference between optimal supply given prices and equilibrium quantities
pd_qtlv_equi_vs_supply = pd_qtlv_all - pd_qtlv_all_supply
fl_diff_equi_supply = np.sum(np.sum(np.abs(pd_qtlv_equi_vs_supply)))
if verbose:
    print(f"{pd_qtlv_all_supply=}")
    print(f"{pd_qtlv_equi_vs_supply=}")
    print(f"{fl_diff_equi_supply=}")

# %%
