# -*- coding: utf-8 -*-
r"""
Equilibrium solutions, single-layer CES, multinomial labor supply.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

https://github.com/FanWangEcon/PrjLECM/issues/5

On this page, we:

1. Provide demand and supply side equations and parameters.
2. Create dataframes `df_supply_params` and `df_demand_params` that store the parameters
3. Convert these dataframes to dictionaries: `dc_supply_lgt` and `dc_demand_ces`
4. Generate the dictionary of demand and supply arrays from (3): `dc_sprl_intr_slpe` and `dc_dmrl_intr_slpe`, as well as the array of total potential workers: `ar_splv_totl_acrs_i`
5. Solve the first step of the equilibrium solution, which is the single occupation equilibrium: `dc_equi_solve_sone`
6. Solve the equilibrium labor quantity and prices (wages) solutions, via `prjlecm.equi.cme_equi_solve.cme_equi_solve()`, given quantity output target `fl_output_target=1`

Suppose we have the following production function

Demand side
===========

Suppose we have the following production function:

.. math::

   \begin{align}
   \begin{split}
   Y &= 
   \left(
      \theta_{1,1} \cdot x_{1,1}^{\psi} + 
      \theta_{1,2} \cdot x_{1,2}^{\psi} + 
      \theta_{2,1} \cdot x_{2,1}^{\psi} + 
      \theta_{2,2} \cdot x_{2,2}^{\psi}
   \right)^{\frac{1}{\psi}}\\
   &=
   \left(
      0.40 \cdot x_{1,1}^{0.60} + 
      0.16 \cdot x_{1,2}^{0.60} + 
      0.13 \cdot x_{2,1}^{0.60} + 
      0.31 \cdot x_{2,2}^{0.60}
   \right)^{\frac{1}{0.60}}\\
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

Supply side derivations
-----------------------

Let :math:`J` be the occupation max indicator, :math:`j \in \{0, 1, \dots, J\}`. Let :math:`o` be the occupational choice variable. The probability that an individual of type :math:`i` chooses occupation :math:`o=j` is:

.. math::

   \begin{align}
   \begin{split}
   P(o=j) &= \frac{
       \exp\left(\alpha_{i,j} + \beta \cdot \ln\left( W_{i,j} \right)\right)
   }{
       1 + \sum_{\widehat{j}=1}^J\exp\left(
           \alpha_{i,\widehat{j}} + \beta \cdot \ln\left( W_{i,\widehat{j}} \right)
           \right)
   }
   \end{split}
   \end{align}

Note that :math:`\exp(0)=1` for the leisure term, and hence:

.. math::

   \begin{align}
   \begin{split}
   P(o=0) &= \frac{
      1
   }{
       1 + \sum_{\widehat{j}=1}^J\exp\left(
           \alpha_{i,\widehat{j}} + \beta \cdot \ln\left( W_{i,\widehat{j}} \right)
           \right)
   }
   \end{split}
   \end{align}

Subsequently, we have the "log odds-ratio", the log of the probability of choosing occupation :math:`o=j` relative to choosing occupation :math:`o=0`:

.. math::

   \begin{align}
   \begin{split}
   \ln\left(\frac{P(o=j)}{P(o=0)}\right) &= 
   \alpha_{i,j} + \beta \cdot \ln\left( W_{i,j} \right)
   \end{split}
   \end{align}

Which means furthermore:

.. math::

   \begin{align}
   \begin{split}
   \ln\left(\frac{P(o=j)}{P(o=1)}\right) &= 
   \ln\left(\frac{L_{i} \cdot P(o=j)}{L_{i} \cdot P(o=1)}\right) = 
   \ln\left(\frac{L_{i,j}}{L_{i,0}}\right) = 
   \left(\alpha_{i,j} - \alpha_{i,1}\right) + 
   \beta \cdot 
   \ln\left( 
       \frac{W_{i,j}}{W_{i,1}}
   \right)
   \end{split}
   \end{align}

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

import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand

import prjlecm.demand.cme_dslv_opti as cme_dslv_opti
import prjlecm.input.cme_inpt_parse as cme_inpt_parse
import prjlecm.supply.cme_splv_opti as cme_splv_opti

import prjlecm.input.cme_inpt_parse_wgqt as cme_inpt_parse_wgqt

import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.equi.cme_equi_solve as cme_equi_solve
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
dc_supply_lgt = cme_inpt_convert.cme_convert_pd2dc(
    df_supply_params, input_type="supply", verbose=True
)

# %%
# Step 2, Demand side input data frame
# ------------------------------------
# Create a pandas dataframe with the information below, exclude the the columns with only None values.
# Enter data row by row, after first creating the dataframe's column names.
# See :doc:`../demand/ffv_cme_dslv_opti_nonnested.py` for more details::
#
#       key_node  lyr   prt   wkr   occ   nvi   nvl   nvb   nvr   nlb       shr  \
#    0         1    1     0     0     0  <NA>  None  None  None  None  0.395547
#    1         2    1     0     0     1  <NA>  None  None  None  None  0.162508
#    2         3    1     0     1     0  <NA>  None  None  None  None  0.128836
#    3         4    1     0     1     1  <NA>  None  None  None  None  0.313109
#    4         5    0  <NA>  <NA>  <NA>  <NA>  None  None  None  None       NaN

#            pwr           ipt   qty   wge   drv   drc   shc   sni
#    0       NaN          None  None  None  None  None  None  None
#    1       NaN          None  None  None  None  None  None  None
#    2       NaN          None  None  None  None  None  None  None
#    3       NaN          None  None  None  None  None  None  None
#    4  0.603628  [1, 2, 3, 4]  None  None  None  None  None  None

# Define the column names that are NOT always None across all rows
columns = ["key_node", "lyr", "prt", "wkr", "occ", "nvi", "shr", "pwr", "ipt"]

# Build the data rows, omitting columns where all values are None
# For missing values, use pd.NA, None, or float('nan') as appropriate

data = [
    [1, 1, 5, 0, 0, pd.NA, 0.395547, float("nan"), None],
    [2, 1, 5, 0, 1, pd.NA, 0.162508, float("nan"), None],
    [3, 1, 5, 1, 0, pd.NA, 0.128836, float("nan"), None],
    [4, 1, 5, 1, 1, pd.NA, 0.313109, float("nan"), None],
    [5, 0, pd.NA, pd.NA, pd.NA, pd.NA, float("nan"), 0.603628, [1, 2, 3, 4]],
]

df_demand_params = pd.DataFrame(data, columns=columns)
print(df_demand_params)
dc_demand_ces = cme_inpt_convert.cme_convert_pd2dc(
    df_demand_params, input_type="demand", verbose=True
)

# %%
# Step 3, Generate supply and demand side dictionaries.
# ------------------------------------
# Given demand and supply side input dataframes, we generated corresponding dictionaries. 
# given these, we generate input dictionaries with all arrays and matrixes needed for equilibrium solution functions.
# 
ar_splv_totl_acrs_i = (
    df_supply_params.drop_duplicates(subset=["wkr", "qtp"])
    .sort_values(by="wkr")["qtp"]
    .values
)
dc_sprl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest(
    dc_supply_lgt, verbose=True
)
dc_dmrl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_demand_dict_converter_nonest(
    dc_demand_ces, verbose=True
)

# %%
# Step 4, Solving the equilibrium problem
# ------------------------------------
# We call our equilibrium solution function, and solve for the equilibrium wages and quantities. 
# See also :doc:`ffv_sme_equi_solve_step_s1`, :doc:`ffv_sme_equi_solve_step_s2`, 
# :doc:`ffv_sme_equi_solve_step_s3`, :doc:`ffv_sme_equi_solve_step_s4`, and
# :doc:`ffv_sme_equi_solve_step_s5` for more details on the solution steps, and
# there are five of them. The function below calls step 1, and then the other
# steps jointly.

dc_equi_solve_sone = cme_equi_solve.cme_equi_solve_sone(
    dc_sprl_intr_slpe, dc_dmrl_intr_slpe, verbose=True
)

# D.2 Solve second step
dc_ces = []
fl_output_target = 1
fl_nu1_solved, dc_equi_solv_sfur, fl_ces_output_max = cme_equi_solve.cme_equi_solve(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_ces,
    bl_nested_ces=False,
    fl_output_target=fl_output_target,
    verbose_slve=False,
    verbose=False,
)

cme_supt_misc.print_dict_aligned(fl_nu1_solved)
cme_supt_misc.print_dict_aligned(fl_ces_output_max)
cme_supt_misc.print_dict_aligned(dc_equi_solv_sfur)

# %%
# Solve for demand and supply quantities given prices, check if quantities match equilibrium quantities
# =====================================================================================================
# Above, we have just solved for the equilibrium wages and quantities jointly,
# given demand and supply parameters.  Here we solve for the demand and supply
# problems separately, using the wages we just found, to see if he optimal 
# quantities supplied and demand match up with the equilibrium quantities. This
# is a check to see if the equilibrium solution is consistent with the demand
# and supply solutions. 
# 
# This check is important because of equilibrium solution algorithm does not rely on the demand 
# and supply solvers below, so it is not by default that the results would match up. In 
# most equilibrium solution set-ups, one solves for quantity demanded end supplied given different
# levels of prices (wages), and then look for market clearing prices (wages). So the equilibrium
# solution builds on the demand and supply solutions given wages.
# 
# That is not the case in our
# solution structure, our equilibrium solution is a fully separate solution algorithm that does not
# call the demand and supply solver. We are not iterating over prices to solve for market clearing prices, 
# we have an semi-analytical equilibrium solution structure where the only unknown is the share of workers in 
# leisure for one of the types of workers. 


pd_wglv_all = dc_equi_solv_sfur["pd_wglv_all"]
pd_qtlv_all = dc_equi_solv_sfur["pd_qtlv_all"]
dc_demand_ces, dc_supply_lgt = cme_inpt_parse_wgqt.cme_parse_wgqt_pd2dc(
    fl_output_target,
    dc_demand_ces, dc_supply_lgt, pd_wglv_all, pd_qtlv_all
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