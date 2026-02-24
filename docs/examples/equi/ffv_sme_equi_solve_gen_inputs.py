# -*- coding: utf-8 -*-
r"""
Equilibrium solutions, single-layer CES, multi-nomial labor supply.
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
import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.equi.cme_equi_solve as cme_equi_solve
import prjlecm.equi.cme_equi_solve_gen_inputs as cme_equi_solve_gen_inputs
import prjlecm.util.cme_supt_misc as cme_supt_misc
import numpy as np
import pandas as pd

# Create a pandas dataframe with the information below, exclude the wge and qty columns
# Enter data row by row, after first creating the dataframe's column names.
#    key_node  wkr  occ       itc  slp   wge   qty       qtp  lyr
# 0         1    0    0 -0.124716  0.5  None  None  4.913439    0
# 1         2    1    0 -3.412040  0.5  None  None  3.581734    0
# 2         3    0    1 -2.997025  0.5  None  None  4.913439    0
# 3         4    1    1 -1.140797  0.5  None  None  3.581734    0

# Define the column names, excluding 'wge' and 'qty'
columns = ["key_node", "wkr", "occ", "itc", "slp", "qtp", "lyr"]

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
# Create a pandas dataframe with the information below, exclude the the columns with only None values.
# Enter data row by row, after first creating the dataframe's column names.
#
#    key_node  lyr   prt   wkr   occ   nvi   nvl   nvb   nvr   nlb       shr  \
# 0         1    1     0     0     0  <NA>  None  None  None  None  0.395547
# 1         2    1     0     0     1  <NA>  None  None  None  None  0.162508
# 2         3    1     0     1     0  <NA>  None  None  None  None  0.128836
# 3         4    1     0     1     1  <NA>  None  None  None  None  0.313109
# 4         0    0  <NA>  <NA>  <NA>  <NA>  None  None  None  None       NaN

#         pwr           ipt   qty   wge   drv   drc   shc   sni
# 0       NaN          None  None  None  None  None  None  None
# 1       NaN          None  None  None  None  None  None  None
# 2       NaN          None  None  None  None  None  None  None
# 3       NaN          None  None  None  None  None  None  None
# 4  0.603628  [1, 2, 3, 4]  None  None  None  None  None  None

# Define the column names that are NOT always None across all rows
columns = ["key_node", "lyr", "prt", "wkr", "occ", "nvi", "shr", "pwr", "ipt"]

# Build the data rows, omitting columns where all values are None
# For missing values, use pd.NA, None, or float('nan') as appropriate

data = [
    [1, 1, 0, 0, 0, pd.NA, 0.395547, float("nan"), None],
    [2, 1, 0, 0, 1, pd.NA, 0.162508, float("nan"), None],
    [3, 1, 0, 1, 0, pd.NA, 0.128836, float("nan"), None],
    [4, 1, 0, 1, 1, pd.NA, 0.313109, float("nan"), None],
    [0, 0, pd.NA, pd.NA, pd.NA, pd.NA, float("nan"), 0.603628, [1, 2, 3, 4]],
]

df_demand_params = pd.DataFrame(data, columns=columns)
print(df_demand_params)
dc_demand_ces = cme_inpt_convert.cme_convert_pd2dc(
    df_demand_params, input_type="demand", verbose=True
)

# %%
# AIS ~ 2026-02-17 08:48:38
# Given df_supply_params:
# - Select unique `wkr` and` `qtp` combinations
# - Sort by `wkr` in ascending
# - Extract `qtp` values into a numpy array equal to ar_splv_totl_acrs_i
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