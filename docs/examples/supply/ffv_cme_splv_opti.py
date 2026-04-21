# -*- coding: utf-8 -*-
r"""
Solving for labor supply given wages for multinomial logistic occupational choice model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

https://github.com/FanWangEcon/PrjLECM/issues/8

On this page, we provide supply side equations for a multi-nomial logistic 
occupational choice model, solve for the quantity of labor supplied, 
across different worker types and occupations, given wages and total worker pool
for each type of workers.


Occupation-specific indirect utility
=================================================================

To illustrate, we have the following occupation-specific indirect utility functions for two types of workers (i=1,2) and two non-leisure occupations (j=1,2), with leisure as the outside option (j=0):

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

Consider the following parameter values:

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

Probability of choosing each occupation
=================================================================

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

Number of workers in each occupation
=================================================================

Let the total population be:

.. math::

   \begin{align}
   \begin{split}
   L_1 = 4.91, L_2 = 3.58
   \end{split}
   \end{align}

Then the number of workers of type :math:`i` choosing occupation :math:`j` is:

.. math::
    \begin{align}
    \begin{split}
    N_{i,j} = L_i \cdot P(o=j)
    \end{split}
    \end{align}
    

Storing parameters, prices, and quantities in a supply system data structure
============================================================================
We store information from the supply system, in the following structure. 
Consider each occupation for a type of worker as a node. 
We have, associated with each node, the indirect utility :math:`V_{i,j}`, which 
is defined by the "intercept" :math:`\alpha_{i,j}` coefficient and the "slope" :math:`\beta` coefficient on log wages.
If we have worker- and occupation-specific observables that enter the indirect utility, the dot product of the vector of coefficients and vector of observable are reflected
in the "intercept" term. 
There could also be worker-specific observables that enter into all indirect utility functions with potentially different coefficient vectors, in that case the dot product of the coefficients and observables still map to to the "intercept" term. 
From the perspective of solving the supply-side model, it is only the occupation- and worker-specific intercept and the slope on log wages that matter, and we store these two pieces of information for each node.
Hence, we have, for each worker and occupation combination:

1. its node ID.
2. its "intercept"
3. its "slope"

Additionally, we store, for this worker and ocupation combination:

1. an occupation id 
2. a worker type id
3. the wage paid for this type of worker in this occupation (to be solved for in equilibrium setting)
4. the quantity demanded of this input (to be solved for in equilibrium but given wages)
5. potential total worker count for this type of worker

There are two ways of thinking about wages and quantity from a solution point of view:

- The wage and quantity values could be solved in equilibrium given parameters. This is what we do in file :doc:`../equi/ffv_sme_equi_solve_gen_inputs.py`, where we solve for equilibrium wages and quantities given parameters on the demand and supply side.
- Given wages, we can solve for quantities, this is what we do here on this page.

"""

import pandas as pd
import numpy as np
import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.supply.cme_splv_opti as cme_splv_opti
import prjlecm.input.cme_inpt_parse_wgqt as cme_inpt_parse_wgqt


# %%
# Implementation
# ================
# Our implementation involves three steps. First, we create a dataframe storing
# all supply side information, along with wages. Second, we convert the
# dataframe to a supply dictionary. Third, we call the supply optimization solver to solve for optimal supply.
#
# Step 1, Supply side input data frame
# ------------------------------------
#
# We will create a pandas dataframe with the information below. The columns ``wge`` and
# ``qty`` are included but left as ``None`` since they are to be solved for. Enter the
# rows after first creating the dataframe's column names::
#
#       key_node  wkr  occ       itc  slp   wge   qty       qtp  lyr
#    0         1    0    0 -0.124716  0.5  None  None  4.913439    0
#    1         2    1    0 -3.412040  0.5  None  None  3.581734    0
#    2         3    0    1 -2.997025  0.5  None  None  4.913439    0
#    3         4    1    1 -1.140797  0.5  None  None  3.581734    0
#
# Build rows as dicts so each key is explicitly paired with its value, column key meanings:
#
#  * **key_node** : unique integer ID for this supply node
#  * **wkr**      : worker-type index identifying which worker group this node belongs to
#  * **occ**      : occupation index identifying which occupation slot this node fills (0-indexed over non-leisure occupations)
#  * **itc**      : intercept alpha_{i,j} in the occupation-specific indirect utility V_{i,j} = alpha_{i,j} + beta * ln(W_{i,j})
#  * **slp**      : slope beta on log wages in the indirect utility function, shared across all nodes
#  * **wge**      : wage w_{i,j} paid to this worker type in this occupation (None for now; given or solved in equilibrium)
#  * **qty**      : quantity supplied, i.e. number of workers of type i choosing occupation j (None for now; to be solved)
#  * **qtp**      : potential total worker count L_i for this worker type, the pool from which occupational choices are drawn
#  * **lyr**      : layer index (0 for all supply nodes; the supply system is flat, not a tree)
#
#  To create the table, we generate data row by row below as a list of dictionaries, and then convert the list of dicts to a dataframe. We also print the dataframe to check that it looks correct.
#
verbose = True

data = [
    # Node: worker type 0 (wkr=0), occupation 0 (occ=0)
    # Intercept alpha_{1,1}=-0.124716; slope beta=0.5; total worker pool L_1=4.913439
    {
        "key_node": 1,  # unique node ID for this supply node
        "wkr": 0,  # worker type index 0 (type 1 in 1-indexed notation)
        "occ": 0,  # occupation index 0 (first non-leisure occupation)
        "itc": -0.124716,  # alpha_{1,1}: intercept in indirect utility V_{1,1}
        "slp": 0.5,  # beta: slope on log wages, shared across all nodes
        "wge": 0.9185,  # wage w_{1,1}
        "qty": None,  # quantity supplied N_{1,1}: to be solved for
        "qtp": 4.913439,  # L_1: total worker pool for worker type 0
        "lyr": 0,  # layer index (flat supply system)
    },
    # Node: worker type 1 (wkr=1), occupation 0 (occ=0)
    # Intercept alpha_{2,1}=-3.412040; slope beta=0.5; total worker pool L_2=3.581734
    {
        "key_node": 2,  # unique node ID for this supply node
        "wkr": 1,  # worker type index 1 (type 2 in 1-indexed notation)
        "occ": 0,  # occupation index 0 (first non-leisure occupation)
        "itc": -3.412040,  # alpha_{2,1}: intercept in indirect utility V_{2,1}
        "slp": 0.5,  # beta: slope on log wages, shared across all nodes
        "wge": 1.0628,  # wage w_{2,1}
        "qty": None,  # quantity supplied N_{2,1}: to be solved for
        "qtp": 3.581734,  # L_2: total worker pool for worker type 1
        "lyr": 0,  # layer index (flat supply system)
    },
    # Node: worker type 0 (wkr=0), occupation 1 (occ=1)
    # Intercept alpha_{1,2}=-2.997025; slope beta=0.5; total worker pool L_1=4.913439
    {
        "key_node": 3,  # unique node ID for this supply node
        "wkr": 0,  # worker type index 0 (type 1 in 1-indexed notation)
        "occ": 1,  # occupation index 1 (second non-leisure occupation)
        "itc": -2.997025,  # alpha_{1,2}: intercept in indirect utility V_{1,2}
        "slp": 0.5,  # beta: slope on log wages, shared across all nodes
        "wge": 1.1306,  # wage w_{1,2}
        "qty": None,  # quantity supplied N_{1,2}: to be solved for
        "qtp": 4.913439,  # L_1: total worker pool for worker type 0
        "lyr": 0,  # layer index (flat supply system)
    },
    # Node: worker type 1 (wkr=1), occupation 1 (occ=1)
    # Intercept alpha_{2,2}=-1.140797; slope beta=0.5; total worker pool L_2=3.581734
    {
        "key_node": 4,  # unique node ID for this supply node
        "wkr": 1,  # worker type index 1 (type 2 in 1-indexed notation)
        "occ": 1,  # occupation index 1 (second non-leisure occupation)
        "itc": -1.140797,  # alpha_{2,2}: intercept in indirect utility V_{2,2}
        "slp": 0.5,  # beta: slope on log wages, shared across all nodes
        "wge": 1.0519,  # wage w_{2,2}
        "qty": None,  # quantity supplied N_{2,2}: to be solved for
        "qtp": 3.581734,  # L_2: total worker pool for worker type 1
        "lyr": 0,  # layer index (flat supply system)
    },
]

df_supply_params = pd.DataFrame(data)
print(df_supply_params)

# %%
# Step 2, convert the dataframe to a supply dictionary
# ---------------------------------------------------------

dc_supply_lgt = cme_inpt_convert.cme_convert_pd2dc(
    df_supply_params, input_type="supply", verbose=True
)
print(dc_supply_lgt)

# %%
# Step 3, solve the quantity supplied
# ---------------------------------------------------------
# The supply system is solved by calling the supply solver,
# which takes in the supply dictionary as input, and solves for the
# quantity supplied given wages and total worker pool. The quantity supplied in
# each node in the is stored in `dc_supply_lgt` after the solver is
# called.

dc_supply_lgt, pd_qtlv_all_supply = cme_splv_opti.cme_supply_lgt_solver(dc_supply_lgt)
print(dc_supply_lgt)

# %%
# Step 4, Print the solutions for labor quantities as dataframe
# -------------------------------------------------------------------------------

print(pd_qtlv_all_supply)

# %%
