# -*- coding: utf-8 -*-
r"""
Solving a 2-layer nested CES optimal demand problem (cost-minimization given output quantity and wages)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Part 2 of https://github.com/FanWangEcon/PrjLECM/issues/7

On this page, we provide demand side equations for a nested CES problem. There are
three nests in three layers:

1. the root (layer 0) branches out towards two nests (layer 1).
2. each of the nest branches out towards two canopy nodes (layer 2).

Given wages output, quantity target, and all CES parameters, we solve for cost-minimizing demand for labor.

We build on the :doc:`ffv_cme_dslv_opti_nonnested` file, which solves the non-nested-CES demand system problem. 

Suppose we have the following production function:

Production function
===================

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
      \left(
          0.30 \cdot x_{1,1}^{0.20} + 
          0.70 \cdot x_{1,2}^{0.20}
      \right)^{\frac{0.70}{0.20}} + 
      \left(
          0.10 \cdot x_{2,1}^{0.20} + 
          0.90 \cdot x_{2,2}^{0.20}
      \right)^{\frac{0.70}{0.20}}
   \right)^{\frac{1}{0.70}}\\
   \end{split}
   \end{align}

Note that the :math:`\theta_{i,j}` values sum up to one within each nest, and the :math:`{i,j}` subscript denote individual :math:`i` and occupation :math:`j`. Additionally, for this example, we will assume that the elasticity of substitution between the two inputs in each nest of the canopy layer are the same.

Cost minimization problem
=========================

Given the production function above, we solve the following cost minimization problem, given prices :math:`w_{i,j}` and output quantity target :math:`\overline{Y}`:

.. math::

   \begin{align}
   \begin{split}
   \min_{x_{i,j}} &\sum_{i=1}^2 \sum_{j=1}^2 w_{i,j} \cdot x_{i,j}\\
   s.t. & \overline{Y} =  Y(x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2})
   \end{split}
   \end{align}
   
To solve the problem, we need: (1) wages :math:`w_{i,j}` for each input, and (2)
output quantity target :math:`\overline{Y}`, as well as (3) the production function
parameters :math:`\theta_{i,j}` and :math:`\psi`. 
   
Optimality conditions and optimal choices
=========================================

Optimality conditions and optimal choices are shown for example on this page: 
`multi-input CES optimality conditions <https://fanwangecon.github.io/Py4Econ/prod/ces/htmlpdfr/fs_ces_multi_input.html>`_

Storing parameters, prices, and quantities in as a tree, stored in dictionary and dataframes
==============================================================================================
How do we store the information from a CES demand system? We do so as a tree, from root to canopy, with branching nests, assuming that each node can only have a single parent node.

- **root**: that is the bottom-most layer, the final aggregation
- **canopy**: that is the top-most layer, the canopy nodes, which have corresponding inputs with wages and final demand quantities
- **nest**: the tree is a stack of linked nests. Each node, has a parent node, and has children nodes. A node and its child-nodes form a nest. The canopy nodes do not have children nodes, and the root node does not have a parent node.

For each node, we store the following information:

1. its node ID.
2. ID of its parent node.
3. A list of IDs of its child nodes.
4. a "power" parameter for the aggregation over its child nodes. This is the :math:`\psi` parameter for the CES function, which is nest-specific.
5. a "share" parameter for its relative weight as a child among siblings in its parent nest. This is the :math:`\theta_{i,j}` parameter for the CES function, which is specific to each node within the parent nest, and aggregates up to 1 within each nest. 

For canopy nodes, we have some additional information. To help with the identity of each node, in an occupational-choice model setting where workers fall into different worker types and choose among occupations, for nodes in the canopy layer, we also store the following information:

1. an occupation id 
2. a worker type id
3. the wage paid for one unit of this input (to be solved for in equilibrium setting)
4. the quantity demanded of this input (to be solved for in equilibrium but given wages)

There are two ways of thinking about wages and quantity from a solution point of view:

- The wage and quantity values could be solved in equilibrium given parameters. This is what we do in file :doc:`../equi/ffv_sme_equi_solve_gen_inputs.py`, where we solve for equilibrium wages and quantities given parameters on the demand and supply side.
- Given wages, we can solve for quantities, this is what we do here on this page.

"""

import pandas as pd
import numpy as np
import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.demand.cme_dslv_opti as cme_dslv_opti
import prjlecm.input.cme_inpt_parse_wgqt as cme_inpt_parse_wgqt

# %%
# Implementation
# ================
# Our implementation involves three steps. First, we create a dataframe storing
# all CES production function information, along with wages. Second, we convert the
# dataframe to a demand dictionary. Third, we call the demand cost minimization solver to solve for optimal demand.
#
#
# Step 1, set up demand parameters dataframe, with wages
# ---------------------------------------------------------
#
# Build rows as dicts so each key is explicitly paired with its value, column key meanings:
#
#  * **key_node** : unique integer ID for this node in the CES tree
#  * **lyr**      : layer index; lyr=0 is the root aggregate node, counting from the 0th layer downward, we can have lyr=1,2,..., for this test, non-nested only two layers, 0 and 1. Think about this as a tree, where at the bottom layer = 0, we have the root. and at the top, we have the canopy layer, this layer will have the largest layer numbering.
#  * **prt**      : key_node of the parent node that aggregates this input (NA for the root)
#  * **wkr**      : worker-type index identifying which worker group supplies this input (NA for root and non-canopy nodes)
#  * **occ**      : occupation index identifying which occupation slot this input fills (NA for root non-canopy nodes)
#  * **shr**      : CES share parameter theta_{i,j}; governs each input's weight in each nest (NaN for root)
#  * **pwr**      : CES power parameter psi = (sigma-1)/sigma; defined only on the root/aggregate node (NaN for canopy nodes)
#  * **ipt**      : list of key_node IDs that feed into this node as child nodes (None for canopy nodes)
#  * **qty**      : quantity demanded of this input (to be solved for; None for now)
#  * **wge**      : wage, input price w_{i,j} paid for one unit of this canopy input (available only for canopy nodes)
#
#  To create the table, we generate data row by row below as a list of dictionaries, and then convert the list of dicts to a dataframe. We also print the dataframe to check that it looks correct.
#

data = [
    # Canopy node: worker type 0 (wkr=0), occupation 0 (occ=0), i.e. x_{1,1}
    # Parent is the root node (prt=0); has no sub-inputs (ipt=None)
    {
        "key_node": 1,  # unique node ID for this canopy
        "lyr": 2,  # canopy layer, layer = 1 is canopy since this is non-nested
        "prt": 5,  # parent node is the root node (key_node=5)
        "wkr": 0,  # worker type index 0
        "occ": 0,  # occupation index 0
        "shr": 0.30,  # theta_{1,1}: share weight in CES aggregate
        "pwr": float("nan"),  # power psi not defined at canopy level
        "ipt": None,  # canopy nodes have no sub-inputs
        "wge": 0.9185,  # wage w_{1,1}
        "qty": None,  # quantity demanded to be solved for
    },
    # Canopy node: worker type 0 (wkr=0), occupation 1 (occ=1), i.e. x_{1,2}
    {
        "key_node": 2,  # unique node ID for this canopy
        "lyr": 2,  # canopy layer, layer = 1 is canopy since this is non-nested
        "prt": 5,  # parent node is the root node (key_node=5)
        "wkr": 0,  # worker type index 0
        "occ": 1,  # occupation index 1
        "shr": 0.70,  # theta_{1,2}: share weight in CES aggregate
        "pwr": float("nan"),  # power psi not defined at canopy level
        "ipt": None,  # canopy nodes have no sub-inputs
        "wge": 1.1306,  # wage w_{1,2}
        "qty": None,  # quantity demanded to be solved for
    },
    # Canopy node: worker type 1 (wkr=1), occupation 0 (occ=0), i.e. x_{2,1}
    {
        "key_node": 3,  # unique node ID for this canopy
        "lyr": 2,  # canopy layer, layer = 1 is canopy since this is non-nested
        "prt": 6,  # parent node is the root node (key_node=6)
        "wkr": 1,  # worker type index 1
        "occ": 0,  # occupation index 0
        "shr": 0.10,  # theta_{2,1}: share weight in CES aggregate
        "pwr": float("nan"),  # power psi not defined at canopy level
        "ipt": None,  # canopy nodes have no sub-inputs
        "wge": 1.0628,  # wage w_{2,1}
        "qty": None,  # quantity demanded to be solved for
    },
    # Canopy node: worker type 1 (wkr=1), occupation 1 (occ=1), i.e. x_{2,2}
    {
        "key_node": 4,  # unique node ID for this canopy
        "lyr": 2,  # canopy layer, layer = 1 is canopy since this is non-nested
        "prt": 6,  # parent node is the root node (key_node=6)
        "wkr": 1,  # worker type index 1
        "occ": 1,  # occupation index 1
        "shr": 0.90,  # theta_{2,2}: share weight in CES aggregate
        "pwr": float("nan"),  # power psi not defined at canopy level
        "ipt": None,  # canopy nodes have no sub-inputs
        "wge": 1.0519,  # wage w_{2,2}
        "qty": None,  # quantity demanded to be solved for
    },
    # Aggregator over notes 1 and 2
    # pwr = psi = 0.20 corresponds to sigma = 1/(1-psi) ~= 1.25
    # ipt lists the key_node IDs of the two direct inputs; no worker/occ/share/wage at this level
    {
        "key_node": 5,  # root node ID (by convention 0)
        "lyr": 1,  # aggregate layer (root)
        "prt": 7,  # parent node is the root node (key_node=7)
        "wkr": pd.NA,  # not applicable at aggregate level
        "occ": pd.NA,  # not applicable at aggregate level
        "shr": 0.50,  # share parameter at layer 1
        "pwr": 0.20,  # psi = (sigma-1)/sigma; governs elasticity of substitution
        "ipt": [1, 2],  # key_node IDs of the two canopy inputs to this aggregate
        "wge": None,  # aggregate input price/wage to be constructed
        "qty": None,  # quantity demanded to be solved for at the aggregate level
    },
    # Aggregator over notes 3 and 4
    # pwr = psi = 0.20 corresponds to sigma = 1/(1-psi) ~= 1.25, same as the other nest
    # ipt lists the key_node IDs of the two direct inputs; no worker/occ/share/wage at this level
    {
        "key_node": 6,  # root node ID (by convention 0)
        "lyr": 1,  # aggregate layer (root)
        "prt": 7,  # parent is the root node (key_node=7)
        "wkr": pd.NA,  # not applicable at aggregate level
        "occ": pd.NA,  # not applicable at aggregate level
        "shr": 0.50,  # share parameter at layer 1
        "pwr": 0.20,  # psi = (sigma-1)/sigma; governs elasticity of substitution
        "ipt": [3, 4],  # key_node IDs of the two canopy inputs to this aggregate
        "wge": None,  # aggregate input price/wage to be constructed
        "qty": None,  # quantity demanded to be solved for at the aggregate level
    },
    # Root / aggregate node: produces aggregate output Y from all four canopy inputs
    # pwr = psi = 0.603628 corresponds to sigma = 1/(1-psi) ~= 2.5
    # ipt lists the key_node IDs of the four direct inputs; no worker/occ/share/wage at this level
    {
        "key_node": 7,  # root node ID (by convention 0)
        "lyr": 0,  # aggregate layer (root)
        "prt": pd.NA,  # root has no parent
        "wkr": pd.NA,  # not applicable at aggregate level
        "occ": pd.NA,  # not applicable at aggregate level
        "shr": float("nan"),  # no share parameter at the root
        "pwr": 0.70,  # psi = (sigma-1)/sigma; governs elasticity of substitution
        "ipt": [5, 6],  # key_node IDs of the four canopy inputs to this aggregate
        "wge": None,  # no wage at the root
        "qty": None,  # quantity demanded to be solved for at the aggregate level
    },
]

df_demand_params = pd.DataFrame(data)
print(df_demand_params)


# %%
# Step 2, convert the dataframe to a demand dictionary
# ---------------------------------------------------------

dc_demand_ces = cme_inpt_convert.cme_convert_pd2dc(
    df_demand_params, input_type="demand", verbose=True
)
print(dc_demand_ces)

# %%
# Step 3, solve the non-nested CES demand system
# ---------------------------------------------------------
# The demand solutions are found by calling the cost minimization
# solver, which takes in the demand dictionary as input, and solves for the
# cost-minimizing demand quantities given wages and output quantity target. We
# set the output quantity target to 1 for this test. The quantity demanded in
# each node in the canopy is stored in `dc_demand_ces` after the solver is
# called.
# 
# How are we solving through the nested system?
#
# 1. We know at the wages at the outter-most canopy layer and we know all the paraemters of the production system.
# 2. Given (1), we can construct MC of one more unit of aggregate output each successive higher nest. This is a function of parameters and prices.
# 3. Then we start from the root, solving for optimal quantity demanded of each successive node/nest, going from the root up to the nest for each subsets of canopy nodes.
# 4. Now we have the prices, parameter, and quantity target for each of the nest for the canopy nodes, and can solve for the expenditure minimization problem.
# 
# This solution structure works for any layer of nesting. 

fl_output_target = 1
dc_demand_ces = cme_dslv_opti.cme_prod_ces_nested_solver(
    dc_demand_ces, fl_Q_agg=fl_output_target, verbose=False, verbose_debug=False
)
print(dc_demand_ces)

# %%
# Step 4, retrieve the equilibrium demand quantity results from demand dictionary
# -------------------------------------------------------------------------------

# ar_splv_totl_acrs_i = np.array([4.913439, 3.581734])
pd_qtlv_all_demand, __ = cme_inpt_parse_wgqt.cme_parse_qtwg_dc2pd_demand(
    dc_demand_ces, ar_splv_totl_acrs_i=None
)
print(pd_qtlv_all_demand)

# %%
