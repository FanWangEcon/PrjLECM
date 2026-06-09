# -*- coding: utf-8 -*-
r"""
Nested CES: marginal products along the tree with :func:`~prjlecm.demand.cme_dslv_eval.cme_prod_ces_nest_mpl`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Part 4 of https://github.com/FanWangEcon/PrjLECM/issues/7.

On this page, we:

1. Present the same nested CES production tree used on :doc:`ffv_cme_dslv_eval_agg_q_p` and map each economic object to ``key_node`` ids.
2. Derive subnest-specific and overall (chain-rule) marginal products of labor (MPL).
3. Define the five node-level MPL-related keys used in the code: ``drc``, ``drv``, ``shr``, ``shc``, and ``sni``.
4. Implement the same logic in code with :func:`~prjlecm.demand.cme_dslv_eval.cme_prod_ces_nest_mpl`.

We build on :doc:`ffv_cme_dslv_opti_nested` for the calibrated two-layer example and on :doc:`ffv_cme_dslv_eval_agg_q_p` for the production-tree setup.

Production function
===================

Suppose we have the following production function:

.. math::

    \begin{align}
    \begin{split}
    Y(x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}) &=
    \left(
        	heta_{1} \cdot
        \left(
             	heta_{1,1} \cdot x_{1,1}^{\psi_1} +
             	heta_{1,2} \cdot x_{1,2}^{\psi_1}
        \right)^{\frac{\psi}{\psi_1}} +
        	heta_{2} \cdot
        \left(
          	heta_{2,1} \cdot x_{2,1}^{\psi_2} +
          	heta_{2,2} \cdot x_{2,2}^{\psi_2}
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

Note that the :math:`\theta_{i,j}` values sum up to one within each nest, and the :math:`{i,j}` subscript denote individual :math:`i` and occupation :math:`j`. Additionally, for this example, we will assume that the elasticity of substitution between the two inputs in each nest of the canopy layer are the same.

Node labels
-----------

This example uses the same two-layer model as :doc:`ffv_cme_dslv_eval_agg_q_p`.

- node 1 :math:`\to` :math:`x_{1,1}`; node 2 :math:`\to` :math:`x_{1,2}`; node 3 :math:`\to` :math:`x_{2,1}`; node 4 :math:`\to` :math:`x_{2,2}`
- node 5 :math:`\to` :math:`Y_1`; node 6 :math:`\to` :math:`Y_2`; node 7 :math:`\to` :math:`Y`

Three subnest production functions
----------------------------------

Define the two lower-nest composites and the root nest:

.. math::

    Y_1(x_{1,1}, x_{1,2}) = \left(\theta_{1,1}x_{1,1}^{\psi_1}+\theta_{1,2}x_{1,2}^{\psi_1}\right)^{1/\psi_1},

.. math::

    Y_2(x_{2,1}, x_{2,2}) = \left(\theta_{2,1}x_{2,1}^{\psi_2}+\theta_{2,2}x_{2,2}^{\psi_2}\right)^{1/\psi_2},

.. math::

    Y(Y_1, Y_2) = \left(\theta_1Y_1^{\psi}+\theta_2Y_2^{\psi}\right)^{1/\psi}.

Marginal Product of Labor
=========================

Subnest-specific marginal product of labor
------------------------------------------

At the root nest (node 7), MPLs with respect to the two composite inputs are

.. math::

    \frac{\partial Y}{\partial Y_1}=Y^{1-\psi}\theta_1Y_1^{\psi-1},
    \qquad
    \frac{\partial Y}{\partial Y_2}=Y^{1-\psi}\theta_2Y_2^{\psi-1}.

At the first lower nest (node 5), MPLs with respect to canopy inputs are

.. math::

    \frac{\partial Y_1}{\partial x_{1,1}}=Y_1^{1-\psi_1}\theta_{1,1}x_{1,1}^{\psi_1-1},
    \qquad
    \frac{\partial Y_1}{\partial x_{1,2}}=Y_1^{1-\psi_1}\theta_{1,2}x_{1,2}^{\psi_1-1}.

At the second lower nest (node 6), MPLs with respect to canopy inputs are

.. math::

    \frac{\partial Y_2}{\partial x_{2,1}}=Y_2^{1-\psi_2}\theta_{2,1}x_{2,1}^{\psi_2-1},
    \qquad
    \frac{\partial Y_2}{\partial x_{2,2}}=Y_2^{1-\psi_2}\theta_{2,2}x_{2,2}^{\psi_2-1}.

Overall marginal product of labor
---------------------------------

For canopy inputs, overall MPL is the chain rule product of root and subnest derivatives:

.. math::

    \frac{\partial Y}{\partial x_{1,1}}=\frac{\partial Y}{\partial Y_1}\frac{\partial Y_1}{\partial x_{1,1}},\qquad
    \frac{\partial Y}{\partial x_{1,2}}=\frac{\partial Y}{\partial Y_1}\frac{\partial Y_1}{\partial x_{1,2}},

.. math::

    \frac{\partial Y}{\partial x_{2,1}}=\frac{\partial Y}{\partial Y_2}\frac{\partial Y_2}{\partial x_{2,1}},\qquad
    \frac{\partial Y}{\partial x_{2,2}}=\frac{\partial Y}{\partial Y_2}\frac{\partial Y_2}{\partial x_{2,2}}.

MPL-related keys tracked on each node
-------------------------------------

``drc`` is the cumulative derivative from the root to the current node.
For nodes 5 and 6, this is :math:`\partial Y/\partial Y_1` and :math:`\partial Y/\partial Y_2`.
For nodes 1-4, this is :math:`\partial Y/\partial x_{i,j}`.

``drv`` is the local derivative from the parent output to the current node input.
For nodes 5 and 6, this is :math:`\partial Y/\partial Y_1` and :math:`\partial Y/\partial Y_2`.
For nodes 1-4, this is :math:`\partial Y_1/\partial x_{1,j}` or :math:`\partial Y_2/\partial x_{2,j}`.

``shr`` is the node-level share parameter, i.e. the relevant :math:`\theta`.

``shc`` is the cumulative share product along the path from the root to the node.
On this tree, :math:`\texttt{shc}[5]=\theta_1`, :math:`\texttt{shc}[6]=\theta_2`,
:math:`\texttt{shc}[1]=\theta_1\theta_{1,1}`, :math:`\texttt{shc}[2]=\theta_1\theta_{1,2}`,
:math:`\texttt{shc}[3]=\theta_2\theta_{2,1}`, and :math:`\texttt{shc}[4]=\theta_2\theta_{2,2}`.

``sni`` is the share-nest intercept component used in the MPL recursion.
For this two-layer tree, it is the cumulative derivative with the own-input power term removed.
Concretely,

.. math::

    \texttt{sni}[1]=\frac{\texttt{drc}[1]}{x_{1,1}^{\psi_1-1}},\quad
    \texttt{sni}[2]=\frac{\texttt{drc}[2]}{x_{1,2}^{\psi_1-1}},\quad
    \texttt{sni}[3]=\frac{\texttt{drc}[3]}{x_{2,1}^{\psi_2-1}},\quad
    \texttt{sni}[4]=\frac{\texttt{drc}[4]}{x_{2,2}^{\psi_2-1}},

.. math::

    \texttt{sni}[5]=\frac{\texttt{drc}[5]}{Y_1^{\psi-1}},\quad
    \texttt{sni}[6]=\frac{\texttt{drc}[6]}{Y_2^{\psi-1}},\quad
    \texttt{sni}[7]=1.

"""

import copy

import numpy as np
import pandas as pd

import prjlecm.demand.cme_dslv_eval as cme_dslv_eval
import prjlecm.demand.cme_dslv_opti as cme_dslv_opti
import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand

np.set_printoptions(precision=8, suppress=True)

# %%
# Implementation
# ================
# We want to generate the various MPL-objects discussed above. They require all the
# demand parameters throughout the nested demand tree, and they also require the quantities
# at each node, from the canopy layer, towards the root.
# 
# We are proceeding:
# 
# (1) Given parameters and wages at the canopy layer, solve for canopy layer quantities.  
# (2) Generate quantities at all layers
# (3) Generate the MPL-related objects at all layers.
#
# We use the same tree structure as :doc:`ffv_cme_dslv_eval_agg_q_p` and
# :doc:`ffv_cme_dslv_opti_nested`; see that page for the shared key meanings and
# dictionary/dataframe layout.
#
# The code below shows the two-layer calibrated example first, then a simulated
# three-layer tree with homogeneous :math:`\psi=\tfrac{1}{2}`.
#
#
# Step 1, set up the two-layer demand tree
# ---------------------------------------------------------------------------
#
# Same rows as :doc:`ffv_cme_dslv_opti_nested`; the canopy wages are the calibrated
# equilibrium values used to illustrate the wage rollup.
#

data_two_layer = [
    {
        "key_node": 1,
        "lyr": 2,
        "prt": 5,
        "wkr": 0,
        "occ": 0,
        "shr": 0.30,
        "pwr": float("nan"),
        "ipt": None,
        "wge": 1.340259,
        "qty": None,
    },
    {
        "key_node": 2,
        "lyr": 2,
        "prt": 5,
        "wkr": 0,
        "occ": 1,
        "shr": 0.70,
        "pwr": float("nan"),
        "ipt": None,
        "wge": 12.671998,
        "qty": None,
    },
    {
        "key_node": 3,
        "lyr": 2,
        "prt": 6,
        "wkr": 1,
        "occ": 0,
        "shr": 0.10,
        "pwr": float("nan"),
        "ipt": None,
        "wge": 5.526463,
        "qty": None,
    },
    {
        "key_node": 4,
        "lyr": 2,
        "prt": 6,
        "wkr": 1,
        "occ": 1,
        "shr": 0.90,
        "pwr": float("nan"),
        "ipt": None,
        "wge": 7.251012,
        "qty": None,
    },
    {
        "key_node": 5,
        "lyr": 1,
        "prt": 7,
        "wkr": pd.NA,
        "occ": pd.NA,
        "shr": 0.50,
        "pwr": 0.20,
        "ipt": [1, 2],
        "wge": None,
        "qty": None,
    },
    {
        "key_node": 6,
        "lyr": 1,
        "prt": 7,
        "wkr": pd.NA,
        "occ": pd.NA,
        "shr": 0.50,
        "pwr": 0.20,
        "ipt": [3, 4],
        "wge": None,
        "qty": None,
    },
    {
        "key_node": 7,
        "lyr": 0,
        "prt": pd.NA,
        "wkr": pd.NA,
        "occ": pd.NA,
        "shr": float("nan"),
        "pwr": 0.70,
        "ipt": [5, 6],
        "wge": None,
        "qty": None,
    },
]

df_two = pd.DataFrame(data_two_layer)
print(df_two)


# %%
# Step 2, two-layer: solve quantities, then MPLs
# ------------------------------------------------
# ``cme_prod_ces_nested_solver`` fills all ``qty`` at :math:`\overline{Y}=1``.
# ``cme_prod_ces_nest_mpl`` then fills ``drc``, ``shc``, ``sni``, etc.

dc_two = cme_inpt_convert.cme_convert_pd2dc(
    df_two, input_type="demand", verbose=False
)
dc_two = cme_dslv_opti.cme_prod_ces_nested_solver(
    dc_two, fl_Q_agg=1.0, verbose=False, verbose_debug=False
)
dc_two = cme_dslv_eval.cme_prod_ces_nest_mpl(
    dc_two, verbose=False, verbose_debug=False
)

print("Inner nodes and root after MPL (chain factors drc_parent * drv -> canopy drc):")
for kn in [7, 5, 6]:
    row = dc_two[kn]
    drv_s = f"{row['drv']:.8f}" if row.get("drv") is not None else "None"
    print(
        f"  node {kn}: qty={row['qty']:.8f}, drv={drv_s}, drc={row['drc']:.8f}, "
        f"shc={row['shc']:.8f}"
    )

print("Canopy MPL-related fields (two-layer example):")
for kn in [1, 2, 3, 4]:
    row = dc_two[kn]
    print(
        f"  node {kn}: qty={row['qty']:.8f}, drv={row['drv']:.8f}, drc={row['drc']:.8f}, "
        f"shc={row['shc']:.8f}, sni={row['sni']:.8f}"
    )

print(
    "Chain check (node 1): drc[5]*drv[1] =",
    f"{dc_two[5]['drc'] * dc_two[1]['drv']:.8f},",
    "drc[1] =",
    f"{dc_two[1]['drc']:.8f}",
)

# Nodes 1 and 2 share the same inner psi_1 = 0.20; sni ratio should match shc ratio.
_r_sni = dc_two[1]["sni"] / dc_two[2]["sni"]
_r_shc = dc_two[1]["shc"] / dc_two[2]["shc"]
print("sni ratio (node 1 / node 2):", _r_sni)
print("shc ratio (node 1 / node 2):", _r_shc)
print("difference (should be near 0):", abs(_r_sni - _r_shc))


# %%
# Step 3, three-layer simulated tree (fixed seed, homogeneous ``pwr``)
# ----------------------------------------------------------------------
# We use ``cme_simu_demand_params_ces_nested`` with ``ar_it_chd_tre=[2, 2, 2]``,
# ``it_seed=222``, and ``fl_power_min=fl_power_max=0.5`` so every aggregator
# ``pwr`` equals :math:`\psi=\tfrac{1}{2}`. Canopy ``qty`` is simulated; we roll
# quantities to the root with ``st_solve_type='qty'``, then evaluate MPLs.

dc_simu_wrap = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
    ar_it_chd_tre=[2, 2, 2],
    ar_it_occ_lyr=[2],
    fl_power_min=0.5,
    fl_power_max=0.5,
    it_seed=222,
    bl_simu_q=True,
    bl_simu_p=False,
    verbose=False,
    verbose_debug=False,
)
dc_three = copy.deepcopy(dc_simu_wrap["dc_ces_flat"])
dc_three = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
    dc_three, st_solve_type="qty", verbose=False, verbose_debug=False
)
dc_three = cme_dslv_eval.cme_prod_ces_nest_mpl(
    dc_three, verbose=False, verbose_debug=False
)

# Canopy layer is max lyr (here 3)
mx_lyr = max(dc_three[k]["lyr"] for k in dc_three)
print(f"Three-layer example: max lyr = {mx_lyr} (canopy). Canopy MPL fields:")
for kn in sorted(k for k in dc_three if dc_three[k]["lyr"] == mx_lyr):
    row = dc_three[kn]
    print(
        f"  node {kn}: qty={row['qty']:.8f}, drc={row['drc']:.8f}, "
        f"shc={row['shc']:.8f}, sni={row['sni']:.8f}"
    )