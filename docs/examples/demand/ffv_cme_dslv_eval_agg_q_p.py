# -*- coding: utf-8 -*-
r"""
Nested CES: aggregate composite wages and quantities with :func:`~prjlecm.demand.cme_dslv_eval.cme_prod_ces_nest_agg_q_p`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Part 3 of https://github.com/FanWangEcon/PrjLECM/issues/7.

On this page, we:

1. Define the 2-layer nested CES production tree and map economic objects to ``key_node`` ids (1–7).
2. Write the full expenditure-minimization problem and its three nested subproblems, highlighting how root-level :math:`Y_1` and :math:`Y_2` choices correspond to :math:`\hat{Y}_1` and :math:`\hat{Y}_2` passed to lower nests.
3. Derive the first subnest completely to obtain the unit-cost expression (marginal cost), then apply the same formula to the second subnest and root to construct :math:`w_2` and :math:`w_Y` from child wages.
4. Demonstrate the two evaluation passes implemented by :func:`~prjlecm.demand.cme_dslv_eval.cme_prod_ces_nest_agg_q_p`: wage rollup (canopy wages :math:`\to` aggregate wages) and quantity rollup (canopy quantities :math:`\to` aggregate quantities).

We build on :doc:`ffv_cme_dslv_opti_nested`, which solves the nested optimization problem. This page isolates the **evaluation** implications of that solution and shows how they are computed layer by layer on the tree.

Suppose we have the following production function:

Production function
===================

Suppose we have the following production function:

.. math::

   \begin{align}
   \begin{split}
   Y(x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}) &=
   \left(
      \theta_{1} \cdot
      \left(
          \theta_{1,1} \cdot x_{1,1}^{\psi_1} +
          \theta_{1,2} \cdot x_{1,2}^{\psi_1}
      \right)^{\frac{\psi}{\psi_1}} +
      \theta_{2} \cdot
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

Note that the :math:`\theta_{i,j}` values sum up to one within each nest, and the :math:`{i,j}` subscript denote individual :math:`i` and occupation :math:`j`. Additionally, for this example, we will assume that the elasticity of substitution between the two inputs in each nest of the canopy layer are the same.

Node labels
-----------

Each economic object corresponds to a **key_node** in the flat tree:

- node 1 :math:`\to` :math:`x_{1,1}`; node 2 :math:`\to` :math:`x_{1,2}`; node 3 :math:`\to` :math:`x_{2,1}`; node 4 :math:`\to` :math:`x_{2,2}`
- node 5 :math:`\to` :math:`Y_1`; node 6 :math:`\to` :math:`Y_2`; node 7 :math:`\to` :math:`Y`

Three subnest production functions
----------------------------------

The nested display above embeds **three** CRS CES aggregators. Define composite outputs :math:`Y_1` and :math:`Y_2` for the two lower nests and :math:`Y` for the root:

**Subnest at node 5** (over inputs :math:`x_{1,1}`, :math:`x_{1,2}`):

.. math::

   \begin{align}
   Y_1(x_{1,1}, x_{1,2}) &=
   \left(
      \theta_{1,1} \cdot x_{1,1}^{\psi_1} +
      \theta_{1,2} \cdot x_{1,2}^{\psi_1}
   \right)^{1/\psi_1} \\
   &=
   \left(
      0.30 \cdot x_{1,1}^{0.20} +
      0.70 \cdot x_{1,2}^{0.20}
   \right)^{1/0.20}.
   \end{align}

**Subnest at node 6** (over inputs :math:`x_{2,1}`, :math:`x_{2,2}`):

.. math::

   \begin{align}
   Y_2(x_{2,1}, x_{2,2}) &=
   \left(
      \theta_{2,1} \cdot x_{2,1}^{\psi_2} +
      \theta_{2,2} \cdot x_{2,2}^{\psi_2}
   \right)^{1/\psi_2} \\
   &=
   \left(
      0.10 \cdot x_{2,1}^{0.20} +
      0.90 \cdot x_{2,2}^{0.20}
   \right)^{1/0.20}.
   \end{align}

**Root nest at node 7** (over composites :math:`Y_1`, :math:`Y_2`):

.. math::

   \begin{align}
   Y(Y_1, Y_2) &=
   \left(
      \theta_{1} \cdot Y_1^{\psi} +
      \theta_{2} \cdot Y_2^{\psi}
   \right)^{1/\psi} \\
   &=
   \left(
      0.50 \cdot Y_1^{0.70} +
      0.50 \cdot Y_2^{0.70}
   \right)^{1/0.70}.
   \end{align}

Substituting the first two equations into the third recovers the four-input :math:`Y(x_{1,1},x_{1,2},x_{2,1},x_{2,2})` display at the top.

Cost minimization problem
=========================

Overall problem (all four canopy inputs)
----------------------------------------

Given canopy wages :math:`w_{1,1}, w_{1,2}, w_{2,1}, w_{2,2}` and root output requirement :math:`\hat{Y}`, the full expenditure-minimization problem is

.. math::

   \begin{align}
   \min_{x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}}\quad
   & w_{1,1} x_{1,1} + w_{1,2} x_{1,2} + w_{2,1} x_{2,1} + w_{2,2} x_{2,2} \\
   \text{s.t.}\quad
   & \hat{Y} = Y(x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}).
   \end{align}

To solve it we need: (1) the four canopy wages, (2) all production parameters :math:`\theta_{i,j}` and :math:`\psi`, :math:`\psi_1`, :math:`\psi_2`, and (3) the aggregate output target :math:`\hat{Y}` at node 7.

+------------------+-----------+-----------------------------------------------+
| key_node         | lyr       | Role                                          |
+==================+===========+===============================================+
| 1                | 2 (canopy)| input :math:`x_{1,1}`, wage :math:`w_{1,1}`   |
+------------------+-----------+-----------------------------------------------+
| 2                | 2         | input :math:`x_{1,2}`, wage :math:`w_{1,2}`   |
+------------------+-----------+-----------------------------------------------+
| 3                | 2         | input :math:`x_{2,1}`, wage :math:`w_{2,1}`   |
+------------------+-----------+-----------------------------------------------+
| 4                | 2         | input :math:`x_{2,2}`, wage :math:`w_{2,2}`   |
+------------------+-----------+-----------------------------------------------+
| 5                | 1         | composite output :math:`Y_1`, price :math:`w_1` |
+------------------+-----------+-----------------------------------------------+
| 6                | 1         | composite output :math:`Y_2`, price :math:`w_2` |
+------------------+-----------+-----------------------------------------------+
| 7                | 0 (root)  | aggregate output :math:`Y`, price :math:`w_Y` |
+------------------+-----------+-----------------------------------------------+

Three nested cost-minimization subproblems
------------------------------------------

The overall problem decomposes into three single-nest problems—one per aggregator node. Solving the root nest gives the composite targets that feed the lower nests: the optimal :math:`Y_1` and :math:`Y_2` from the root problem become :math:`\hat{Y}_1` and :math:`\hat{Y}_2` for the two lower nests, and because the tree is CRS these are exactly the composite values passed down from the root solution.

**Nest over nodes 1 and 2 (node 5).** Given composite output requirement :math:`\hat{Y}_1`,

.. math::

   \begin{align}
   \min_{x_{1,1}, x_{1,2}}\quad
   & w_{1,1} x_{1,1} + w_{1,2} x_{1,2} \\
   \text{s.t.}\quad
   & \hat{Y}_1 = Y_1(x_{1,1}, x_{1,2}).
   \end{align}

**Nest over nodes 3 and 4 (node 6).** Given :math:`\hat{Y}_2`,

.. math::

   \begin{align}
   \min_{x_{2,1}, x_{2,2}}\quad
   & w_{2,1} x_{2,1} + w_{2,2} x_{2,2} \\
   \text{s.t.}\quad
   & \hat{Y}_2 = Y_2(x_{2,1}, x_{2,2}).
   \end{align}

**Root nest over :math:`Y_1` and :math:`Y_2` (node 7).** Given aggregate prices :math:`w_1`, :math:`w_2` for the two composites (derived below) and root requirement :math:`\hat{Y}`,

.. math::

   \begin{align}
   \min_{Y_1, Y_2}\quad
   & w_1 Y_1 + w_2 Y_2 \\
   \text{s.t.}\quad
   & \hat{Y} = Y(Y_1, Y_2).
   \end{align}

We do **not** know :math:`\hat{Y}_1` and :math:`\hat{Y}_2` when we only observe canopy wages. They are determined jointly when the full tree is solved. The construction below explains how to build aggregate **prices** :math:`w_1`, :math:`w_2`, :math:`w_Y` from canopy wages and parameters alone.

Fully worked solution — first subnest (nodes 1 and 2)
------------------------------------------------------

1. **Problem.** Minimize :math:`w_{1,1} x_{1,1} + w_{1,2} x_{1,2}` subject to :math:`\hat{Y}_1 = Y_1(x_{1,1},x_{1,2})` with :math:`Y_1` from the subnest display above.

2. **Optimal cost-minimizing demands** (see `multi-input CES optimality conditions <https://fanwangecon.github.io/Py4Econ/prod/ces/htmlpdfr/fs_ces_multi_input.html>`_):

.. math::

   x_{1,j}^*
   = \hat{Y}_1
   \left[
     \sum_{k \in \{1,2\}} \theta_{1,k}
     \left(
       \frac{w_{1,k}}{w_{1,j}}
       \frac{\theta_{1,j}}{\theta_{1,k}}
     \right)^{\frac{\psi_1}{1-\psi_1}}
   \right]^{-\frac{1}{\psi_1}},
   \qquad j \in \{1,2\},

where :math:`w_{1,1}, w_{1,2}` and :math:`\theta_{1,1}, \theta_{1,2}` denote the two canopy wages and shares in this nest.

3. **Cost function** — substitute the optima into expenditure (do **not** define marginal cost yet):

.. math::

   C_1(\hat{Y}_1) = w_{1,1}\, x_{1,1}^* + w_{1,2}\, x_{1,2}^*.

Each optimum is proportional to :math:`\hat{Y}_1`, so the cost function is linear in :math:`\hat{Y}_1`:

.. math::

    C_1(\hat{Y}_1)
    = \hat{Y}_1\Bigg[
       w_{1,1}
       \left(
          \sum_{k \in \{1,2\}} \theta_{1,k}
          \left(
             \frac{w_{1,k}}{w_{1,1}}
             \frac{\theta_{1,1}}{\theta_{1,k}}
          \right)^{\frac{\psi_1}{1-\psi_1}}
       \right)^{-\frac{1}{\psi_1}}
       +
       w_{1,2}
       \left(
          \sum_{k \in \{1,2\}} \theta_{1,k}
          \left(
             \frac{w_{1,k}}{w_{1,2}}
             \frac{\theta_{1,2}}{\theta_{1,k}}
          \right)^{\frac{\psi_1}{1-\psi_1}}
       \right)^{-\frac{1}{\psi_1}}
    \Bigg].

4. **Marginal cost** — differentiate the cost function with respect to :math:`\hat{Y}_1`:

.. math::

    MC_1 = \frac{\partial C_1}{\partial \hat{Y}_1}
    = w_{1,1}
       \left(
          \sum_{k \in \{1,2\}} \theta_{1,k}
          \left(
             \frac{w_{1,k}}{w_{1,1}}
             \frac{\theta_{1,1}}{\theta_{1,k}}
          \right)^{\frac{\psi_1}{1-\psi_1}}
       \right)^{-\frac{1}{\psi_1}}
       +
       w_{1,2}
       \left(
          \sum_{k \in \{1,2\}} \theta_{1,k}
          \left(
             \frac{w_{1,k}}{w_{1,2}}
             \frac{\theta_{1,2}}{\theta_{1,k}}
          \right)^{\frac{\psi_1}{1-\psi_1}}
       \right)^{-\frac{1}{\psi_1}}.

Because :math:`C_1` is linear in :math:`\hat{Y}_1`, marginal cost is constant (CRS). This is the standard CES unit-cost formula (equivalent to :func:`~prjlecm.demand.cme_dslv_opti.cme_prod_ces_solver`).

5. **Interpretation.** Define :math:`w_1 := MC_1` as the aggregate price of composite output :math:`Y_1` at node 5.

Parallel results for the second subnest and the root
----------------------------------------------------

**Nest over nodes 3 and 4 (node 6).** The same CES unit-cost logic gives the marginal cost :math:`MC_2` from :math:`(w_{2,1}, w_{2,2})`, :math:`(\theta_{2,1}, \theta_{2,2})`, and :math:`\psi_2`; define :math:`w_2 := MC_2`.

**Root nest (node 7).** Treat :math:`(Y_1, Y_2)` as the two inputs at prices :math:`(w_1, w_2)` with shares :math:`(\theta_1, \theta_2)` and power :math:`\psi`. The same template gives the root marginal cost :math:`MC_Y`; define :math:`w_Y := MC_Y`.

Solving the three nested problems with internally consistent :math:`\hat{Y}_1`, :math:`\hat{Y}_2`, and :math:`\hat{Y}` yields the **same** overall minimum cost as the four-input problem above. That is what :func:`~prjlecm.demand.cme_dslv_opti.cme_prod_ces_nested_solver` automates.

Given four observed canopy wages and all nest parameters, the **same** single-nest marginal-cost formula applies at every layer. Under CRS, :math:`w_1`, :math:`w_2`, and :math:`w_Y` can be computed **without knowing** :math:`\hat{Y}_1` or :math:`\hat{Y}_2`: each aggregate price depends only on child wages and :math:`\theta`, :math:`\psi` at that nest.

Fix a nest with child index set :math:`\mathcal{J}`, child wages :math:`\{w_j : j \in \mathcal{J}\}`, child shares :math:`\{\theta_j\}`, and nest power :math:`\psi_n`. The aggregate input price for that nest's composite output is

.. math::

   w_{\text{agg}}
   = \sum_{j \in \mathcal{J}} w_j
   \left[
     \sum_{k \in \mathcal{J}} \theta_k
     \left(
       \frac{w_k}{w_j}\,\frac{\theta_j}{\theta_k}
     \right)^{\frac{\psi_n}{1-\psi_n}}
   \right]^{-\frac{1}{\psi_n}}.

Applied to this tree:

- Nest 5: :math:`w_1` from :math:`(w_{1,1}, w_{1,2})` and :math:`(\theta_{1,1}, \theta_{1,2}, \psi_1)`.
- Nest 6: :math:`w_2` from :math:`(w_{2,1}, w_{2,2})` and :math:`(\theta_{2,1}, \theta_{2,2}, \psi_2)`.
- Nest 7: :math:`w_Y` from :math:`(w_1, w_2)` and :math:`(\theta_1, \theta_2, \psi)`.

Each layer toward the root reuses the same structure; only the **input prices** change, from canopy wages to previously constructed aggregate prices.

See :doc:`ffv_cme_dslv_opti_nested` for the flat-tree key meanings, parameter conventions, and the companion optimization-side setup.

"""

import copy

import pandas as pd

import prjlecm.demand.cme_dslv_eval as cme_dslv_eval
import prjlecm.demand.cme_dslv_opti as cme_dslv_opti
import prjlecm.input.cme_inpt_convert as cme_inpt_convert

# %%
# Implementation
# ================
# This example uses one function for two objectives:
# (1) given canopy wages and the full parameter tree, roll aggregate wages from the canopy to the root; this is not solving the optimization problem directly, but using its implications to construct the aggregate wages;
# (2) given canopy quantities (solved now given wages) but not wages, roll aggregate quantities from the canopy to the root.
#
# Flow: (1) build demand dataframe; (2) convert to dictionary; (3) evaluate aggregate
# prices w_1, w_2, w_Y via ``st_solve_type='wge'`` (stored as wge[5], wge[6], wge[7]);
# (4) reference canopy quantities from nested solver at hat{Y}=1; (5) qty rollup.
#
#
# Step 1, set up the CES tree data
# ---------------------------------------------------------
#
# Build the flat tree as rows of dicts so the same structure can support both
# the wage rollup and the quantity rollup. Column key meanings:
#
#  * **key_node** : unique integer ID for this node in the CES tree
#  * **lyr**      : layer index; lyr=0 is the root, canopy is the largest lyr
#  * **prt**      : key_node of the parent node (NA for the root)
#  * **wkr** / **occ** : worker type and occupation at the canopy
#  * **shr**      : CES share :math:`\theta`
#  * **pwr**      : CES power :math:`\psi` on aggregator nodes (NaN at canopy)
#  * **ipt**      : list of child key_node IDs (None at canopy)
#  * **qty**      : canopy quantity seeds for the primal rollup; inner values filled upward later
#  * **wge**      : canopy :math:`w_{i,j}`; inner values filled as :math:`w_1`, :math:`w_2`, :math:`w_Y`
#
# The canopy wages here are the equilibrium wages from :doc:`ffv_cme_dslv_opti_nested`.
#

data = [
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

df_demand_params = pd.DataFrame(data)
print(df_demand_params)


# %%
# Step 2, convert the dataframe to a nested-demand dictionary
# ---------------------------------------------------------
#
# This is just a structural conversion; it does not change the economic content.

dc_demand_ces = cme_inpt_convert.cme_convert_pd2dc(
    df_demand_params, input_type="demand", verbose=False
)
print(dc_demand_ces)


# %%
# Step 3, roll aggregate wages upward (``st_solve_type='wge'``)
# ---------------------------------------------------------
# This evaluates the same CES marginal-cost formula described above at each nest:
# node 5 gives w_1, node 6 gives w_2, and node 7 gives w_Y. We pass a copy so later
# steps can reuse the original dataframe.

_dc_wge = copy.deepcopy(dc_demand_ces)
_dc_wge = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
    _dc_wge, st_solve_type="wge", verbose=False, verbose_debug=False
)
print("w_1 (node 5, nest over x_{1,1}, x_{1,2}):", _dc_wge[5]["wge"])
print("w_2 (node 6, nest over x_{2,1}, x_{2,2}):", _dc_wge[6]["wge"])
print("w_Y (node 7, root over Y_1 and Y_2):", _dc_wge[7]["wge"])


# %%
# Step 4, solve the primal reference quantities from the nested solver
# ---------------------------------------------------------
# Using the same tree and ``fl_Q_agg=1``, the nested solver gives a reference set of
# quantities consistent with the wage rollup. We save the canopy quantities so Step 5
# can demonstrate the upward quantity aggregation directly.

_dc_ref = cme_inpt_convert.cme_convert_pd2dc(
    df_demand_params, input_type="demand", verbose=False
)
_dc_ref = cme_dslv_opti.cme_prod_ces_nested_solver(
    _dc_ref, fl_Q_agg=1.0, verbose=False, verbose_debug=False
)
QTY_CANOPY_1 = _dc_ref[1]["qty"]
QTY_CANOPY_2 = _dc_ref[2]["qty"]
QTY_CANOPY_3 = _dc_ref[3]["qty"]
QTY_CANOPY_4 = _dc_ref[4]["qty"]
print("Reference canopy qty (solver), node 1:", QTY_CANOPY_1)
print("Reference canopy qty (solver), node 2:", QTY_CANOPY_2)
print("Reference canopy qty (solver), node 3:", QTY_CANOPY_3)
print("Reference canopy qty (solver), node 4:", QTY_CANOPY_4)
print("Reference inner qty (solver), node 5:", _dc_ref[5]["qty"])
print("Reference root qty (solver), node 7:", _dc_ref[7]["qty"])


# %%
# Step 5, roll aggregate quantities upward (``st_solve_type='qty'``)
# ---------------------------------------------------------
# Seed only the canopy ``qty`` values from Step 4, then call
# :func:`~prjlecm.demand.cme_dslv_eval.cme_prod_ces_nest_agg_q_p` with
# ``st_solve_type='qty'``. The rolled-up inner and root quantities should match the
# reference quantities from Step 4.

_rows_qty = copy.deepcopy(data)
for row in _rows_qty:
    if row["key_node"] == 1:
        row["qty"] = QTY_CANOPY_1
    elif row["key_node"] == 2:
        row["qty"] = QTY_CANOPY_2
    elif row["key_node"] == 3:
        row["qty"] = QTY_CANOPY_3
    elif row["key_node"] == 4:
        row["qty"] = QTY_CANOPY_4

df_qty_seed = pd.DataFrame(_rows_qty)
_dc_qty = cme_inpt_convert.cme_convert_pd2dc(
    df_qty_seed, input_type="demand", verbose=False
)
_dc_qty = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
    _dc_qty, st_solve_type="qty", verbose=False, verbose_debug=False
)
print("Rolled-up Y_1 at node 5 from canopy only:", _dc_qty[5]["qty"])
print("Rolled-up Y_2 at node 6 from canopy only:", _dc_qty[6]["qty"])
print("Rolled-up Y at node 7 (root):", _dc_qty[7]["qty"])