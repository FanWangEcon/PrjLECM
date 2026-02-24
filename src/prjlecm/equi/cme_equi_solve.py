import pprint

import numpy as np
import pandas as pd

import prjlecm.equa.cme_supt_equa_demand as cme_supt_equa_demand
import prjlecm.equa.cme_supt_equa_supply as cme_supt_equa_supply
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply
import prjlecm.input.cme_inpt_parse as cme_inpt_parse
import prjlecm.util.cme_supt_math as cme_supt_math
import prjlecm.util.cme_supt_opti as cme_supt_opti
import prjlecm.demand.cme_dslv_eval as cme_dslv_eval

import prjlecm.util.cme_supt_misc as cme_supt_misc


# Get Sets of Linear lines and solve
def cme_equi_solve_sone(dc_sprl_intr_slpe, dc_dmrl_intr_slpe, verbose=False):
    """Solve for relative equilibrium relative prices and quantities

    Solve for relative equilibrium prices and quantities across occupations within worker types.
    This function finds the intersection of supply and demand curves in log-linear form
    to determine equilibrium relative wages and labor quantities. The solution applies
    to a I worker types (I>= 1) across multiple occupations (J>=2), with results expressed
    relative to the base occupation (j=1).

    The equilibrium is derived by solving the system of two linear equations:

        ln(y) = a + b*ln(x)  [Relative supply curve]
        ln(y) = c + d*ln(x)  [Relative demand curve]

    which yields:

        ln(x) = (a - c) / (d - b)
        ln(y) = a + b*ln(x)

    Parameters
    ----------
    dc_sprl_intr_slpe : dict
        Dictionary containing supply curve parameters:

        - 'mt_sprl_wthn_i_acrs_jv1_intr': I x (J-1) matrix of supply intercepts
          (equals α_{i,j} - α_{i,1})
        - 'mt_sprl_wthn_i_acrs_jv1_slpe': I x (J-1) matrix of supply slopes (equals β)

    dc_dmrl_intr_slpe : dict
        Dictionary containing demand curve parameters:

        - 'mt_dmrl_wthn_i_acrs_jv1_intr': I x (J-1) matrix of demand intercepts
          (equals (1/(1-ψ)) * ln(θ_{i,j}/θ_{i,1}))
        - 'mt_dmrl_wthn_i_acrs_jv1_slpe': I x (J-1) matrix of demand slopes
          (equals -1/(1-ψ))

    verbose : bool, optional
        If True, prints equilibrium results to console. Default is False.

    Returns
    -------
    dict
        Dictionary containing equilibrium outcomes:
        - 'mt_eqrl_wage_wthn_i_acrs_jv1': I x (J-1) matrix of relative equilibrium wages
          (ln(W_{i,j}/W_{i,1}))
        - 'mt_eqrl_qnty_wthn_i_acrs_jv1': I x (J-1) matrix of relative equilibrium quantities
          (ln(L_{i,j}/L_{i,1}))

    Notes
    -----
    This implements Step 1 of the equilibrium solution process documented in:
    https://github.com/FanWangEcon/PrjLECM/issues/4
    The relative approach eliminates occupation 1 as the base and solves for all
    relative prices and quantities with respect to this base occupation.
    """

    # Given the following "intr" = intercept and "slpe" = slope parameters
    # for "sprl" = supply relative and "dmrl" = demand relative functions
    # We solve for the equilibrium relative prices and quantities
    #
    # These correspond to what is written in Part 1 of https://github.com/FanWangEcon/PrjLECM/issues/4:
    #
    # - Supply intercept: :math:`\alpha_{i,j} - \alpha_{i,1}`
    # - Supply slope: :math:`\beta`
    # - Demand intercept: :math:`\left(\frac{1}{1-\psi}\cdot \ln \left( \frac{\theta_{i,j}}{\theta_{i,1}}\right)\right)`
    # - Demand slope: :math:`-\frac{1}{1-\psi}`
    #
    # There is one worker type (I=1) and four occupations (J=4). There are
    # 3 columns, because occupation 1 is the base occupation, and we are solving
    # for relative prices (wages) and quantities for occupations 2, 3, and 4 relative to
    # occupation 1.
    #
    # In calling `cme_equi_solve.cme_equi_solve_sone()`, we generate two outputs:
    # - `mt_eqrl_wage_wthn_i_acrs_jv1`: relative equilibrium wages, :math:`\ln\left(\frac{W_{i,j}}{W_{i,1}})\right)`
    # - `mt_eqrl_qnty_wthn_i_acrs_jv1`: relative equilibrium quantities, :math:`\ln\left(\frac{L_{i,j}}{L_{i,1}}\right)`

    # Step 1 of https://github.com/FanWangEcon/PrjLECM/issues/4

    # Loop over individuals
    # We have two equations:
    # ln(y) = a + b ln(x)
    # ln(y) = c + d ln(x)
    # which becomes
    # a + b ln(x) = c + d ln(x)
    # which means
    # ln(x) = (a - c)/(d - b)
    # and we also have
    # ln(y) = a + b ln(x)
    mt_sprl_wthn_i_acrs_jv1_intr = dc_sprl_intr_slpe["mt_sprl_wthn_i_acrs_jv1_intr"]
    mt_sprl_wthn_i_acrs_jv1_slpe = dc_sprl_intr_slpe["mt_sprl_wthn_i_acrs_jv1_slpe"]

    mt_dmrl_wthn_i_acrs_jv1_intr = dc_dmrl_intr_slpe["mt_dmrl_wthn_i_acrs_jv1_intr"]
    mt_dmrl_wthn_i_acrs_jv1_slpe = dc_dmrl_intr_slpe["mt_dmrl_wthn_i_acrs_jv1_slpe"]

    it_wkr_cnt, it_occ_cnt_minus_1 = np.shape(mt_sprl_wthn_i_acrs_jv1_slpe)
    # mt_eqrl_wage_wthn_i_acrs_jv1 : a I x (J-1) matrix of optimal relative wages, within i across j, compare j>1 to j=1
    mt_eqrl_wage_wthn_i_acrs_jv1 = np.empty(
        [it_wkr_cnt, it_occ_cnt_minus_1], dtype=float
    )
    # mt_eqrl_qnty_wthn_i_acrs_jv1 : a I x (J-1) matrix of optimal relative wages, within i across j, compare j>1 to j=1
    mt_eqrl_qnty_wthn_i_acrs_jv1 = np.empty(
        [it_wkr_cnt, it_occ_cnt_minus_1], dtype=float
    )

    for it_wkr_ctr in np.arange(it_wkr_cnt):
        for it_occ_ctr in np.arange(it_occ_cnt_minus_1):
            a = mt_sprl_wthn_i_acrs_jv1_intr[it_wkr_ctr, it_occ_ctr]
            b = mt_sprl_wthn_i_acrs_jv1_slpe[it_wkr_ctr, it_occ_ctr]

            c = mt_dmrl_wthn_i_acrs_jv1_intr[it_wkr_ctr, it_occ_ctr]
            d = mt_dmrl_wthn_i_acrs_jv1_slpe[it_wkr_ctr, it_occ_ctr]

            fl_ln_of_x, fl_ln_of_y = cme_supt_math.cme_math_lin2n(a, b, c, d)
            # fl_ln_of_x = (a-c)/(d-b)
            # fl_ln_of_y = a + b * fl_ln_of_x

            mt_eqrl_wage_wthn_i_acrs_jv1[it_wkr_ctr, it_occ_ctr] = fl_ln_of_x
            mt_eqrl_qnty_wthn_i_acrs_jv1[it_wkr_ctr, it_occ_ctr] = fl_ln_of_y

    dc_equi_solve_sone = {
        "mt_eqrl_wage_wthn_i_acrs_jv1": mt_eqrl_wage_wthn_i_acrs_jv1,
        "mt_eqrl_qnty_wthn_i_acrs_jv1": mt_eqrl_qnty_wthn_i_acrs_jv1,
    }

    if verbose:
        for st_key, dc_val in dc_equi_solve_sone.items():
            print("d-30900 key:" + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_equi_solve_sone


def cme_equi_solve_stwo(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_equi_solve_sone,
    fl_spsh_j0_i1=0.1,
    verbose=False,
):
    """
    Solve for equilibrium wage and labor quantity levels in Step 2 of CME equilibrium.
    This function takes relative equilibrium quantities and prices from Step 1 and
    converts them to absolute levels by determining the equilibrium for occupation 1
    and worker type 1. It uses the share of type 1 individuals choosing leisure and
    total population to scale up from relative to absolute values.

    Parameters
    ----------
    dc_sprl_intr_slpe : dict
        Dictionary containing supply parameters:

        - 'ar_sprl_wthn_i_acrs_j1v0_intr': array of supply intercepts for each
            worker type, shape (I,), represents :math:`\alpha_{i,1}`
        - 'ar_sprl_wthn_i_acrs_j1v0_slpe': array of supply slopes (wage coefficients)
            for each worker type, shape (I,), represents :math:`\beta`

    ar_splv_totl_acrs_i : array
        Total population count for each worker type i, shape (I,), represents :math:`\hat{L}_{i}`
    dc_equi_solve_sone : dict
        Dictionary containing Step 1 equilibrium outputs:

        - 'mt_eqrl_qnty_wthn_i_acrs_jv1': log relative quantities of occupation j
            vs occupation 1 for each worker type i, shape (I, J-1),
            represents :math:`\ln\left(\frac{L_{i,j}}{L_{i,1}}\right)`

    fl_spsh_j0_i1 : float, optional
        Share of type 1 worker population choosing leisure (occupation 0),
        default is 0.1, represents :math:`v_i`
    verbose : bool, optional
        Flag for verbose output (currently unused), default is False

    Returns
    -------
    dict
        Dictionary containing equilibrium values:

        - 'fl_splv_i1_j1': equilibrium level of workers of type 1 in occupation 1,
            :math:`L_{1,1}(v_1)`
        - 'fl_wglv_i1_j1': equilibrium wage level for type 1 workers in occupation 1,
            :math:`W_{1,1}`
        - 'ar_spsh_j1_of_totw': share of occupation 1 among all working individuals
            of type 1, :math:`\frac{L_{1,1}}{\sum_{j=1}^{J}L_{1,j}}`, shape (I,)

    Notes
    -----
    The function operates in two stages:
    1. Calculates the share of occupation 1 among working individuals using relative
        quantities from Step 1
    2. Uses this share along with leisure choice and total population to solve for
        absolute equilibrium wage and quantity levels
    """

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
    # Note that we can do this concurrently for all I individual types.
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
    # - Supply intercept, `ar_sprl_wthn_i_acrs_j1v0_intr`: :math:`\alpha_{i,1}`
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
    # In calling `cme_equi_solve.cme_equi_solve_stwo()`, we generate
    # equilibrium quantity and wage levels condition on unknown `v_1`:
    #
    # - `fl_splv_i1_j1`: equilibrium level for occupation 1 of worker type 1, :math:`L_{1,1}(v_1)`
    # - `fl_wglv_i1_j1`: relative equilibrium quantities, :math:`\ln\left(\frac{L_{i,j}}{L_{i,1}}\right)`
    # - `ar_spsh_j1_of_totw`: share of occupation 1 among all working individuals of type 1, :math:`\frac{L_{1,1}}{\sum_{j=1}^{J}L_{1,j}}`. Note that `ar_spsh_j1_of_totw` really should have been found in step one, it is a function of the relative labor quantities among working population of each individual type.
    #
    # Expected output:
    # {
    #     'fl_splv_i1_j1': 1.3937921476703075,
    #     'fl_wglv_i1_j1': 47.590328650689194,
    #     'ar_spsh_j1_of_totw':
    #         np.array([0.65])
    # }

    # Get matrixes from supply
    # two things below should be identical
    # mt_sprl_intr = dc_sprl_intr_slpe["mt_sprl_intr"]
    # ar_sprl_slpe = dc_sprl_intr_slpe["ar_sprl_slpe"]

    ar_sprl_wthn_i_acrs_j1v0_intr = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_intr"]
    ar_sprl_wthn_i_acrs_j1v0_slpe = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_slpe"]

    # Non-leisure category j=1 share
    # mt_eqrl_wage_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_wage_wthn_i_acrs_jv1"]
    mt_eqrl_qnty_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_qnty_wthn_i_acrs_jv1"]
    # + 1 below is because summation is from j=1 to J, one of which is L_{i, 1}/L_{i,1}, but that is not in the matrix
    ar_sum_lab_summ_j1_unit = np.sum(np.exp(mt_eqrl_qnty_wthn_i_acrs_jv1), 1) + 1
    # totw = total work
    # array over i types of workers.
    ar_spsh_j1_of_totw = 1 / ar_sum_lab_summ_j1_unit

    # Solve for L_{1,1}
    fl_splv_i1_j1, _ = cme_supt_equa_supply.cme_agg_splv_j1(
        ar_splv_totl_acrs_i[0], fl_spsh_j0_i1, ar_spsh_j1_of_totw[0]
    )

    # Solve for W_{1,1}
    fl_wglv_i1_j1_p1 = ((1 - fl_spsh_j0_i1) * (1 / fl_spsh_j0_i1)) * ar_spsh_j1_of_totw[
        0
    ]
    # fl_wglv_i1_j1_p2 = np.log(fl_wglv_i1_j1_p1) - mt_sprl_intr[0, 0]
    fl_wglv_i1_j1_p2 = np.log(fl_wglv_i1_j1_p1) - ar_sprl_wthn_i_acrs_j1v0_intr[0]
    # fl_wglv_i1_j1 = np.exp(fl_wglv_i1_j1_p2*(1/ar_sprl_slpe[0]))
    fl_wglv_i1_j1 = np.exp(fl_wglv_i1_j1_p2 * (1 / ar_sprl_wthn_i_acrs_j1v0_slpe[0]))

    return {
        "fl_splv_i1_j1": fl_splv_i1_j1,
        "fl_wglv_i1_j1": fl_wglv_i1_j1,
        "ar_spsh_j1_of_totw": ar_spsh_j1_of_totw,
    }


def cme_equi_solve_sthr(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_stwo,
    fl_spsh_j0_i1=0.1,
    verbose=False,
):
    """
    Solve for equilibrium leisure shares across worker types in occupation 1.
    This is Step 3 of the equilibrium solution procedure. Given the equilibrium
    quantities and relative wages within and across occupations from Steps 1-2,
    solves for the leisure share (nu) for each worker type i=2,3,...,I, conditional
    on the leisure share for worker type 1 (which was determined in Step 2).
    The function sets up equilibrium conditions by combining supply-side and demand-side
    equations. For each worker type i>1, it computes the coefficients (lambda, omega,
    and coef_ln1misnui) that characterize the equilibrium condition, then uses bisection
    to solve for the leisure share that satisfies the equilibrium.

    Parameters
    ----------
    dc_sprl_intr_slpe : dict
        Supply-side parameters containing:
        - ar_sprl_wthn_i_acrs_j1v0_intr: intercept alpha_{i,1} for each worker type i
        - ar_sprl_wthn_i_acrs_j1v0_slpe: slope beta_i for each worker type i
    ar_splv_totl_acrs_i : array_like
        Total labor supply across all occupations for each worker type i
    dc_dmrl_intr_slpe : dict
        Demand-side parameters containing:
        - ar_dmrl_acrs_iv1_cnd_j1_intr: demand intercept ln(theta_{i,1}/theta_{1,1})
        - ar_dmrl_acrs_iv1_cnd_j1_slpe: demand slope (psi-1)
    dc_equi_solve_stwo : dict
        Results from Step 2 containing:
        - fl_splv_i1_j1: labor quantity L_{1,1}(nu_1) for worker type 1 in occupation 1
        - fl_wglv_i1_j1: wage W_{1,1}(nu_1) for worker type 1 in occupation 1
        - ar_spsh_j1_of_totw: share of worker type i in occupation 1 relative to total
    fl_spsh_j0_i1 : float, optional
        Leisure share for worker type 1 (exogenously specified). Default is 0.1
    verbose : bool, optional
        If True, print solution details. Default is False

    Returns
    -------
    dict
        Dictionary containing:
        - ar_nu_solved : ndarray of shape (I,)
            Equilibrium leisure share nu_i for each worker type i
        - pd_nu_solu : pd.DataFrame
            Dataframe with columns: wkr_i, nu_i, coef_ln1misnui_i, omega_i, lambda_i
            containing detailed solution results for each worker type

    Notes
    -----
    The leisure share nu_i represents the proportion of worker type i choosing leisure
    over labor market participation in occupation 1. The function solves I-1 equilibrium
    conditions (one for each worker type i=2,...,I) using bisection method.
    """

    # These correspond to what is written in Part 3 of https://github.com/FanWangEcon/PrjLECM/issues/4:
    #
    # In Step 1, we solve for relative equilibrium quantitles and wages within
    # individual type i, of each occupation j vs occupation 1.
    #
    # In Step 2, we focus on individual type 1. We make a guess for share of
    # individual of type 1 who choose leisure, and find the number of individual of
    # type 1 participating in occupation type 1 given this, and then find the level
    # of wage for occupation type 1 and individual type 1.
    #
    # In Step 3, the current step, we now use the relatively quantity and wage
    # relations across individuals for the same occupation j=1. All ratios are
    # against individual 1's wages and quantity, which we found in Step 2. So the
    # unknown are no longer relative wages and quantites, but levels of wages and
    # quantities in occupation j. We arrive at $I-1$ set of equilibrium conditions
    # for $I-1$ equilibrium wage levels for $I-1$ individual types, all in
    # occupation j=1.
    #
    # We provide as input the share of worker type 1 choosing leisure, and solve for
    # the the share of workers of type 2, 3, ..., I choosing leisure. Each one of
    # these $I-1$ result is solved via bisetion from
    # :function:`prjlecm.equi.cme_supt_opti.cme_opti_nu1_bisect()`.  When we know
    # the share of leisure for all individuals, given the relative equilibrium
    # quantities we had solved, we have the equilibrium quantities of workers for
    # all $I$ and all $J$. And we also have, via supply-side equations with relative
    # log quantity of worker in occupation j>1 with respect to individuals in j=1,
    # the wage levels for all $I$ and all $J$.
    #
    # Inputs here not used in step one or two, random values for these could be generated
    # by `cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest()`:
    #
    # - Demand intercept, `ar_dmrl_acrs_iv1_cnd_j1_intr` =
    # :math:`\frac{1}{1-\psi}\ln\left(\frac{\theta_{i,1}}{\theta_{1,1}}\right)`, 1
    # by (I-1) array, across $I$, each versus $i=1$, conditional on $j=1$. When
    # $I=1$, this is `np.array([])`.
    # - Demand slope, `ar_dmrl_acrs_iv1_cnd_j1_slpe` = :math:`\frac{1}{1-\psi}`, 1
    # by (I-1) array, can be I-1 specific, but in practice is same for all $i$. When
    # $I=1$, this is `np.array([])`.
    #
    # In calling `cme_equi_solve.cme_equi_solve_stwo()`, we generate two two putputs:
    #
    # - `ar_nu_solved`: the equilibrium value of nu for each individual type i,
    # :math:`\nu_{i}`, but when I=1, this is just our input value `fl_spsh_j0_i1`
    # - `pd_nu_solu`: a dataframe of solution details for each individual type i
    #
    # Expected output example when I=1:
    # {
    #     'ar_nu_solved':
    #         np.array([0.1]),
    #     'pd_nu_solu':    wkr_i  nu_i  coef_ln1misnui_i  omega_i  lambda_i
    #               0      1   0.1               NaN      NaN       NaN
    # }
    #
    # Expected Output When I=3:
    # {
    #     'ar_nu_solved':
    #         np.array([0.1 , 0.02, 0.08]),
    #     'pd_nu_solu':    wkr_i      nu_i  coef_ln1misnui_i   omega_i  lambda_i
    # 0      1  0.100000               NaN       NaN       NaN
    # 1      2  0.017654          1.244278  4.014653  1.721928
    # 2      3  0.080099          1.284408  2.417268  0.135908
    # }

    # Step 3 of https://github.com/FanWangEcon/PrjLECM/issues/4
    # sthr = step three function

    # Supply inputs
    ar_sprl_wthn_i_acrs_j1v0_intr = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_intr"]
    ar_sprl_wthn_i_acrs_j1v0_slpe = dc_sprl_intr_slpe["ar_sprl_wthn_i_acrs_j1v0_slpe"]

    # Demand Inputs
    ar_dmrl_acrs_iv1_cnd_j1_intr = dc_dmrl_intr_slpe["ar_dmrl_acrs_iv1_cnd_j1_intr"]
    ar_dmrl_acrs_iv1_cnd_j1_slpe = dc_dmrl_intr_slpe["ar_dmrl_acrs_iv1_cnd_j1_slpe"]

    # Step 2 results inputs
    # splv_i1_j1 = L_{1,1}(nu_1)
    # wglv_i1_j1 = W_{1,1}(nu_1)
    # ar_spsh_j1_of_totw  = \hat{chi}^{\ast}_i
    fl_splv_i1_j1 = dc_equi_solve_stwo["fl_splv_i1_j1"]
    fl_wglv_i1_j1 = dc_equi_solve_stwo["fl_wglv_i1_j1"]
    ar_spsh_j1_of_totw = dc_equi_solve_stwo["ar_spsh_j1_of_totw"]

    it_wkr_cnt = np.shape(ar_sprl_wthn_i_acrs_j1v0_intr)[0]
    ar_it_wkr_i2tI_cnt = np.arange(it_wkr_cnt - 1) + 1
    # Obtain inputs for nonlinear solver to find shares
    # for supply structure: Loop over i from 2 to last, already solved for i = 1
    # demand matrix already I-1, excluding the first
    ar_lambda = np.empty(
        [
            it_wkr_cnt - 1,
        ],
        dtype=float,
    )
    ar_omega = np.empty(
        [
            it_wkr_cnt - 1,
        ],
        dtype=float,
    )
    ar_coef_ln1misnui = np.empty(
        [
            it_wkr_cnt - 1,
        ],
        dtype=float,
    )
    for it_wkr_ctr in ar_it_wkr_i2tI_cnt:
        # intr = alpha_{i,1}, slpe = beta_{i}
        fl_sprl_i_intr = ar_sprl_wthn_i_acrs_j1v0_intr[it_wkr_ctr]
        fl_sprl_i_slpe = ar_sprl_wthn_i_acrs_j1v0_slpe[it_wkr_ctr]

        # Invert demand-side intercept and slope
        fl_dmrl_i_intr = ar_dmrl_acrs_iv1_cnd_j1_intr[it_wkr_ctr - 1]
        fl_dmrl_i_slpe = ar_dmrl_acrs_iv1_cnd_j1_slpe[it_wkr_ctr - 1]
        # intr_inv = ln(theta_{i,1}/theta_{1,1})
        # slpe_inv = (psi-1)
        fl_dmrl_i_intr_inv, fl_dmrl_i_slpe_inv = cme_supt_math.cme_math_lininv(
            fl_dmrl_i_intr, fl_dmrl_i_slpe
        )

        # Lambda: lambda is a part of ln(W_{i,1})
        fl_lambda_p1 = (
            np.log(fl_wglv_i1_j1)
            - fl_dmrl_i_slpe_inv * np.log(fl_splv_i1_j1)
            + fl_dmrl_i_intr_inv
        )
        fl_lambda = fl_sprl_i_intr + fl_sprl_i_slpe * fl_lambda_p1

        # Omega: Omega has all the non nu_i i > 1 components
        fl_omega_p1 = ar_splv_totl_acrs_i[it_wkr_ctr] * ar_spsh_j1_of_totw[it_wkr_ctr]
        fl_omega_p2 = fl_sprl_i_slpe * fl_dmrl_i_slpe_inv * np.log(fl_omega_p1)
        fl_omega = fl_lambda + fl_omega_p2 - np.log(ar_spsh_j1_of_totw[it_wkr_ctr])

        # coefficient (1-beta*(psi-1)) before ln(1-nu_i)
        fl_coef_ln1misnui = 1 - fl_sprl_i_slpe * fl_dmrl_i_slpe_inv

        # store
        ar_lambda[it_wkr_ctr - 1] = fl_lambda
        ar_omega[it_wkr_ctr - 1] = fl_omega
        ar_coef_ln1misnui[it_wkr_ctr - 1] = fl_coef_ln1misnui

    # Apply non-linear solver to find nu_i(nu_1)
    ar_nu_solved = np.empty(
        [
            it_wkr_cnt,
        ],
        dtype=float,
    )
    ar_nu_solved[0] = fl_spsh_j0_i1
    for it_wkr_ctr in ar_it_wkr_i2tI_cnt:
        # store
        fl_omega = ar_omega[it_wkr_ctr - 1]
        fl_coef_ln1misnui = ar_coef_ln1misnui[it_wkr_ctr - 1]

        fl_nu_solved, fl_obj = cme_supt_opti.cme_opti_nu_bisect(
            fl_coef_ln1misnui, fl_omega
        )
        ar_nu_solved[it_wkr_ctr] = fl_nu_solved

    # Results matrix collection
    pd_nu_solu = pd.DataFrame(
        {
            "wkr_i": np.insert(ar_it_wkr_i2tI_cnt, 0, 0) + 1,
            "nu_i": ar_nu_solved,
            "coef_ln1misnui_i": np.insert(ar_coef_ln1misnui, 0, None),
            "omega_i": np.insert(ar_omega, 0, None),
            "lambda_i": np.insert(ar_lambda, 0, None),
        }
    )

    # Print results
    if verbose:
        print(pd_nu_solu.round(decimals=3))

    # return
    return {"ar_nu_solved": ar_nu_solved, "pd_nu_solu": pd_nu_solu}


def cme_equi_solve_sfur(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_equi_solve_stwo,
    dc_equi_solve_sthr,
    dc_ces,
    bl_nested_ces = True,
    verbose=False,
):
    """
    Solve for equilibrium labor quantities and wages across worker types and occupations
    in a multi-layer CES production model, following steps outlined in Part 4 of
    https://github.com/FanWangEcon/PrjLECM/issues/4.
    This function computes equilibrium supply and wage levels for all worker types (i)
    and all occupations (j), given previously solved relative quantities and wages,
    and the share of leisure for each worker type. It supports both nested and non-nested
    CES structures.
    
    Steps:
        1. Compute supply and wage levels for all workers in occupation j=1.
        2. Compute supply and wage levels for all workers in all occupations j>1.
        3. Aggregate results into pandas DataFrames.
        4. Compute total CES output, either via nested CES aggregation or standard CES.
        
    Parameters
    ----------
    dc_sprl_intr_slpe : dict
        Dictionary containing supply intercepts and slopes for each worker type and occupation.
        Keys: 'mt_sprl_intr' (matrix), 'ar_sprl_slpe' (array).
    ar_splv_totl_acrs_i : np.ndarray
        Array of total supply values across worker types.
    dc_dmrl_intr_slpe : dict
        Dictionary containing demand intercepts, slopes, shares, and elasticity.
        Keys: 'mt_dmrl_share', 'fl_elas', etc.
    dc_equi_solve_sone : dict
        Dictionary containing relative equilibrium quantities and wages within worker types
        across occupations (step one results).
    dc_equi_solve_stwo : dict
        Dictionary containing equilibrium share of workers in occupation j=1 (step two results).
        Keys: 'ar_spsh_j1_of_totw'.
    dc_equi_solve_sthr : dict
        Dictionary containing equilibrium leisure shares for all worker types (step three results).
        Keys: 'ar_nu_solved'.
    dc_ces : dict
        Dictionary describing the CES production structure, including layer keys, worker and
        occupation indices, and quantity placeholders.
    bl_nested_ces : bool, optional
        If True, compute output using nested CES aggregation. If False, use standard CES.
        Default is True.
    verbose : bool, optional
        If True, print intermediate results and DataFrames. Default is False.
        
    Returns
    -------
    dict
        Dictionary containing:
            - 'fl_ces_output': float, total CES output.
            - 'pd_qtlv_all': pandas.DataFrame, equilibrium labor quantities for all worker types and occupations.
            - 'pd_wglv_all': pandas.DataFrame, equilibrium wage levels for all worker types and occupations.
            
    Notes
    -----
    - The function assumes that all input dictionaries are properly structured and contain
      necessary keys as described above.
    - For nested CES, the function updates the quantity terms in the CES dictionary and
      computes aggregate output using the nested structure.
    - For non-nested CES, the function computes output using standard CES aggregation.
    - Intermediate results can be printed for debugging by setting `verbose=True`.
    """

    # These correspond to what is written in Part 4 of https://github.com/FanWangEcon/PrjLECM/issues/4:
    #
    # In Step 1, we solve for relative equilibrium quantitles and wages within
    # individual type i, of each occupation j vs occupation 1.
    #
    # In Step 2, we focus on individual type 1. We make a guess for share of
    # individual of type 1 who choose leisure, and find the number of individual of
    # type 1 participating in occupation type 1 given this, and then find the level
    # of wage for occupation type 1 and individual type 1.
    #
    # In Step 3, given the level of individual type 1 and occupation j=1 worker and
    # wage, we find the number of individuals of all types i > 1 working in
    # occupation j=1. This uses within occupation j=1, relative demand optimality
    # conditions across worker i>1 vs i=1.  The solution is based on solving I-1
    # nonlinear equations of each of which is one equation with one unknown.
    #
    # In Step 4, given the number of individuals (and their wage) of all types i
    # working in occupation j=1, since we also have the relative equilibrium
    # quantities and wages from Step (1), we now have the equilibrium quantities and
    # wages for all worker types i and all occupations j.
    #
    # Note that Steps 2, 3, and 4 are all dependent on the share of individuals of
    # type i=1 choosing leisure. After step 4, all equilibrium quantity and wage
    # levels are determined by this single unknown.
    #
    # the current step, we now use the relatively quantity and wage
    # relations across individuals for the same occupation j=1. All ratios are
    # against individual 1's wages and quantity, which we found in Step 2. So the
    # unknown are no longer relative wages and quantites, but levels of wages and
    # quantities in occupation j. We arrive at $I-1$ set of equilibrium conditions
    # for $I-1$ equilibrium wage levels for $I-1$ individual types, all in
    # occupation j=1.
    #
    # We provide as input the share of worker type 1 choosing leisure, and solve for
    # the the share of workers of type 2, 3, ..., I choosing leisure. Each one of
    # these $I-1$ result is solved via bisetion from
    # :function:`prjlecm.equi.cme_supt_opti.cme_opti_nu1_bisect()`.  When we know
    # the share of leisure for all individuals, given the relative equilibrium
    # quantities we had solved, we have the equilibrium quantities of workers for
    # all $I$ and all $J$. And we also have, via supply-side equations with relative
    # log quantity of worker in occupation j>1 with respect to individuals in j=1,
    # the wage levels for all $I$ and all $J$.
    #
    # Inputs here not used in step one or two, random values for these could be generated
    # by `cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest()`:
    #
    # - Demand intercept, `ar_dmrl_acrs_iv1_cnd_j1_intr` =
    # :math:`\frac{1}{1-\psi}\ln\left(\frac{\theta_{i,1}}{\theta_{1,1}}\right)`, 1
    # by (I-1) array, across $I$, each versus $i=1$, conditional on $j=1$. When
    # $I=1$, this is `np.array([])`.
    # - Demand slope, `ar_dmrl_acrs_iv1_cnd_j1_slpe` = :math:`\frac{1}{1-\psi}`, 1
    # by (I-1) array, can be I-1 specific, but in practice is same for all $i$. When
    # $I=1$, this is `np.array([])`.
    #
    # In calling `cme_equi_solve.cme_equi_solve_stwo()`, we generate two two putputs:
    #
    # - `ar_nu_solved`: the equilibrium value of nu for each individual type i,
    # :math:`\nu_{i}`, but when I=1, this is just our input value `fl_spsh_j0_i1`
    # - `pd_nu_solu`: a dataframe of solution details for each individual type i
    #
    # Expected output example when I=1:
    # {
    #     'fl_ces_output': 0.6036272833447756,
    #     'pd_qtlv_all':       j0        j1        j2        j3       j4
    # i1  0.24  1.398727  0.107646  0.071457  0.58217,
    #     'pd_wglv_all':            j1         j2         j3         j4
    # i1  47.936469  90.112749  91.091013  63.812943
    # }

    # Step 4 of https://github.com/FanWangEcon/PrjLECM/issues/4
    # sfur = step four function

    # A. Supply levels, all workers, for occupation j=1
    ar_spsh_j1_of_totw = dc_equi_solve_stwo["ar_spsh_j1_of_totw"]
    ar_nu_solved = dc_equi_solve_sthr["ar_nu_solved"]
    ar_splv_j1, ar_splv_j0 = cme_supt_equa_supply.cme_agg_splv_j1(
        ar_splv_totl_acrs_i, ar_nu_solved, ar_spsh_j1_of_totw
    )

    # B. Wage levels, all workers, for occupation j=1
    mt_sprl_intr = dc_sprl_intr_slpe["mt_sprl_intr"]
    ar_sprl_slpe = dc_sprl_intr_slpe["ar_sprl_slpe"]
    ar_wglv_j1 = cme_supt_equa_supply.cme_hh_lgt_wglv_j1(
        ar_splv_j1, ar_splv_j0, mt_sprl_intr[:, 0], ar_sprl_slpe
    )

    # Print results
    pd_splv_j1 = pd.DataFrame(
        {
            "nu_i": ar_nu_solved,
            "ar_splv_totl_acrs_i": ar_splv_totl_acrs_i,
            "ar_spsh_j1_of_totw": ar_spsh_j1_of_totw,
            "ar_splv_j1": ar_splv_j1,
            "ar_splv_j0": ar_splv_j0,
            "ar_wglv_j1": ar_wglv_j1,
        }
    )

    # Print results
    if verbose:
        print(pd_splv_j1.round(decimals=4))

    # C. Supply levels, all workers, all occupations j>1
    # We found the supply of workers types in j=1, and have `mt_eqrl_qnty_wthn_i_acrs_jv1` contains
    # relative labor quantity for all occupations j>1 vs occupation j=1 for all worker type i.
    mt_eqrl_qnty_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_qnty_wthn_i_acrs_jv1"]
    mt_qtlv_j2tJ = np.transpose(
        np.transpose(np.exp(mt_eqrl_qnty_wthn_i_acrs_jv1)) * ar_splv_j1
    )
    # mt_splv_all is I by J+1 matrix, all labor supply quantities, across all occupations
    mt_qtlv_all = np.column_stack([ar_splv_j0, ar_splv_j1, mt_qtlv_j2tJ])

    # D. Wage levels, all workers, all j
    # We found the wage of workers types in j=1, and have `mt_eqrl_wage_wthn_i_acrs_jv1` contains
    # relative wage for all occupations j>1 vs occupation j=1 for all worker type i.
    mt_eqrl_wage_wthn_i_acrs_jv1 = dc_equi_solve_sone["mt_eqrl_wage_wthn_i_acrs_jv1"]
    mt_wglv_j2tJ = np.transpose(
        np.transpose(np.exp(mt_eqrl_wage_wthn_i_acrs_jv1)) * ar_wglv_j1
    )
    # mt_splv_all is I by J matrix, all labor supply quantities, across all occupations
    mt_wglv_all = np.column_stack([ar_wglv_j1, mt_wglv_j2tJ])

    # E. Store results to dataframe.
    # Outputs as panda
    pd_qtlv_all = pd.DataFrame(
        mt_qtlv_all,
        index=["i" + str(i + 1) for i in np.arange(np.shape(mt_qtlv_all)[0])],
        columns=["j" + str(j) for j in np.arange(np.shape(mt_qtlv_all)[1])],
    )

    pd_wglv_all = pd.DataFrame(
        mt_wglv_all,
        index=["i" + str(i + 1) for i in np.arange(np.shape(mt_wglv_all)[0])],
        columns=["j" + str(j + 1) for j in np.arange(np.shape(mt_wglv_all)[1])],
    )

    if verbose:
        ar_splv_totw = np.sum(mt_qtlv_all, 1)
        pd_qtwg_all = pd.DataFrame(
            {
                "nu_i": ar_nu_solved,
                "ar_splv_totl_acrs_i": ar_splv_totl_acrs_i,
                "ar_spsh_j1_of_totw": ar_spsh_j1_of_totw,
                "ar_splv_totw": ar_splv_totw,
                "ar_splv_j1": ar_splv_j1,
                "ar_wglv_j1": ar_wglv_j1,
            }
        )
        print(pd_qtwg_all.round(decimals=3))
        print(pd_qtlv_all.round(decimals=3))
        print(pd_wglv_all.round(decimals=3))

    # F. Generate total output
    if bl_nested_ces:
        # dc_dmrl_intr_slpe = None
        # Nested solution to find output based on `dc_ces` information
        __, ls_maxlyr_key, it_lyr0_key = cme_inpt_parse.cme_parse_demand_tbidx(dc_ces)
        for it_key_idx in ls_maxlyr_key:
            # Get wkr and occ index for current child
            it_wkr_idx = dc_ces[it_key_idx]["wkr"]
            it_occ_idx = dc_ces[it_key_idx]["occ"]

            # wrk index matches with rows, 1st row is first worker, wkr index 0
            # occ + 1 because first column is leisure quantityt
            fl_qtlv = pd_qtlv_all.iloc[it_wkr_idx, it_occ_idx + 1]
            # Replace highest/bottommost layer's qty terms with equi solution qs.s
            dc_ces[it_key_idx]["qty"] = fl_qtlv
        dc_ces = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
            dc_ces, verbose=False, verbose_debug=False
        )
        fl_ces_output = dc_ces[it_lyr0_key]["qty"]
    else:
        # Get share parameters
        mt_dmrl_share = dc_dmrl_intr_slpe["mt_dmrl_share"]
        ar_dmrl_share_flat = np.ravel(mt_dmrl_share)
        # Get currently solved quantities
        ar_splv_all_flat = np.ravel(mt_qtlv_all[:, 1::])
        fl_elas = dc_dmrl_intr_slpe["fl_elas"]

        # dc_ces = None
        # this is general except for this last spot which does not work with nested problem
        # share inputs needed for building up CES output function.
        fl_ces_output = cme_supt_equa_demand.cme_prod_ces(
            fl_elas, ar_dmrl_share_flat, ar_splv_all_flat
        )
        
    # corrected solution
    if verbose:
        print(f"{fl_ces_output=}")
        # print(f'{fl_ces_output_alt=}')

    return {
        "fl_ces_output": fl_ces_output,
        "pd_qtlv_all": pd_qtlv_all,
        "pd_wglv_all": pd_wglv_all,
    }


def cme_equi_solve_stwthfu(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_ces,
    bl_nested_ces=True,
    fl_spsh_j0_i1=0.1, 
    verbose=False,
):
    """
    Solves steps 2, 3, and 4 of the equilibrium computation process for CME (as described in PrjLECM issue #4).
    
    This function sequentially calls three sub-functions to:
        1. Solve the second step of the equilibrium process.
        2. Solve the third step using results from the second step.
        3. Generate all equilibrium levels using results from previous steps.
        
    Args:
        dc_sprl_intr_slpe (dict): Spiral interest slope parameters.
        ar_splv_totl_acrs_i (array-like): Array of total across indices.
        dc_dmrl_intr_slpe (dict): Demoral interest slope parameters.
        dc_equi_solve_sone (dict): Results from the first step of equilibrium solving.
        dc_ces (dict): CES (Constant Elasticity of Substitution) parameters.
        fl_spsh_j0_i1 (float, optional): Smoothing parameter for steps. Default is 0.1.
        verbose (bool, optional): If True, prints detailed progress information. Default is False.
        
    Returns:
        dict: Results from the fourth step, containing all equilibrium levels.
    """

    # Steps 2, 3 and 4 of https://github.com/FanWangEcon/PrjLECM/issues/4
    # stwthfu = steps 2 3 and 4 together

    # D.2 Solve second step
    dc_equi_solve_stwo = cme_equi_solve_stwo(
        dc_sprl_intr_slpe,
        ar_splv_totl_acrs_i,
        dc_equi_solve_sone,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=verbose,
    )

    # D.3 Solve third step
    dc_equi_solve_sthr = cme_equi_solve_sthr(
        dc_sprl_intr_slpe,
        ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_stwo,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=verbose,
    )

    # D.4 Generate all Levels
    dc_equi_solve_sfur = cme_equi_solve_sfur(
        dc_sprl_intr_slpe,
        ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_sone,
        dc_equi_solve_stwo,
        dc_equi_solve_sthr,
        dc_ces,
        bl_nested_ces = bl_nested_ces,
        verbose=verbose,
    )

    return dc_equi_solve_sfur


def cme_equi_solve(
    dc_sprl_intr_slpe,
    ar_splv_totl_acrs_i,
    dc_dmrl_intr_slpe,
    dc_equi_solve_sone,
    dc_ces,
    bl_nested_ces=True,
    fl_output_target=0.3,
    verbose_slve=False,
    verbose=False,
):

    # step 5 of https://github.com/FanWangEcon/PrjLECM/issues/4
    # Gateway run steps 1, 2, 3 and 4 of 

    def cmei_opti_nu1_bisect_solver(fl_spsh_j0_i1):
        return cme_equi_solve_stwthfu(
            dc_sprl_intr_slpe,
            ar_splv_totl_acrs_i,
            dc_dmrl_intr_slpe,
            dc_equi_solve_sone,
            dc_ces,
            bl_nested_ces=bl_nested_ces,
            fl_spsh_j0_i1=fl_spsh_j0_i1,
            verbose=verbose_slve,
        )

    # minimal leisure time, max output
    fl_nu1_min = 1e-5
    fl_ces_output_max = cmei_opti_nu1_bisect_solver(fl_spsh_j0_i1=fl_nu1_min)[
        "fl_ces_output"
    ]
    if fl_output_target >= fl_ces_output_max:
        fl_nu1_solved = None
        fl_obj = None
        dc_equi_solv_sfur = None
    else:
        fl_nu1_solved, fl_obj, dc_equi_solv_sfur = cme_supt_opti.cme_opti_nu1_bisect(
            cmei_opti_nu1_bisect_solver, fl_output_target, fl_nu1_min=fl_nu1_min
        )

    if verbose:
        print(f"d-257707: {fl_nu1_solved=} and {fl_obj=}")
        if dc_equi_solv_sfur is not None:
            for st_key, dc_val in dc_equi_solv_sfur.items():
                print("d-257707 key:" + str(st_key))
                print(dc_val)

    return fl_nu1_solved, dc_equi_solv_sfur, fl_ces_output_max


if __name__ == "__main__":
    import cme_equi_solve_gen_inputs

    # Parameters set up
    # Simulation parameter
    it_worker_types = 10
    it_occ_types = 40
    # it_seed_supply = 123
    # it_seed_demand = 456

    it_worker_types = 3
    it_occ_types = 4
    it_seed_supply = np.random.randint(1, 100)
    it_seed_demand = np.random.randint(1, 100)
    # it_seed_supply = 59
    # it_seed_demand = 65
    it_seed_supply = 71
    it_seed_demand = 71

    fl_itc_min = -2
    fl_itc_max = 1
    fl_slp_min = 0.4
    fl_slp_max = 0.6

    fl_power_min = 0.20
    fl_power_max = 0.20

    # Print control
    bl_verbose_simu = False
    bl_verbose_prep = False
    bl_verbose_slve = True

    # A. Simulate
    # A.1 Simulate supply parameters
    dc_supply_lgt, ar_splv_totl_acrs_i = (
        cme_inpt_simu_supply.cme_simu_supply_params_lgt(
            it_worker_types=it_worker_types,
            it_occ_types=it_occ_types,
            fl_itc_min=fl_itc_min,
            fl_itc_max=fl_itc_max,
            fl_slp_min=fl_slp_min,
            fl_slp_max=fl_slp_max,
            it_seed=it_seed_supply,
            verbose=bl_verbose_simu,
        )
    )

    # A.2 Simulate demand parameters
    bl_simu_q = True
    dc_demand_ces = cme_inpt_simu_demand.cme_simu_demand_params_ces_single(
        it_worker_types=it_worker_types,
        it_occ_types=it_occ_types,
        fl_power_min=fl_power_min,
        fl_power_max=fl_power_max,
        bl_simu_q=bl_simu_q,
        it_seed=it_seed_demand,
        verbose=bl_verbose_simu,
    )
    # if bl_simu_q:
    #     de_demand_ces = cme_dslv_eval.cme_prod_ces_nest_output(
    #         dc_demand_ces, verbose=True, verbose_debug=True)

    # B. Supply and Demand input structures
    dc_sprl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest(
        dc_supply_lgt, verbose=bl_verbose_prep
    )
    dc_dmrl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_demand_dict_converter_nonest(
        dc_demand_ces, verbose=bl_verbose_prep
    )

    # C. Solve first step
    dc_equi_solve_sone = cme_equi_solve_sone(
        dc_sprl_intr_slpe, dc_dmrl_intr_slpe, verbose=bl_verbose_slve
    )
    cme_supt_misc.print_dict_aligned(dc_equi_solve_sone, "dc_equi_solve_sone")

    # D.1 Solution parameter
    fl_spsh_j0_i1 = 0.1
    # D.2 Solve second step
    dc_equi_solve_stwo = cme_equi_solve_stwo(
        dc_sprl_intr_slpe,
        ar_splv_totl_acrs_i,
        dc_equi_solve_sone,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=bl_verbose_slve,
    )
    cme_supt_misc.print_dict_aligned(dc_equi_solve_stwo, "dc_equi_solve_stwo")

    # D.3 Solve third step
    dc_equi_solve_sthr = cme_equi_solve_sthr(
        dc_sprl_intr_slpe,
        ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_stwo,
        fl_spsh_j0_i1=fl_spsh_j0_i1,
        verbose=bl_verbose_slve,
    )
    cme_supt_misc.print_dict_aligned(dc_equi_solve_sthr, "dc_equi_solve_sthr")

    # D.4 Generate all Levels
    dc_equi_solve_sfur = cme_equi_solve_sfur(
        dc_sprl_intr_slpe,
        ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_sone,
        dc_equi_solve_stwo,
        dc_equi_solve_sthr,
        dc_demand_ces,
        verbose=bl_verbose_slve,
    )
    cme_supt_misc.print_dict_aligned(dc_equi_solve_sfur, "dc_equi_solve_sfur")

    # solve for nu_1
    fl_output_target = 0.3
    fl_nu1_solved, dc_equi_solv_sfur, fl_ces_output_max = cme_equi_solve(
        dc_sprl_intr_slpe,
        ar_splv_totl_acrs_i,
        dc_dmrl_intr_slpe,
        dc_equi_solve_sone,
        dc_demand_ces,
        fl_output_target=fl_output_target,
        verbose_slve=False,
        verbose=True,
    )

    print(f"{it_seed_supply=}, {it_seed_demand=}")

# %%
