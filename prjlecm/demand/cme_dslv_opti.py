# Solving for demand side optimal choices.

import pprint

import numpy as np

import prjlecm.demand.cme_dslv_eval as cme_dslv_eval
import prjlecm.input.cme_inpt_parse as cme_inpt_parse


def cme_prod_ces_solver(ar_price, ar_share,
                        fl_elasticity,
                        fl_Q, fl_A, verbose=False):
    """Solves Optimal CES Demand Given CRS CES with N inputs

    This provides the CES optimal expenditure minimizing choices. There
    is a single layer.

    Parameters
    ----------
    ar_price : _type_
        _description_
    ar_share : _type_
        _description_
    fl_elasticity : _type_
        _description_
    fl_Q : _type_
        _description_
    fl_A : _type_
        _description_

    Returns
    -------
    tuple
       fl_mc_aggprice marginal cost or aggregate price.  
    """
    # Price Ratio: divide each price by every other price
    # PRICES x trans(1/PRICE)
    # SHARES x trans(1/SHARES)
    # fl_elasticity = 0.5
    # fl_Q = 1
    # fl_A = 1

    # np.random.seed(8888)
    # ar_price = np.random.uniform(size=3)
    # # ar_price = np.array([1,1,1])
    # # ar_price = np.array([0.6965, 0.2861, 0.2269])

    # ar_share = np.random.uniform(size=3)
    # ar_share = np.array([1,1,1])/3
    # ar_share = np.array([0.3255, 0.4247, 0.2498])
    ar_share = ar_share / sum(ar_share)

    # Each column divides by a different number
    # [a;b;c] x [1/a, 1/b, 1/c] : (3 by 1) x (1 by 3)
    mt_rela_price = np.outer(ar_price, np.reciprocal(ar_price))
    np.reciprocal(mt_rela_price)
    mt_rela_share = np.outer(ar_share, np.reciprocal(ar_share))

    # (p_j/p_i) x (alpha_i/alpha_j)
    mt_rela_prcshr = np.multiply(mt_rela_price, np.reciprocal(mt_rela_share))
    mt_rela_prcshr_rho = mt_rela_prcshr ** (fl_elasticity / (1 - fl_elasticity))

    # alpha_i x [(p_j/p_i) x (alpha_i/alpha_j)]^(rho/(1-rho))
    # 1/p_i are columns
    # (N by N) x (N by 1)
    ar_sum = np.matmul(mt_rela_prcshr_rho, ar_share)
    ar_sum_rho = ar_sum ** (-1 / fl_elasticity)

    # rescale
    if fl_Q is not None:
        ar_opti_costmin_x = ar_sum_rho * fl_Q / fl_A
    else:
        ar_opti_costmin_x = None
    # ar_opti_costmin_x

    # weighted average of prices
    fl_mc_aggprice = (np.dot(ar_sum_rho, ar_price) / fl_A)
    # fl_mc_aggprice

    if verbose:
        print('d-333574 ces single-layer optimal choices and aggregate price:')
        print(f'{ar_opti_costmin_x=}')
        print(f'{fl_mc_aggprice=}')

    return ar_opti_costmin_x, fl_mc_aggprice


def cme_prod_ces_nested_solver(dc_ces_flat, fl_Q_agg=None, verbose=False, verbose_debug=False):
    """Solves optimal nested-CES demands with arbitrary number of nests and layers 

    Given prices (wages), solve for optimal choices (optimal labor demand quantities), for expenditure
    minimization problems in nested-CES setting. While the equilibrium solution program assumes
    homogeneous elasticity of substitution within the lowest layer of nesting, this functions
    allows for any elasticity of substitution at any layer and in any nest. 

    Assume wage at the lowest layer
    1. Aggregated wages up via
    prjlecm.demand.cme_dslv_eval.cme_prod_ces_nest_agg_q_p(st_solve_type='wge')
    Also assume the 'qty' value at the top layer is available (overall y)
    but not at any of the lower layers.
    2. Find all layers, iterate from the top to the bottom
    3. At each layer, for each keys at the layer, find keys following the node in the layer below
    4. Solve sub-tree qty given the qty of the top layer, and the share and wge at all sub-tree

    Parameters
    ----------
    dc_ces_flat : _type_
        Flat nested-ces dictionary, assumed to have 'wge' (price) values at the lowest
        level of nest. This is to solve optimal choices given prices
    fl_Q_agg : _type_, optional
        This is the aggregate output quantity, if it is not None, then this will
        replace what iin the "top" layer's single key's value's qty field, if this 
        is not provided, this would already be in there. An aggregate quantity output
        is required by expenditure minimization problems, by default None
    verbose : bool, optional
        _description_, by default False
    verbose_debug : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    # 1. Aggregate prices up
    # output dc_ces_flat is the same dict as the input, but with updated info
    st_solve_type = 'wge'
    dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
        dc_ces_flat, st_solve_type=st_solve_type, verbose=True, verbose_debug=True)

    # 2. Find all layers, iterate from the top to the bottom
    it_max_layer, __, it_lyr0_key = cme_inpt_parse.cme_parse_demand_tbidx(
        dc_ces_flat)
    if verbose_debug:
        print(f'{it_max_layer=}')
    if (fl_Q_agg is not None):
        # assign aggregate Q value to top layer
        dc_ces_flat[it_lyr0_key]['qty'] = fl_Q_agg

    for it_layer in np.arange(it_max_layer):
        if verbose_debug:
            print(f'{it_layer=}')

        # 3. At each layer, for each keys at the layer, find keys following the node in the layer below
        ls_it_prt_keys = [prt_key for prt_key in dc_ces_flat
                          if dc_ces_flat[prt_key]['lyr'] == (it_layer)]
        if verbose_debug:
            print(f'{ls_it_prt_keys=}')

        for it_prt_key in ls_it_prt_keys:
            dc_parent = dc_ces_flat[it_prt_key]

            fl_prt_qty = dc_parent['qty']
            fl_pwr = dc_parent['pwr']
            ar_it_ipt = dc_parent['ipt']

            ar_shr = np.array([])
            ar_wge = np.array([])

            for it_chd_key in ar_it_ipt:
                dc_child = dc_ces_flat[it_chd_key]
                ar_shr = np.append(ar_shr, dc_child['shr'])
                ar_wge = np.append(ar_wge, dc_child['wge'])

            if verbose_debug:
                print(f'{it_prt_key=} and {ar_it_ipt=}')
                print(f'{fl_pwr=} and {fl_prt_qty=}')
                print(f'{ar_shr=}')
                print(f'{ar_wge=}')

            # 3. Solve sub-tree qty given the qty of the top layer, and the share and wge at all sub-tree
            ar_opti_costmin_x, __ = cme_prod_ces_solver(ar_price=ar_wge, ar_share=ar_shr,
                                                        fl_elasticity=fl_pwr,
                                                        fl_Q=fl_prt_qty, fl_A=1, verbose=verbose_debug)
            for idx, it_chd_key in enumerate(ar_it_ipt):
                dc_ces_flat[it_chd_key]['qty'] = ar_opti_costmin_x[idx]

    # Print
    if verbose:
        print('d-333574 ces nested optimal choices:')
        pprint.pprint(dc_ces_flat, width=2)
        # # check on aggregate output quantity
        # st_solve_type = 'qty'
        # dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
        #     dc_ces_flat, st_solve_type=st_solve_type, verbose=False, verbose_debug=False)
        # fl_Q_agg_fromQsolu = dc_ces_flat[it_lyr0_key]['qty']
        # print(f'{fl_Q_agg_fromQsolu=} and {fl_Q_agg_fromQsolu - fl_Q_agg=}')

    return dc_ces_flat


if __name__ == "__main__":

    it_test_group = 2

    if it_test_group == 1:
        verbose = True

        # it_test == 1: for non-nested optimal solutions tests
        # Test function
        fl_elasticity = 0.5
        fl_Q = 1
        fl_A = 1

        # Test 1 fixed inputs
        ar_price = np.array([0.6965, 0.2861, 0.2269])
        ar_share = np.array([0.3255, 0.4247, 0.2498])
        ar_opti_costmin_x, fl_mc_aggprice = cme_prod_ces_solver(ar_price, ar_share,
                                                                fl_elasticity,
                                                                fl_Q, fl_A, verbose=verbose)
        # Test 2 common price
        ar_price_common = np.array([1, 1, 1])
        cme_prod_ces_solver(ar_price_common, ar_share,
                            fl_elasticity,
                            fl_Q, fl_A, verbose=verbose)

        # Test 3 common share
        ar_share_common = np.array([1, 1, 1]) / 3
        cme_prod_ces_solver(ar_price, ar_share_common,
                            fl_elasticity,
                            fl_Q, fl_A, verbose=verbose)

        # Test 4 fixed price and shares
        np.random.seed(8888)
        ar_price_rand = np.random.uniform(size=3)
        ar_share_rand = np.random.uniform(size=3)
        cme_prod_ces_solver(ar_price_rand, ar_share_rand,
                            fl_elasticity,
                            fl_Q, fl_A, verbose=verbose)

        # Test 5, fl_Q is None
        cme_prod_ces_solver(ar_price_rand, ar_share_rand,
                            fl_elasticity,
                            fl_Q=None, fl_A=1, verbose=verbose)


        # relative price function
        def ffi_ces_rela_opti(ar_inputs_x, ar_share,
                              fl_elasticity, it_idx_base=0):
            # relative prices, with it_index_price_1 price normalized to 1.
            ar_rela_prices = np.multiply((ar_share * 1 / ar_share[it_idx_base]),
                                         (ar_inputs_x * 1 / ar_inputs_x[it_idx_base]) ** (fl_elasticity - 1))

            return (ar_rela_prices)


        # Test
        # Input optimal quantities
        fl_elasticity = 0.5
        fl_Q = 1
        fl_A = 1
        ar_price = np.array([0.6965, 0.2861, 0.2269])
        ar_share = np.array([0.3255, 0.4247, 0.2498])
        ar_opti_costmin_x, fl_mc_aggprice = cme_prod_ces_solver(ar_price, ar_share,
                                                                fl_elasticity,
                                                                fl_Q, fl_A, verbose=verbose)
        # Solve for relative prices given demand optimality
        it_index_price_1 = 0
        ar_rela_prices = ffi_ces_rela_opti(ar_opti_costmin_x, ar_share, fl_elasticity,
                                           it_idx_base=it_index_price_1)
        print(f'{ar_price / ar_price[it_index_price_1]=}')

    elif it_test_group == 2:
        # For nested tests:
        # Assume wage at the lowest layer
        # 1. Aggregated wages up via
        # prjlecm.demand.cme_dslv_eval.cme_prod_ces_nest_agg_q_p(st_solve_type='wge')
        # Also assume the 'qty' value at the top layer is available (overall y)
        # but not at any of the lower layers.
        # 2. Find all layers, iterate from the top to the bottom
        # 3. At each layer, for each keys at the layer, find keys following the node in the layer below
        # 4. Solve sub-tree qty given the qty of the top layer, and the share and wge at all sub-tree

        import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand

        # Simulate with Wages at the bottom layer
        verbose = True
        verbose_debug = True
        bl_simu_q = False
        bl_simu_p = True
        fl_Q_agg = 120.399
        dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
            ar_it_chd_tre=[2, 2, 2], ar_it_occ_lyr=[2],
            fl_power_min=-0.5,
            fl_power_max=0.5,
            it_seed=222,
            bl_simu_q=bl_simu_q,
            bl_simu_p=bl_simu_p,
            verbose=True, verbose_debug=False)
        dc_ces_flat = dc_dc_ces_nested['dc_ces_flat']
        dc_ces_flat = cme_prod_ces_nested_solver(dc_ces_flat, fl_Q_agg=fl_Q_agg,
                                                 verbose=verbose, verbose_debug=verbose_debug)

        # Print
        if verbose:
            print('d-333574 ces single-layer optimal choices and aggregate price:')
            pprint.pprint(dc_ces_flat, width=2)
            # check on aggregate output quantity
            st_solve_type = 'qty'
            dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_agg_q_p(
                dc_ces_flat, st_solve_type=st_solve_type, verbose=False, verbose_debug=False)
            __, __, it_lyr0_key = cme_inpt_parse.cme_parse_demand_tbidx(
                dc_ces_flat)
            fl_Q_agg_fromQsolu = dc_ces_flat[it_lyr0_key]['qty']
            print(f'{fl_Q_agg_fromQsolu=} and {fl_Q_agg_fromQsolu - fl_Q_agg=}')
