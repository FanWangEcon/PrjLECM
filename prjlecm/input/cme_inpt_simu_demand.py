import ast
import pprint

import numpy as np
import pandas as pd


def cme_simu_demand_ces_inner_dict(
        it_lyr=None, it_prt=None,
        it_wkr=None, it_occ=None,
        fl_shr=None, fl_pwr=None,
        ar_it_ipt=None,
        fl_qty=None, fl_wge=None,
        fl_drv=None, fl_drc=None,
        fl_sch=None, fl_sni=None):
    # if (it_wkr is not None):
    #     it_wkr = it_wkr + 100
    # if (it_occ is not None):
    #     it_occ = it_occ + 100

    # lyr = layer
    # prt = parent
    # drv = derivative
    # drc = cumulative-derivative

    dc_type_return = {
        'lyr': int, 'prt': int,
        'wkr': int, 'occ': int,
        'shr': float, 'pwr': float,
        'ipt': ast.literal_eval,
        'qty': float, 'wge': float,
        'drv': float, 'drc': float,
        'shc': float, 'sni': float
    }

    dc_val_return = {
        'lyr': it_lyr, 'prt': it_prt,
        'wkr': it_wkr, 'occ': it_occ,
        'shr': fl_shr, 'pwr': fl_pwr,
        'ipt': ar_it_ipt,
        'qty': fl_qty, 'wge': fl_wge,
        'drv': fl_drv, 'drc': fl_drc,
        'shc': fl_sch, 'sni': fl_sni
    }

    return dc_val_return, dc_type_return


def cme_simu_demand_mnest_wkr_occ_keys(ar_it_chd_tre=[2, 2, 2, 2], ar_it_occ_lyr=[2, 3], verbose=False):
    """Multi-nest Occ and Wkr Key Generator

    Assume symmetric nesting. There are some M layers of nesting. 
    N_m is the number of children at each layer. And we have
    A subset of these layers that identical occupational groups, with the
    remaining subset of layers identifying worker types. We want to generate
    unique occ and wrk keys based on these meshed-nests.

    Parameters
    ----------
    ar_it_chd_tre : list, optional
        A list of M element for the M layers of nesting, each number corresponds to N_m, NOte if M is 2, then the root layer 
        is 0, the first layer is 1, and the second layer is 2, the list here includes [N_1, N_2], by default [2,2,2,2]
    ar_it_occ_lyr : list, optional
        A list of less than M elements, including which of the M layers is for occupations suppose
        the second and third layer are then [2,3]. by default [2,3]
    verbose : bool, optional
        print details, by default False        
    """

    # ls_chd_uqe_idx_all : list of child unique index all
    ls_chd_uqe_idx_all = [np.arange(it_chd_tre)
                          for it_chd_tre in ar_it_chd_tre]
    mt_occ_wkr_mesh = np.array(np.meshgrid(
        *ls_chd_uqe_idx_all)).T.reshape(-1, len(ar_it_chd_tre))

    # Resort columns, from last to first, jointly
    ls_all_cols_last2first = [
        mt_occ_wkr_mesh[:,it_col_ctr]
        for it_col_ctr in reversed(np.arange(0, len(ar_it_chd_tre)))]
    mt_occ_wkr_mesh = mt_occ_wkr_mesh[np.lexsort(ls_all_cols_last2first)]

    # which columns to select
    ar_it_col_sel_occ = np.array(ar_it_occ_lyr) - 1
    ar_it_col_sel_wkr = [idx for idx in np.arange(
        len(ar_it_chd_tre)) if idx not in ar_it_col_sel_occ]

    # generate panda from matrix
    pd_occ_wkr_mesh = pd.DataFrame(mt_occ_wkr_mesh).applymap(str)

    # Generate string concated multi-column names
    pd_occ_wkr_mesh['occ_str'] = pd_occ_wkr_mesh[ar_it_col_sel_occ].apply(
        '_'.join, axis=1)
    pd_occ_wkr_mesh['wkr_str'] = pd_occ_wkr_mesh[ar_it_col_sel_wkr].apply(
        '_'.join, axis=1)

    # Generate unique ascending index keys
    ls_occ_concat_str = list(pd_occ_wkr_mesh['occ_str'])
    ls_wrk_concat_str = list(pd_occ_wkr_mesh['wkr_str'])

    ls_unique_occ_str = []
    ls_unique_occ_idx = []
    ls_it_occ_idx = []
    it_occ_idx = -1
    for occ_str in ls_occ_concat_str:
        if (occ_str not in ls_unique_occ_str):
            it_occ_idx = it_occ_idx + 1
            ls_unique_occ_str.append(occ_str)
            ls_unique_occ_idx.append(it_occ_idx)
        ls_it_occ_idx.append(ls_unique_occ_str.index(occ_str))

    ls_unique_wkr_str = []
    ls_unique_wkr_idx = []
    ls_it_wkr_idx = []
    it_wkr_idx = -1
    for wkr_str in ls_wrk_concat_str:
        if (wkr_str not in ls_unique_wkr_str):
            it_wkr_idx = it_wkr_idx + 1
            ls_unique_wkr_str.append(wkr_str)
            ls_unique_wkr_idx.append(it_wkr_idx)
        ls_it_wkr_idx.append(ls_unique_wkr_str.index(wkr_str))

    # Add index to columns
    pd_occ_wkr_mesh['occ'] = ls_it_occ_idx
    pd_occ_wkr_mesh['wkr'] = ls_it_wkr_idx

    dc_return = {'ls_it_occ_idx': ls_it_occ_idx,
                 'ls_it_wkr_idx': ls_it_wkr_idx,
                 'pd_occ_wkr_mesh': pd_occ_wkr_mesh}

    if verbose:
        for st_key, dc_val in dc_return.items():
            print('d-436188 key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_return


def cme_simu_demand_params_ces_single(it_worker_types=2,
                                      it_occ_types=2,
                                      fl_power_min=0.1,
                                      fl_power_max=0.8,
                                      bl_simu_q=False, bl_simu_p=False,
                                      bl_simu_params=True,
                                      it_seed=123,
                                      verbose=True, tp_fl_share_rescalar=1):
    # it_worker_types = 2
    # it_occ_types = 2
    # fl_power_min = 0.1
    # fl_power_max = 0.8
    # it_seed = 123

    if bl_simu_q:
        mt_rand_q = np.random.rand(it_worker_types, it_occ_types)
    if bl_simu_p:
        mt_rand_p = np.random.rand(it_worker_types, it_occ_types)

    # Simulate parameters
    if bl_simu_params:
        np.random.seed(it_seed)
        mt_rand_coef_shares = np.random.rand(it_worker_types, it_occ_types)
        # DEBUG: Was not sure if sum of share < 1 works, it does, note have to adjust lower total output since TFP is
        # lower, but the optimal demand function solution will generate different results due to differeing TFP.
        # mt_rand_coef_shares = (mt_rand_coef_shares/np.sum(mt_rand_coef_shares))/tp_fl_share_rescalar
        mt_rand_coef_shares = (mt_rand_coef_shares / np.sum(mt_rand_coef_shares))
        fl_power = (np.random.uniform(
            low=fl_power_min, high=fl_power_max, size=(1,))).item()
    else:
        mt_rand_coef_shares = np.reshape(
            [None] * (it_worker_types * it_occ_types),
            [it_worker_types, it_occ_types])
        fl_power = None

    # Each input
    dc_ces = {}
    it_input_key_ctr = 0
    ar_it_inputs = np.array([], dtype=int)
    for it_worker_type_ctr in np.arange(it_worker_types):
        for it_occ_type_ctr in np.arange(it_occ_types):

            fl_qty = None
            if bl_simu_q:
                fl_qty = mt_rand_q[it_worker_type_ctr, it_occ_type_ctr]
            fl_wge = None
            if bl_simu_p:
                fl_wge = mt_rand_p[it_worker_type_ctr, it_occ_type_ctr]

            dc_cur_input, _ = cme_simu_demand_ces_inner_dict(
                it_lyr=1, it_prt=0,
                it_wkr=it_worker_type_ctr, it_occ=it_occ_type_ctr,
                fl_shr=mt_rand_coef_shares[it_worker_type_ctr,
                it_occ_type_ctr],
                fl_qty=fl_qty, fl_wge=fl_wge)

            it_input_key_ctr = it_input_key_ctr + 1
            dc_ces[it_input_key_ctr] = dc_cur_input
            ar_it_inputs = np.append(ar_it_inputs, it_input_key_ctr)

    # Aggregation layer
    dc_ces[0], _ = cme_simu_demand_ces_inner_dict(
        it_lyr=0, fl_pwr=fl_power,
        ar_it_ipt=ar_it_inputs)

    # Print
    if (verbose):
        print('d-833467 name: dc_ces')
        pprint.pprint(dc_ces, width=2)

    return dc_ces


def cme_simu_demand_params_ces_nested(ar_it_chd_tre=[2, 2, 3], ar_it_occ_lyr=[1],
                                      fl_power_min=0.1,
                                      fl_power_max=0.8,
                                      bl_simu_q=False, bl_simu_p=False,
                                      bl_simu_params=True,
                                      it_seed=123,
                                      verbose=False, verbose_debug=False):
    """Simulated nested-ces tree and flattened dict

    A nested-ces problem, stored as a nested-dictionary. The nested-CES has potentially
    different elasticity of substitution at each layer, but the same elasticity of substitution
    within layer. 

    We allow for any number of CES layers, and any number of children at each layer. The number of child for each node 
    at each layer is assumed to be identical. 

    We randomly draw the share parameters for each sub-nest, with 
    the share parameter values within each subnest summing up to one. 
    We also randomly draw the elasticity of substitution relevant parameter at each 
    subnest. 

    Assume symmetric nesting. There are some M layers of nesting. 
    N_m is the number of children at each layer. And we have a subset of 
    these layers that identical occupational groups, with the
    remaining subset of layers identifying worker types. We want to generate
    unique occ and wrk keys based on these meshed-nests.

    We generated a `dc_ces_flat`, which is a dictionary with keys for every child in the 
    nested-ces tree, including child along all branches, and a dictionary as value for the
    parameters associated with this child, index for this child's parents, and index
    for this child's children.

    We also generate `dc_ces_nest_keys_tree` that stores keys for each child, in tree
    structure for easy view. 

    dc_ces_nest_keys_tree = {
        7: [1, 2],
        8: [3, 4],
        9: [5, 6]
        }
    }

    # How to generate the tree above, several layers,
    # One option is to have homogeneous structures
    # array of number of children in each layer of subtree
    # Specify from top to bottom, this is for even tree construction only
    ar_it_chd_tre = [2, 2, 3, 3]
    ar_it_chd_tre = [2, 2, 2, 2]
    ar_it_chd_tre = [2, 2, 2]
    ar_it_chd_tre = [2, 3]
    # ar_it_chd_tre = [10]
    # [lyr1,lyr2,lyr3,lyr4], layer0 is the layer over layer1

    # counter for which layers are the occupation layer, counting starts at 0
    # suppose we have skill, occupation, gender, three layers in sequence
    # layer count is [2 (lay1), 2 (lay2), 2 (lay3)] this correspond
    # to the layer naming convention in the single-layer-nested-dict below
    ar_it_occ_lyr = [1]

    Parameters
    ----------
    ar_it_chd_tre : list, optional
        A list of M element for the M layers of nesting, each number corresponds to N_m, NOte if M is 2, then the root layer 
        is 0, the first layer is 1, and the second layer is 2, the list here includes [N_1, N_2], by default [2,2,2,2]
    ar_it_occ_lyr : list, optional
        A list of less than M elements, including which of the M layers is for occupations suppose
        the second and third layer are then [2,3]. by default [2,3]
    fl_power_min : float, optional
        Minimum value of the elasticity of substitution parameter (not elasticity of substitution itself) 
        to be drawn that is homogeneous within layer and possibly heterogeneous across layers, by default 0.8
    fl_power_max : float, optional
        Maximum value of the elasticity of substitution parameter (not elasticity of substitution itself) 
        to be drawn that is homogeneous within layer and possibly heterogeneous across layers, by default 0.8
    bl_simu_q : boolean, optional
        Whether to generate random quantities at the bottom layer, by default False.
    bl_simu_p : boolean, optional
        Whether to generate random prices at the bottom layer, by default False.
    bl_simu_params: boolean, optional
        All demand-side parameters are set to None, not randomly drawn. This is to generate skeleton frame
        to be merged with actual estimates/data from the model, not testing random parameters. Keeping all
        None values assures that parameters from merged table are not based on random draws.
    it_seed : int, optional
        Random seed for drawing parameters, by default 123
    verbose : bool, optional
        _description_, by default True
    """
    # We multiple layers, and need to generate power terms and share parameters at each layer.
    # # A. This is a tree structure, but do not need to do it too formally
    # # list of of lists
    # # Single layer
    # ls_ls_it_itp = [1, 2, 3, 4]
    # # Multiple layer fully specified tree
    # ls_ls_it_ipt = {
    #     1: [2, 3],
    #     4: [5, 6, 7],
    #     8: {
    #         9: [10, 11],
    #         12: [13, 14]
    #     }
    # }

    # Generate work and occ keys
    ls_mnest_wrk_occ_keys = cme_simu_demand_mnest_wkr_occ_keys(
        ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=ar_it_occ_lyr,
        verbose=verbose_debug)
    ls_it_occ_idx = ls_mnest_wrk_occ_keys['ls_it_occ_idx']
    ls_it_wkr_idx = ls_mnest_wrk_occ_keys['ls_it_wkr_idx']
    pd_occ_wkr_mesh = ls_mnest_wrk_occ_keys['pd_occ_wkr_mesh']

    # Shared parameters
    it_layer_cnt = len(ar_it_chd_tre)

    # Initialize share parameters and power parameters
    # Assumed here that the power coefficients are homogeneous within-layer
    # 1. Total coefficient count
    ar_it_tre_up_cnt = np.empty([it_layer_cnt, ], dtype=int)

    # this
    for it_layer_ctr_rev in np.arange(it_layer_cnt):
        it_tre_up_cnt = np.prod(
            np.array(ar_it_chd_tre[0:(it_layer_cnt - it_layer_ctr_rev)]))
        ar_it_tre_up_cnt[it_layer_ctr_rev] = it_tre_up_cnt
    if verbose_debug:
        print(f'{ar_it_tre_up_cnt=}')

    # Simulate data Q and P
    if bl_simu_q:
        ar_rand_q = np.random.rand(ar_it_tre_up_cnt[0])
    if bl_simu_p:
        ar_rand_p = np.random.rand(ar_it_tre_up_cnt[0])

    # Simulate parameters
    if bl_simu_params:
        np.random.seed(it_seed)
        ar_rand_coef_shares = np.random.rand(np.sum(ar_it_tre_up_cnt))
        ar_power = np.random.uniform(
            low=fl_power_min, high=fl_power_max, size=(it_layer_cnt,))
    else:
        ar_rand_coef_shares = np.array([None] * np.sum(ar_it_tre_up_cnt))
        ar_power = np.array([None] * it_layer_cnt)

    # Handling bottom layer trees
    it_leaf_nodes = ar_it_chd_tre[-1]
    ls_it_leaf_nodes = list(range(1, (it_leaf_nodes + 1)))
    it_tre_up_cnt = np.prod(np.array(ar_it_chd_tre[0:-1]))
    it_lower_count = it_tre_up_cnt * it_leaf_nodes + 1
    # Construct bottom most dictionary, flat, to be reorganized later.
    dc_ces_keys_tree = dict([(it_tre_ctr + it_lower_count,
                              list(ls_it_leaf_nodes + (it_tre_ctr) * it_leaf_nodes))
                             for it_tre_ctr in np.arange(it_tre_up_cnt)])

    # Storage
    dc_ces_flat = {}
    # Bottom-most layer, generate dictionary elements
    for it_parent, ar_it_children in dc_ces_keys_tree.items():
        if verbose_debug:
            print(f'{it_parent=} and {ar_it_children=}')

        # sum shares within-nest to be equal to 1
        if bl_simu_params:
            ar_rand_coef_shares_children = ar_rand_coef_shares[ar_it_children]
            ar_rand_coef_shares_children = ar_rand_coef_shares_children / \
                                           np.sum(ar_rand_coef_shares_children)
        else:
            # All None Values
            ar_rand_coef_shares_children = ar_rand_coef_shares[ar_it_children]

        # loop over the list of bottom children
        for it_chd_ctr, it_child in enumerate(ar_it_children):
            # Possibly generating random quantities
            if bl_simu_p:
                fl_wge = ar_rand_p[it_child - 1]
            else:
                fl_wge = None

            if bl_simu_q:
                fl_qty = ar_rand_q[it_child - 1]
            else:
                fl_qty = None

            # occ and wkr index pre-calculated
            it_occ_idx = ls_it_occ_idx[it_child - 1]
            it_wkr_idx = ls_it_wkr_idx[it_child - 1]

            # bottom most layer, no power coefficient.
            fl_power = None
            ar_it_ipt = None

            # these fl_shr sum to 1 within nest
            fl_shr = ar_rand_coef_shares_children[it_chd_ctr]
            dc_ces_flat[it_child], _ = cme_simu_demand_ces_inner_dict(
                it_lyr=it_layer_cnt, it_prt=it_parent,
                it_wkr=it_wkr_idx, it_occ=it_occ_idx,
                fl_pwr=fl_power, fl_shr=fl_shr,
                ar_it_ipt=ar_it_ipt,
                fl_qty=fl_qty, fl_wge=fl_wge)

    if verbose_debug:
        pprint.pprint(dc_ces_keys_tree, width=10)
        pprint.pprint(dc_ces_flat, width=10)

    # Working on higher layers
    it_lower_count_last = it_lower_count
    it_lower_count = it_lower_count + it_tre_up_cnt

    # it_ctr_layer: layer bottom most is counter 1 but layer = len(ar)
    it_ctr_layer = 1
    # it_brh_chd_layer: number of children per branch at layer
    for it_brh_chd_layer_rev in reversed(ar_it_chd_tre[0:-1]):
        if verbose_debug and it_ctr_layer > 1:
            pprint.pprint(dc_ces_keys_tree, width=10)
            pprint.pprint(dc_ces_flat, width=10)

        it_ctr_layer = it_ctr_layer + 1
        it_tre_up_cnt_last = it_tre_up_cnt
        it_tre_up_cnt = int(np.prod(np.array(ar_it_chd_tre[0:-it_ctr_layer])))
        if verbose_debug:
            print(f'{it_brh_chd_layer_rev=} and {it_ctr_layer=} and {it_tre_up_cnt=}')
            print(f'{it_lower_count=} and {it_tre_up_cnt_last=} ')

        dc_update = {}
        for it_branch_ctr in np.arange(it_tre_up_cnt):
            it_dc_update_ctr = it_branch_ctr + it_lower_count

            dc_nest = {}
            if verbose_debug:
                print(f'{it_branch_ctr=} and {it_dc_update_ctr=}')

            # Created multi-layer nested dictionary
            it_within_cnt = it_lower_count - it_tre_up_cnt_last - \
                            1 + it_branch_ctr * it_brh_chd_layer_rev

            # Single layer nested CES
            ar_it_children = np.arange(
                it_within_cnt, it_within_cnt + it_brh_chd_layer_rev)
            if bl_simu_params:
                ar_rand_coef_shares_children = ar_rand_coef_shares[ar_it_children]
                ar_rand_coef_shares_children = ar_rand_coef_shares_children / \
                                               np.sum(ar_rand_coef_shares_children)
            else:
                ar_rand_coef_shares_children = ar_rand_coef_shares[ar_it_children]

            for it_cur_chd in np.arange(it_brh_chd_layer_rev):
                it_within_cnt = it_within_cnt + 1

                # Multi-layer nested dictionary
                if verbose_debug:
                    print(f'{it_within_cnt=}')
                dc_nest[it_within_cnt] = dc_ces_keys_tree[it_within_cnt]

                # Single layer nested CES
                it_lyr = it_layer_cnt - (it_ctr_layer - 1)
                it_parent = it_dc_update_ctr
                fl_power = ar_power[it_lyr]
                fl_shr = ar_rand_coef_shares_children[it_cur_chd]
                ar_it_ipt = [key for key, nest in dc_ces_flat.items()
                             if nest['prt'] == it_within_cnt]
                dc_ces_flat[it_within_cnt], _ = cme_simu_demand_ces_inner_dict(
                    it_lyr=it_lyr, it_prt=it_parent,
                    fl_pwr=fl_power, fl_shr=fl_shr,
                    ar_it_ipt=ar_it_ipt)

            dc_update[it_dc_update_ctr] = dc_nest

        dc_ces_keys_tree = dc_update
        it_lower_count = it_dc_update_ctr + 1

        # it_lower_count_last = it_lower_count
        # it_lower_count = it_lower_count + it_tre_cnt
    dc_ces_keys_tree = dc_ces_keys_tree[it_lower_count - 1]

    # top layer of CES
    it_top_key_index = it_lower_count - 1
    it_lyr = 0
    it_parent = None
    fl_power = ar_power[it_lyr]
    fl_shr = None
    ar_it_ipt = [key for key, nest in dc_ces_flat.items()
                 if nest['prt'] == it_top_key_index]
    dc_ces_flat[it_top_key_index], _ = cme_simu_demand_ces_inner_dict(
        it_lyr=it_lyr, it_prt=it_parent,
        fl_pwr=fl_power, fl_shr=fl_shr,
        ar_it_ipt=ar_it_ipt)

    dc_return = {'pd_occ_wkr_mesh': pd_occ_wkr_mesh,
                 'dc_ces_keys_tree': dc_ces_keys_tree,
                 'dc_ces_flat': dc_ces_flat}

    if verbose:
        for st_key, dc_val in dc_return.items():
            print('d-90791 key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_return


# Test
if __name__ == "__main__":
    # All filled data/params
    dc_ces_flat = cme_simu_demand_params_ces_single(
        it_worker_types=2,
        it_occ_types=2,
        fl_power_min=0.1,
        fl_power_max=0.8,
        it_seed=123,
        verbose=True)

    # All None
    dc_ces_flat = cme_simu_demand_params_ces_single(
        it_worker_types=2,
        it_occ_types=2,
        bl_simu_params=False,
        verbose=True)

    # All filled data/params, simulated parameters and quantity and wages
    dc_dc_ces_nested = cme_simu_demand_params_ces_nested(
        ar_it_chd_tre=[2, 2, 3], ar_it_occ_lyr=[2],
        fl_power_min=0.1,
        fl_power_max=0.8,
        it_seed=123,
        bl_simu_q=True,
        bl_simu_p=True,
        verbose=True, verbose_debug=True)

    # All None, generate skeleton to be filled with estimates
    # results or data.
    dc_dc_ces_nested = cme_simu_demand_params_ces_nested(
        ar_it_chd_tre=[2,3,4], ar_it_occ_lyr=[1],
        bl_simu_q=False,
        bl_simu_p=False,
        bl_simu_params=False,
        verbose=True, verbose_debug=True)

    pd.pandas.set_option('display.max_rows', None)
    pd.pandas.set_option('display.max_columns', None)
    print(dc_dc_ces_nested['pd_occ_wkr_mesh'])

    import prjlecm.input.cme_inpt_convert as cme_inpt_convert

    dc_ces_flat = dc_dc_ces_nested['dc_ces_flat']
    pd_ces_flat = cme_inpt_convert.cme_convert_dc2pd(dc_ces_flat)
    print(pd_ces_flat)

    pd.pandas.set_option('display.max_rows', 10)
    pd.pandas.set_option('display.max_columns', 10)
