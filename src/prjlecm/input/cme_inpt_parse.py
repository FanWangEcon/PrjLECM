import numpy as np
import pprint
import pandas as pd
import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
import prjlecm.input.cme_inpt_simu_demand_shrval as cme_inpt_simu_demand_shrval

# Contains some functions that parse the demand single-nest CES structure

# Get various keys
# not fully implemented below to replace hard-code
dc_key_return, _, _ = cme_inpt_simu_demand.cme_simu_demand_ces_inner_dict()
st_lyr_key = dc_key_return["lyr"]
st_prt_key = dc_key_return["prt"]
st_nvi_key = dc_key_return["nvi"]
st_nvl_key = dc_key_return["nvl"]
st_nvb_key = dc_key_return["nvb"]
st_nvr_key = dc_key_return["nvr"]
st_nlb_key = dc_key_return["nlb"]
st_key_node = cme_inpt_convert.cme_dictkey_pdvarname()


def cme_parse_occ_wkr_lst(dc_dm_or_sp, ls_it_idx=None, verbose=False):
    """Get occ and wrk index list from demand supply dicts
    """
    
    if ls_it_idx is None:
        # Might be called for the supply function, loop over all elements
        ls_it_idx = dc_dm_or_sp.keys()

    # Find the list of values contained in occ (j 1 through J) and wkr (i 1 through I)
    ls_it_occ = [dc_dm_or_sp[it_idx]['occ']
                 for it_idx in ls_it_idx
                 if dc_dm_or_sp[it_idx]['occ'] is not None]
    ls_it_wkr = [dc_dm_or_sp[it_idx]['wkr']
                 for it_idx in ls_it_idx
                 if dc_dm_or_sp[it_idx]['wkr'] is not None]
    ls_it_occ = list(set(ls_it_occ))
    it_occ_cnt = len(ls_it_occ)
    ls_it_wkr = list(set(ls_it_wkr))
    it_wkr_cnt = len(ls_it_wkr)

    if verbose:
        print(f"{ls_it_occ=}")
        print(f"{ls_it_wkr=}")

    # ls_it_occ = dc_equi_sup_occ_wkr_lst["ls_it_occ"]
    # ls_it_wkr = dc_equi_sup_occ_wkr_lst["ls_it_wkr"]
    # it_occ_cnt = dc_equi_sup_occ_wkr_lst["it_occ_cnt"]
    # it_wkr_cnt = dc_equi_sup_occ_wkr_lst["it_wkr_cnt"]
    dc_parse_occ_wkr_lst = {"ls_it_occ": ls_it_occ,
                            "ls_it_wkr": ls_it_wkr,
                            "it_occ_cnt": it_occ_cnt,
                            "it_wkr_cnt": it_wkr_cnt,
                            "ls_it_idx": ls_it_idx}

    if verbose:
        for st_key, dc_val in dc_parse_occ_wkr_lst.items():
            print('d-186126 key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_parse_occ_wkr_lst

def cme_parse_supply_qtp(dc_supply_lgt, verbose=False):
    """Get potential labor array from supply dictionary

    The same potential labor level appears for each occ of the 
    the wkr. Create an array of potential worker levels, length
    equal to the number of worker types, indexed by it_wkr.
    
    Parameters
    ----------
    dc_supply_lgt : _type_
        _description_
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # 1. Get worker index list
    dc_parse_occ_wrk_lst = cme_parse_occ_wkr_lst(dc_supply_lgt, ls_it_idx=None)
    ls_it_wkr = dc_parse_occ_wrk_lst['ls_it_wkr']
    ls_it_idx = dc_parse_occ_wrk_lst['ls_it_idx']
    it_wkr_cnt = dc_parse_occ_wrk_lst['it_wkr_cnt']

    # 2. Fill out supply available vector
    ar_splv_totl_acrs_i = np.random.rand(it_wkr_cnt) 
    for it_wkr in ls_it_wkr:
        ls_fl_qtp = [dc_supply_lgt[it_idx]['qtp']
                    for it_idx in ls_it_idx
                    if dc_supply_lgt[it_idx]['wkr'] == it_wkr]
        fl_qtp = list(set(ls_fl_qtp))[0]
        ar_splv_totl_acrs_i[it_wkr] = fl_qtp

    if verbose:
            print(f'd-186126 {ar_splv_totl_acrs_i=}')

    return ar_splv_totl_acrs_i


        
def cme_parse_demand_lyrpwr(dc_ces_flat, bl_chk_homoinlayer=False):
    """Get the pwr value from each layer

    Assuming pwr value within layer is homogeneous

    Parameters
    ----------
    dc_ces_flat : _type_
        _description_
    bl_chk_homoinlayer : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    # 1. Ge all layers
    ar_layers = list(set([dc_ces_flat[chd_key]['lyr']
                          for chd_key in dc_ces_flat]))
    # 2. array to store "powers"
    ar_pwr_across_layers = np.empty(np.size(ar_layers), dtype=float)

    # 3. Loop over layers
    for it_layer in ar_layers:
        ls_lyr_key = [
            chd_key for chd_key in dc_ces_flat if dc_ces_flat[chd_key]['lyr'] == it_layer]

        ar_pwr_at_layer = list(set([dc_ces_flat[chd_key]['pwr']
                                    for chd_key in ls_lyr_key]))
        ar_pwr_across_layers[it_layer] = ar_pwr_at_layer[0]

    return ar_pwr_across_layers


def cme_parse_demand_tbidx(dc_ces_flat):
    """Get the Number of Layers in nested flat CES Dictionary

    tbidx: top (lyr0) and bottom (max_layer, largest value) layer index

    Parameters
    ----------
    dc_ces_flat : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # 1. Get all layers
    ar_layers = list(set([dc_ces_flat[chd_key]['lyr']
                          for chd_key in dc_ces_flat]))

    # 2. Find the index for the "highest"/bottom-most layer
    it_max_layer = max(ar_layers)

    # 3. Get all keys from the "highest"/bottom-most layer
    ls_maxlyr_key = [
        chd_key for chd_key in dc_ces_flat if dc_ces_flat[chd_key]['lyr'] == it_max_layer]

    # 4. Also get the index for layer 0, this should also be the max key
    ls_lyr0_key = [
        chd_key for chd_key in dc_ces_flat if dc_ces_flat[chd_key]['lyr'] == 0]
    it_lyr0_key = ls_lyr0_key[0]

    return it_max_layer, ls_maxlyr_key, it_lyr0_key


def cme_parse_qtwg_mat(dc_ces_flat, ar_splv_totl_acrs_i=None):
    """
    Obtain from the bottom-most layer, matrix of qty and wge.

    Rows are worker types and columns are occupation types.

    :param dc_ces_flat:
    :return:
    """

    __, ls_maxlyr_key, __ = cme_parse_demand_tbidx(
        dc_ces_flat)
    dc_parse_occ_wkr_lst = cme_parse_occ_wkr_lst(dc_ces_flat)
    it_worker_types = dc_parse_occ_wkr_lst['it_wkr_cnt']
    it_occ_types = dc_parse_occ_wkr_lst['it_occ_cnt']

    mt_qtlv_all = np.zeros([it_worker_types, it_occ_types + 1], dtype=float)
    mt_wglv_all = np.zeros([it_worker_types, it_occ_types], dtype=float)
    for it_key_idx in ls_maxlyr_key:
        it_occ = dc_ces_flat[it_key_idx]['occ']
        it_wkr = dc_ces_flat[it_key_idx]['wkr']

        fl_qty = dc_ces_flat[it_key_idx]['qty']
        mt_qtlv_all[it_wkr, it_occ + 1] = fl_qty
        fl_wge = dc_ces_flat[it_key_idx]['wge']
        mt_wglv_all[it_wkr, it_occ] = fl_wge

    if ar_splv_totl_acrs_i is not None:
        mt_qtlv_all[:, 0] = ar_splv_totl_acrs_i - np.sum(mt_qtlv_all, 1)

    pd_qtlv_all = pd.DataFrame(
        mt_qtlv_all,
        index=["i" + str(i + 1)
               for i in np.arange(np.shape(mt_qtlv_all)[0])],
        columns=["j" + str(j) for j in np.arange(np.shape(mt_qtlv_all)[1])]
    )

    pd_wglv_all = pd.DataFrame(
        mt_wglv_all,
        index=["i" + str(i + 1)
               for i in np.arange(np.shape(mt_wglv_all)[0])],
        columns=["j" + str(j + 1)
                 for j in np.arange(np.shape(mt_wglv_all)[1])]
    )

    return pd_qtlv_all, pd_wglv_all


def cme_parse_tree(dc_ces_flat):
    pass

def cme_parse_node_value(
        dc_ces_flat_with_vallab, 
        it_nthparent_layer = 0,
        it_key_node_child = 1,
        bl_from_pd = False,
        verbose=False):
    """Get node value at node-key or nth-parent of current key

    Construct share parameters using year information, stored as node value
    at the top of the pyramid. Given current node, find parent of parent, 
    until layer of parent matches `it_nthparent_layer`. 

    Parameters
    ----------
    dc_ces_flat_with_vallab : dict
        Nested CES dictionary with node-specific values
    it_nthparent_layer : int, optional
        The parental layer where to obtain node value
    it_key_node_child : int, optional
        The key node of the child
    bl_from_pd : bool, optional
        Convert and obtain from pandas
    verbose : bool, optional
        print more results
    """

    if bl_from_pd:
        pd_ces_flat = cme_inpt_convert.cme_convert_dc2pd(dc_ces_flat_with_vallab)

    it_lyr_cur = 1e5
    it_prt_key_cur = it_key_node_child
    # Going to "higher-ranked" layer
    while it_lyr_cur > it_nthparent_layer:

        if bl_from_pd:
            pd_row_cur = pd_ces_flat.loc[pd_ces_flat[st_key_node] == it_prt_key_cur]
            it_lyr_cur = pd_row_cur[st_lyr_key].item()
            it_prt_key_cur = pd_row_cur[st_prt_key].item()
        else:
            dc_cur_node = dc_ces_flat_with_vallab[it_prt_key_cur]
            it_lyr_cur = dc_cur_node[st_lyr_key]
            it_prt_key_cur = dc_cur_node[st_prt_key]

        if verbose:
            print(f'{it_lyr_cur=}, {it_prt_key_cur=}')

    # Get value at the higher (or even current) layer
    if bl_from_pd:
        fl_nthparent_nvl = pd_row_cur[st_nvl_key].item()
        ar_nthparent_nvb = pd_row_cur[st_nvb_key].item()
    else:
        fl_nthparent_nvl = dc_cur_node[st_nvl_key]
        ar_nthparent_nvb = dc_cur_node[st_nvb_key]

    if verbose:
        print(f'{fl_nthparent_nvl=}')
        print(f'{ar_nthparent_nvb=}')

    return fl_nthparent_nvl, ar_nthparent_nvb

if __name__ == "__main__":

    # Generate flat nested dict
    import cme_inpt_simu_demand
    ar_it_chd_tre = [3, 2, 3]
    dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
        ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=[2],
        fl_power_min=0.1,
        fl_power_max=0.8,
        it_seed=222,
        bl_simu_q=True,
        verbose=False, verbose_debug=False)
    dc_ces_flat = dc_dc_ces_nested['dc_ces_flat']
    pprint.pprint(dc_ces_flat, width=10)

    # Add labels
    dc_chd_tre_val = {"min": [1999, 1, 0, 30],
                        "max": [1999, 3, 1, 50],
                        "min_bndx": [1994, 1, 0, 30],
                        "max_bndx": [2018, 3, 1, 50],
                        "str": ["year", "occ", "gender", "age"]}
    dc_chd_tre_lab = {
        0: ["year 1999"],
        1: ["analytical", "routine", "manual"],
        2: ["male", "female"],
        3: ["age"]
    }
    # return all labels and values
    pd_vallab_all_nodes = cme_inpt_simu_demand_shrval.cme_simu_demand_key_vallab(
        ar_it_chd_tre=ar_it_chd_tre,
        dc_chd_tre_val=dc_chd_tre_val, dc_chd_tre_lab=dc_chd_tre_lab,
        verbose=True)
    # update dict
    dc_ces_flat = cme_inpt_simu_demand_shrval.cme_simu_demand_vallab_to_dict(
        dc_ces_flat, pd_vallab_all_nodes)

    # Test parsing functions
    it_max_layer, ls_maxlyr_key, it_lyr0_key = cme_parse_demand_tbidx(
        dc_ces_flat)
    print(f'{it_max_layer=} and {it_lyr0_key}')
    print(f'{ls_maxlyr_key=}')

    # Test parsing function
    ar_pwr_across_layers = cme_parse_demand_lyrpwr(dc_ces_flat)
    fl_elas = ar_pwr_across_layers[it_max_layer-1]
    print(f'{ar_pwr_across_layers=}')
    print(f'{fl_elas=}')

    dc_ces_flat
    ar_it_chd_tre_restruct = []
    for it_layer in np.arange(it_max_layer):
        ar_it_children_at_node = [len(dc_ces_flat[chd_key]['ipt'])
                                  for chd_key in dc_ces_flat
                                  if dc_ces_flat[chd_key]['lyr'] == it_layer]
        ls_it_children_at_node_unique = list(set(ar_it_children_at_node))
        if len(ls_it_children_at_node_unique)==1:
            ar_it_chd_tre_restruct.append(ls_it_children_at_node_unique[0])
        else:
            ar_it_chd_tre_restruct.append(ls_it_children_at_node_unique) 
    print(ar_it_chd_tre_restruct)

    # Test finding node value
    bl_from_pd = False
    cme_parse_node_value(
        dc_ces_flat, it_nthparent_layer = 3, it_key_node_child = 1, 
        bl_from_pd = bl_from_pd, verbose = True)
    cme_parse_node_value(
        dc_ces_flat, it_nthparent_layer = 2, it_key_node_child = 1, 
        bl_from_pd = bl_from_pd, verbose = True)
    cme_parse_node_value(
        dc_ces_flat, it_nthparent_layer = 1, it_key_node_child = 1, 
        bl_from_pd = bl_from_pd, verbose = True)
    cme_parse_node_value(
        dc_ces_flat, it_nthparent_layer = 0, it_key_node_child = 1, 
        bl_from_pd = bl_from_pd, verbose = True)