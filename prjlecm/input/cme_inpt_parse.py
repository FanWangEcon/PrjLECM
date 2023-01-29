import numpy as np
import pprint
# Contains some functions that parse the demand single-nest CES structure


def cme_parse_occ_wkr_lst(dc_dm_or_sp, ls_it_idx=None, verbose=False):

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
        for st_key, dc_val in dc_parse_occ_wkr_lst .items():
            print('d-186126 key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_parse_occ_wkr_lst


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


def cme_parse_tree(dc_ces_flat):
    pass


if __name__ == "__main__":

    # Generate flat nested dict
    import cme_inpt_simu_demand
    dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
        ar_it_chd_tre=[2, 2, 3], ar_it_occ_lyr=[2],
        fl_power_min=0.1,
        fl_power_max=0.8,
        it_seed=222,
        bl_simu_q=True,
        verbose=False, verbose_debug=False)
    dc_ces_flat = dc_dc_ces_nested['dc_ces_flat']
    pprint.pprint(dc_ces_flat, width=10)

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