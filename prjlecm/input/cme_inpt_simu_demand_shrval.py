import pprint as pprint

import numpy as np
import pandas as pd

import cme_inpt_simu_demand as cme_inpt_simu_demand


def cme_simu_demand_key_vallab(
        ar_it_chd_tre=[3, 2, 3],
        dc_chd_tre_val={"min": [1999, None, 0, 30],
                        "max": [1999, None, 1, 50],
                        "str": ["year", "occ", "gender", "age"]},
        dc_chd_tre_lab={
            0: ["year 1999"],
            1: ["analytical", "routine", "manual"],
            2: ["male", "female"],
            3: ["age"]
        },
        verbose=False, verbose_debug=False):
    """Create dataframe with label and value for each key_node at each layer

    Each key_node at each layer is a unique worker type or an occupation.

    Parameters
    ----------
    ar_it_chd_tre : list
        see prjlecm.input.cme_inpt_simu_demand.cme_simu_demand_params_ces_nested
    dc_chd_tre_val: dict
        Specify the minimum and maximum value, for children at each layer across subnests within the layer.
        The str key is the variable name for the children. Assuming that the children across subnests
        within the same layer have the same set of index and associated values. If min or max is None, that means
        this is not a numerical variable. Will generate linspace sequance from min to max given the number of
        children in each subnest at the layer.
    dc_chd_tre_lab : dict
        Integer key corresponds to each layer, from the tip of the pyramid (0) to the bottom base (highest hkey)
        Specify the string names corresponding to the values, or if there if the value length is one, and the
        value array length is larger than 1, append the string to each value and use this as string name associated
        with each index/value among children within subnest.
    verbose : bool, optional
        Print key outputs
    verbose_debug : bool, optional
        Print detailed outputs

    Returns
    -------
    pd_vallab_all_nodes : pandas dataframe
        Dataframe with lyr and child index (nvi) as observation, with values, labels, strings.
    """

    dc_key_return, _, _ = cme_inpt_simu_demand.cme_simu_demand_ces_inner_dict()
    st_lyr_key = dc_key_return["lyr"]
    st_nvi_key = dc_key_return["nvi"]
    st_nvl_key = dc_key_return["nvl"]
    st_nvr_key = dc_key_return["nvr"]
    st_nlb_key = dc_key_return["nlb"]

    # 10. Get the min, max and strings
    chd_tre_val_min = dc_chd_tre_val['min']
    chd_tre_val_max = dc_chd_tre_val['max']
    chd_tre_val_str = dc_chd_tre_val['str']

    # 15. Expand ar_it_chd_tre with pyramid tip
    # tip of pyramid is not a part of ar_it_chd_tre
    # the tip always has only 1 element
    ar_it_chd_tre_wth_zr = [1] + ar_it_chd_tre

    # 20. Loop over the nest tree base-up triangle
    dc_pd_data = {}
    # Go from the second to tip of the triangle to the base
    # it_chd_tre provides the len of subnest at the layer
    # The top of the triangle is layer 0
    for it_layer, it_chd_tre in enumerate(ar_it_chd_tre_wth_zr):

        # Get parameters for seq
        fl_val_min = chd_tre_val_min[it_layer]
        fl_val_max = chd_tre_val_max[it_layer]
        st_desc = chd_tre_val_str[it_layer]

        # 30. At each layer, the value in ar_it_chd_tre shows the number of unique index
        # vni index
        ar_vni = np.arange(it_chd_tre)

        # 40. Given min and max, unique index length, and current layer, generate dataframe with
        # - lyr layer index index
        # - lyn layer variable name
        # - nvi index
        # - nvl value at index
        # Generate sequence of values at current layer
        # same values across all subnests.
        if fl_val_min is not None:
            ar_vnl = np.linspace(fl_val_min, fl_val_max, it_chd_tre)
        else:
            ar_vnl = None

        # 45. Get variable value labels
        ar_node_lab = dc_chd_tre_lab[it_layer]
        if (len(ar_node_lab) == 1) and \
                (len(ar_vni) != 1) and \
                (ar_vnl is not None):
            ar_node_lab = [ar_node_lab[0] + " " +
                           str(fl_vnl) for fl_vnl in ar_vnl]

        # Create dictionary
        dc_data = {
            st_lyr_key: it_layer,
            st_nvi_key: ar_vni,
            st_nvl_key: ar_vnl,
            st_nvr_key: st_desc,
            st_nlb_key: ar_node_lab
        }
        pd_node_data = pd.DataFrame.from_dict(dc_data, orient="columns")
        dc_pd_data[it_layer] = pd_node_data

        if verbose_debug:
            print(f'd-45210 pd_data, {it_layer=}')
            print(pd_node_data)

    # 50. Stack dataframes
    pd_vallab_all_nodes = pd.concat(dc_pd_data, axis=0).reset_index(drop=True)
    if verbose:
        print('d-45210 pd_vallab_all_nodes')
        print(pd_vallab_all_nodes)

    return pd_vallab_all_nodes


def cme_simu_demand_vallab_to_dict(
        dc_ces_flat, pd_vallab_all_nodes,
        verbose=False, verbose_debug=False):
    """

    Parameters
    ----------
    dc_ces_flat : dict
        see value under dc_ces_flat key for the dict outputed by
        prjlecm.input.cme_inpt_simu_demand.cme_simu_demand_params_ces_nested
    pd_vallab_all_nodes : pandas dataframe
        Output from prjlecm.input.cme_inpt_simu_demand_shrval.cme_simu_demand_key_vallab
    verbose : bool, optional
        Print key outputs
    verbose_debug : bool, optional
        Print detailed outputs

    Returns
    -------
    dc_ces_flat: dict
        Updated dict with the same structure as prior, but values added in for labels and values.
        These repeat across subnests among children at the sama layer
    """
    dc_key_return, _, _ = cme_inpt_simu_demand.cme_simu_demand_ces_inner_dict()
    st_lyr_key = dc_key_return["lyr"]
    st_nvi_key = dc_key_return["nvi"]
    st_nvl_key = dc_key_return["nvl"]
    st_nvr_key = dc_key_return["nvr"]
    st_nlb_key = dc_key_return["nlb"]

    for key_node, key_dict in dc_ces_flat.items():
        # Get layer and node value index (nvi)
        it_lyr = key_dict[st_lyr_key]
        it_nvi = key_dict[st_nvi_key]

        # Search for nvl, nvr, nlb given lyr and nvi
        pd_bl_jnt = (pd_vallab_all_nodes[st_lyr_key] == it_lyr) & \
                    (pd_vallab_all_nodes[st_nvi_key] == it_nvi)
        pd_node_data_row = pd_vallab_all_nodes.loc[pd_bl_jnt]

        # Get node values for nvl, nvr and nlb
        fl_nvl = pd_node_data_row[st_nvl_key].values[0]
        st_nvr = pd_node_data_row[st_nvr_key].values[0]
        st_nlb = pd_node_data_row[st_nlb_key].values[0]

        # Include nvl, nvr, and nlb as a part of dictionary
        key_dict[st_nvl_key] = fl_nvl
        key_dict[st_nvr_key] = st_nvr
        key_dict[st_nlb_key] = st_nlb
        dc_ces_flat[key_node] = key_dict

    if verbose:
        print('d-45210 dc_ces_flat with labels and values')
        pprint.pprint(dc_ces_flat)

    return dc_ces_flat


if __name__ == "__main__":
    import prjlecm.input.cme_inpt_convert as cme_inpt_convert

    # test
    # parameters
    it_param_grp = 1
    if it_param_grp == 1:
        ar_it_chd_tre = [3, 2, 3]
        dc_chd_tre_val = {"min": [1999, None, 0, 30],
                          "max": [1999, None, 1, 50],
                          "str": ["year", "occ", "gender", "age"]}
        dc_chd_tre_lab = {
            0: ["year 1999"],
            1: ["analytical", "routine", "manual"],
            2: ["male", "female"],
            3: ["age"]
        }

    # return all labels and values
    pd_vallab_all_nodes = cme_simu_demand_key_vallab(
        ar_it_chd_tre=ar_it_chd_tre,
        dc_chd_tre_val=dc_chd_tre_val,dc_chd_tre_lab=dc_chd_tre_lab,
        verbose=True)

    # Generate full dictionary without labels and values
    dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
        ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=[1],
        bl_simu_q=False,
        bl_simu_p=False,
        bl_simu_params=False,
        verbose=True, verbose_debug=True)
    dc_ces_flat = dc_dc_ces_nested["dc_ces_flat"]

    # Append labels and values to dict
    dc_ces_flat_with_vallab = cme_simu_demand_vallab_to_dict(
        dc_ces_flat, pd_vallab_all_nodes)

    # print
    pd.pandas.set_option('display.max_rows', None)
    pd.pandas.set_option('display.max_columns', None)
    pd_ces_flat = cme_inpt_convert.cme_convert_dc2pd(dc_ces_flat_with_vallab)
    print(pd_ces_flat)
    pd.pandas.set_option('display.max_rows', 10)
    pd.pandas.set_option('display.max_columns', 10)
