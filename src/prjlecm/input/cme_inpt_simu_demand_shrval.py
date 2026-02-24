import pprint as pprint

import ast as ast
import numpy as np
import pandas as pd

import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand


def cme_simu_demand_key_vallab(
        ar_it_chd_tre=[3, 2, 3],
        dc_chd_tre_val={"min": [1999, None, 0, 30],
                        "max": [1999, None, 1, 50],
                        # min bound across trees
                        "min_bndx": [1994, None, 0, 30],
                        # max bound across trees
                        "max_bndx": [2018, None, 1, 30],
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
        children in each subnest at the layer. The min and max bound across trees are specified in case other
        trees will have alternative bounds on the values, this is useful because in generating random share
        parameters, we want consistency across trees for rescaling purposes, so we can use the max boudns to
        normalize, rather than the indiviual-tree specific bound.
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
    st_nvb_key = dc_key_return["nvb"]
    st_nvr_key = dc_key_return["nvr"]
    st_nlb_key = dc_key_return["nlb"]

    # 10. Get the min, max and strings
    chd_tre_val_min = dc_chd_tre_val['min']
    chd_tre_val_max = dc_chd_tre_val['max']
    chd_tre_val_str = dc_chd_tre_val['str']
    chd_tre_val_minbndx = dc_chd_tre_val['min_bndx']
    chd_tre_val_maxbndx = dc_chd_tre_val['max_bndx']

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

        # 46. Get min and max values
        fl_val_minbndx = chd_tre_val_minbndx[it_layer]
        fl_val_maxbndx = chd_tre_val_maxbndx[it_layer]
        ls_nvb = f'[{fl_val_minbndx}, {fl_val_maxbndx}]'

        # Create dictionary
        # combinging arrays and scalar values
        dc_data = {
            st_lyr_key: it_layer,
            st_nvi_key: ar_vni,
            st_nvl_key: ar_vnl,
            st_nvb_key: ls_nvb,
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
    st_nvb_key = dc_key_return["nvb"]
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
        st_nvb = pd_node_data_row[st_nvb_key].values[0]
        ls_nvb = ast.literal_eval(st_nvb)
        st_nvr = pd_node_data_row[st_nvr_key].values[0]
        st_nlb = pd_node_data_row[st_nlb_key].values[0]

        # Include nvl, nvr, and nlb as a part of dictionary
        key_dict[st_nvl_key] = fl_nvl
        key_dict[st_nvb_key] = ls_nvb
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
                          "min_bndx": [1994, None, 0, 30],
                          "max_bndx": [2018, None, 1, 50],
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
        dc_chd_tre_val=dc_chd_tre_val, dc_chd_tre_lab=dc_chd_tre_lab,
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

    dc_chd_gen_shr = {"prd": [0, 3, 1, 3],
                      "psc": [5e0, 1e1, 1e1, 1e1],
                      "psd": [123, 456, 789, 101],
                      "plr": [0, 0, 2, 3],
                      "prw": [None, None, True, True],
                      "pcl": [None, True, False, False],
                      "pex": [False, False, True, True]}

    ar_it_chd_tre_wth_zr = [1] + ar_it_chd_tre

    verbose = True
    verbose_debug = True

    # dc_mn_fl_poly keys are layers
    dc_mn_fl_poly = {}
    # Loop over layers, from the tip of the pyramid to the base
    for it_layer, it_chd_tre in enumerate(ar_it_chd_tre_wth_zr):

        if verbose_debug:
            print(f'{it_layer=} and {it_chd_tre=}')

        # Not the total number of children at layer, but the total
        # number of subnests.
        it_nests_at_layer = int(np.prod(np.array(ar_it_chd_tre_wth_zr[0:it_layer])))
        it_child_pernest_at_layer = it_chd_tre
        print(f'{it_nests_at_layer=}')

        # Get layer-specific parameters
        it_prd = dc_chd_gen_shr["prd"][it_layer]
        fl_psc = dc_chd_gen_shr["psc"][it_layer]
        it_psd = dc_chd_gen_shr["psd"][it_layer]
        bl_prw = dc_chd_gen_shr["prw"][it_layer]
        bl_pcl = dc_chd_gen_shr["pcl"][it_layer]
        if verbose_debug:
            print(f'd-43605 dc_chd_gen_shr')
            print(f'{it_prd=}, {fl_psc=}, {it_psd=}')
            print(f'{bl_prw=}, {bl_pcl=}')

        # Draw seeds parameters
        # it_psd is the seed for drawing seeds
        np.random.seed(it_psd)
        if bl_prw is True and bl_pcl is True:
            # each sub-nest parameters differ, each child follows different polynomial
            mt_it_seeds = np.random.randint(
                low=1, high=1e6, size=[it_nests_at_layer, it_child_pernest_at_layer])
        elif bl_prw is True:
            # Each sub-nest follows own polynomial, same polynomial within next across children
            # bl_pcl is false
            mt_it_seeds = np.random.randint(
                low=1, high=1e6, size=[it_nests_at_layer, 1])
            mt_it_seeds = np.tile(mt_it_seeds, (1,it_child_pernest_at_layer))
        elif bl_pcl is True:
            # each child follows the same polynomial
            # bl_prw is False
            mt_it_seeds = np.random.randint(
                low=1, high=1e6, size=[1, it_child_pernest_at_layer])
            mt_it_seeds = np.tile(mt_it_seeds, (it_nests_at_layer, 1))
        else:
            # This is all other cases where uniform draws acoss children
            mt_it_seeds = np.random.randint(
                low=1, high=1e6, size=[1, 1])
            mt_it_seeds = np.tile(mt_it_seeds, (it_nests_at_layer, it_child_pernest_at_layer))
        if verbose_debug:
            print(f'd-43605 mt_it_seeds, rows=nests, cols=children-per-nest')
            print(mt_it_seeds)

        # Generate random parameters
        # + 1 for 2nd order polynomial to have 3 coefficients.
        # wi = with intercept
        it_prd_wi = it_prd + 1
        mn_fl_poly = np.empty([it_prd_wi, it_nests_at_layer,it_child_pernest_at_layer])
        for it_row in np.arange(it_nests_at_layer):
            for it_col in np.arange(it_child_pernest_at_layer):

                # set seed for drawing polynomial coefficients
                np.random.seed(mt_it_seeds[it_row, it_col])
                # Draw polynomial coefficients
                ar_poly_coef = np.random.rand(it_prd_wi)-0.5

                # Adjust coefficients
                ar_fl_coef_adj = [fl_psc**i for i in np.arange(it_prd_wi)]
                ar_poly_coef_adj = ar_poly_coef/ar_fl_coef_adj
                # print(f'{ar_fl_rand=}\n{ar_fl_poly_coef=}')

                # Store adjusted coefficients
                mn_fl_poly[:, it_row, it_col] = ar_poly_coef_adj

        dc_mn_fl_poly[it_layer] = mn_fl_poly

        if verbose_debug:
            print(f'd-43605 mn_fl_poly, mat=poly, rows=nests, cols=children-per-nest')
            print(mn_fl_poly)
            print('\n')

    pprint.pprint(dc_mn_fl_poly)

    # do the following:
    # 1. Iterate through all elements of panda of dictionary group by group
    # 2. At each element, get:
    #    - sort by parent key, generate parent key index resindex, as group-indexxc
    #    - nvi = polynomial row
    #    - given the layer to go to, go up up up several times
    #    - if nest-sum-based: generate polynomials within nest, rescale
    #    - with normalization also update the parameters this is the "true parameters"
    #    - have different procedures in dealing with matrix sub-set
    #     depending on if within-nest differ.
    #    - for variation over time, respect the intercept term
    #     of the first column draw, that sets the max boundary how to
    #         draw this and work with table.
    # pd_ces_flat.iloc["lyr"]