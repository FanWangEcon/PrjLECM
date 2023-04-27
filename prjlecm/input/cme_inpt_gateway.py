"""
The :mod:`prjlecm.input.cme_inpt_gateway` loads demand and supply input 
files or simulate input files. Given these inputs, we solve the equilibrium
problem. 

Includes method :func:`ar_draw_random_normal`.
"""

import os
import pprint

import numpy as np
import pandas as pd

import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.input.cme_inpt_parse as cme_inpt_parse
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply


def cme_inpt_gateway_load(spt_path_demand,
                          snm_file_demand,
                          spt_path_supply,
                          snm_file_supply, verbose=False):
    # 1. Demand
    spn_path = os.path.join(os.sep, spt_path_demand, snm_file_demand)
    spn_path = os.path.abspath(spn_path)
    pd_demand = pd.read_csv(spn_path)
    dc_ces_flat = cme_inpt_convert.cme_convert_pd2dc(pd_demand, input_type='demand')

    # 2. Supply
    spn_path = os.path.join(os.sep, spt_path_supply, snm_file_supply)
    spn_path = os.path.abspath(spn_path)
    pd_supply = pd.read_csv(spn_path)
    dc_supply_lgt = cme_inpt_convert.cme_convert_pd2dc(pd_supply, input_type='supply')

    # 3. Output target 
    __, __, it_lyr0_key = cme_inpt_parse.cme_parse_demand_tbidx(
        dc_ces_flat)
    fl_output_target = dc_ces_flat[it_lyr0_key]['qty']

    # 4. Potential labor levels
    ar_splv_totl_acrs_i = cme_inpt_parse.cme_parse_supply_qtp(dc_supply_lgt)

    # 5. delete wage and qty if they exist from file
    ls_it_idx = dc_ces_flat.keys()
    for it_idx in ls_it_idx:
        dc_ces_flat[it_idx]['qty'] = None
        dc_ces_flat[it_idx]['wge'] = None
    ls_it_idx = dc_supply_lgt.keys()
    for it_idx in ls_it_idx:
        dc_supply_lgt[it_idx]['qty'] = None
        dc_supply_lgt[it_idx]['wge'] = None

    # Return
    dc_return = {'dc_ces_flat': dc_ces_flat,
                 'fl_output_target': fl_output_target,
                 'dc_supply_lgt': dc_supply_lgt,
                 'ar_splv_totl_acrs_i': ar_splv_totl_acrs_i}
    if verbose:
        for st_key, dc_val in dc_return.items():
            print('d-27986-cme_inpt_gateway_load key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return (dc_return)


def cme_inpt_gateway_simu(it_fixed_group=11,
                          verbose=False, verbose_debug=False):
    """Generate fixed or random supply and demand side parameters

    Parameters
    ----------
    it_fixed_group : int, optional
        Which type group of parameters to generate, by default 11. 
        11 is for old model nesting structure.
    """

    # 1. Parameters to simulate
    # 1.a. These two parameters matter for both demands and supply, defaults
    ar_it_chd_tre = [5, 2, 10]
    ar_it_occ_lyr = [1]

    # 1.b. Demand side parameter, defaults
    fl_output_target = 0.2
    it_simu_demand_seed = 123
    # the two parameters below are bounds, they do not need to be simulated.
    fl_simu_demand_power_min = -0.1
    fl_simu_demand_power_max = 0.9

    # 1.c. Supply side parameters, defauls
    fl_simu_supply_itc_min = -2
    fl_simu_supply_itc_max = 1
    fl_simu_supply_slp_min = 0.9
    fl_simu_supply_slp_max = 1.2
    it_simu_supply_seed = 456

    if it_fixed_group == 101:
        # Fixed 3 by 3 with default parameters above
        ar_it_chd_tre = np.array([3, 3])
        ar_it_occ_lyr = [2]
    elif it_fixed_group == 102:
        fl_output_target = 0.12716618330033197
        ar_it_chd_tre = np.array([4, 2, 5])
        ar_it_occ_lyr = [2, 3]
        it_simu_demand_seed = 456280
        it_simu_supply_seed = 441313

    # 2. Simulate
    if it_fixed_group is None or (it_fixed_group >= 1 and it_fixed_group <= 100):
        # Random output target
        fl_output_target = np.random.uniform(low=0.1, high=0.3, size=(1,))[0]
        if it_fixed_group == 1:
            # Small test
            ar_it_chd_tre = np.array([2, 2])
            ar_it_occ_lyr = [2]
        elif it_fixed_group == 2:
            # Test solve 5 by 5 problem with random params
            ar_it_chd_tre = np.array([3, 3])
            ar_it_occ_lyr = [2]
        elif it_fixed_group == 3:
            # Three layers testing
            ar_it_chd_tre = np.array([2, 2, 3])
            ar_it_occ_lyr = [2]
        elif it_fixed_group == 4:
            # Three layers testing
            ar_it_chd_tre = np.array([2, 3, 2, 10])
            ar_it_occ_lyr = [2]
        elif it_fixed_group == 11:
            # UK paper, BBF initial version
            # layer 1 = flexible vs not, occupation
            # layer 2 = 3 age groups (25-34, 35-44, 45-55) and 2 gender groups
            ar_it_chd_tre = np.array([2, 6])
            ar_it_occ_lyr = [1]
        elif it_fixed_group == 12:
            # UK paper, BBFW test-initial expanded version
            # layer 1 = flexible vs not, occupation
            # layer 2 = 2 gender groups
            # layer 3 = 11 age groups, every 3 years
            ar_it_chd_tre = np.array([2, 2, 11])
            ar_it_occ_lyr = [1]
        else:
            # 2 to 5 CES layers, each
            ar_it_chd_tre = np.random.randint(
                2, 10, size=np.random.randint(2, 5, size=1))
            ar_it_occ_lyr = list(np.sort(np.random.choice(
                np.arange(len(ar_it_chd_tre)) + 1, size=np.random.randint(1, len(ar_it_chd_tre)), replace=False)))
        # Simulating random demand share and elasticity parameter seed
        it_simu_demand_seed = int(np.random.randint(1, 1e6, size=1))
        # Simulating random supply side "intercept" and "slope" parameters
        it_simu_supply_seed = int(np.random.randint(1, 1e6, size=1))

    if verbose:
        pprint.pprint('d-69828 Test Parameters:')
        print(f'{fl_output_target=}')
        print(f'{ar_it_chd_tre=}')
        print(f'{ar_it_occ_lyr=}')
        print(f'{it_simu_demand_seed=}')
        print(f'{it_simu_supply_seed=}')

    # # 1. Demand
    # dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
    #     ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=ar_it_occ_lyr,
    #     fl_power_min=fl_simu_demand_power_min,
    #     fl_power_max=fl_simu_demand_power_max,
    #     it_seed=it_simu_demand_seed,
    #     bl_simu_q=False,
    #     verbose=False, verbose_debug=False)
    # dc_ces_flat = dc_dc_ces_nested['dc_ces_flat']
    # if verbose:
    #     print(f'{cme_inpt_parse.cme_parse_demand_lyrpwr(dc_ces_flat)=}')
    #
    # if verbose_debug:
    #     pprint.pprint('d-69828 Step 1 dc_ces_flat:')
    #     pprint.pprint(dc_ces_flat)
    #
    # # ar_pwr_across_layers = cme_inpt_parse.cme_parse_demand_lyrpwr(dc_ces_flat)
    # dc_parse_occ_wkr_lst = cme_inpt_parse.cme_parse_occ_wkr_lst(dc_ces_flat)
    # it_worker_types = dc_parse_occ_wkr_lst['it_wkr_cnt']
    # it_occ_types = dc_parse_occ_wkr_lst['it_occ_cnt']
    #
    # # 2. Supply
    # dc_supply_lgt, ar_splv_totl_acrs_i = cme_inpt_simu_supply.cme_simu_supply_params_lgt(
    #     it_worker_types=it_worker_types,
    #     it_occ_types=it_occ_types,
    #     fl_itc_min=fl_simu_supply_itc_min,
    #     fl_itc_max=fl_simu_supply_itc_max,
    #     fl_slp_min=fl_simu_supply_slp_min,
    #     fl_slp_max=fl_simu_supply_slp_max,
    #     it_seed=it_simu_supply_seed,
    #     verbose=False)

    # 3. Call the dicts generator
    dc_ces_flat, dc_supply_lgt, ar_splv_totl_acrs_i = cme_inpt_gateway_gen_dicts(
        ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=ar_it_occ_lyr,
        fl_power_min=fl_simu_demand_power_min, fl_power_max=fl_simu_demand_power_max,
        it_simu_demand_seed=it_simu_demand_seed,
        bl_simu_q=False, bl_simu_p=False,
        fl_itc_min=fl_simu_supply_itc_min, fl_itc_max=fl_simu_supply_itc_max,
        fl_slp_min=fl_simu_supply_slp_min, fl_slp_max=fl_simu_supply_slp_max,
        it_simu_supply_seed=it_simu_supply_seed,
        bl_simu_params=True,
        verbose=verbose, verbose_debug=verbose_debug,
    )

    # Return
    dc_return = {'dc_ces_flat': dc_ces_flat,
                 'fl_output_target': fl_output_target,
                 'dc_supply_lgt': dc_supply_lgt,
                 'ar_splv_totl_acrs_i': ar_splv_totl_acrs_i}

    if verbose:
        for st_key, dc_val in dc_return.items():
            print('d-27986-cme_inpt_gateway_simu key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return (dc_return)

def cme_inpt_gateway_gen_dicts(
        ar_it_chd_tre=[2, 2, 3], ar_it_occ_lyr=[1],
        fl_power_min=0.1, fl_power_max=0.8,
        it_simu_demand_seed=123,
        bl_simu_q=False, bl_simu_p=False,
        fl_itc_min=-2, fl_itc_max=2,
        fl_slp_min=0.4, fl_slp_max=0.6,
        it_simu_supply_seed=456,
        bl_simu_params=True,
        verbose=False, verbose_debug=False):
    """Generates demand and supply dictionaries

    Generate parameters based on simulated/drawn values, or to generate a
    skeleton without values.

    :param ar_it_chd_tre:
    :param ar_it_occ_lyr:
    :param fl_power_min:
    :param fl_power_max:
    :param it_simu_demand_seed:
    :param bl_simu_q:
    :param bl_simu_p:
    :param bl_simu_params:
    :param fl_itc_min:
    :param fl_itc_max:
    :param fl_slp_min:
    :param fl_slp_max:
    :param it_simu_supply_seed:
    :param verbose:
    :param verbose_debug:
    :Other Parameters: A short string remitting reader
        to :py:func:`cme_inpt_simu_supply.cme_simu_supply_params_lgt` function
    :return:
    """
    # 1. Generate demand dictionary
    dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
        ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=ar_it_occ_lyr,
        fl_power_min=fl_power_min,
        fl_power_max=fl_power_max,
        it_seed=it_simu_demand_seed,
        bl_simu_q=bl_simu_q,
        bl_simu_p=bl_simu_p,
        bl_simu_params=bl_simu_params,
        verbose=False, verbose_debug=False)
    dc_ces_flat = dc_dc_ces_nested['dc_ces_flat']
    if verbose:
        print(f'{cme_inpt_parse.cme_parse_demand_lyrpwr(dc_ces_flat)=}')
    if verbose_debug:
        pprint.pprint('d-69828-cme_inpt_gateway_gen_dicts-1:')
        pprint.pprint(dc_ces_flat)

    # 2. Parse worker and occupation counts
    # ar_pwr_across_layers = cme_inpt_parse.cme_parse_demand_lyrpwr(dc_ces_flat)
    dc_parse_occ_wkr_lst = cme_inpt_parse.cme_parse_occ_wkr_lst(dc_ces_flat)
    it_worker_types = dc_parse_occ_wkr_lst['it_wkr_cnt']
    it_occ_types = dc_parse_occ_wkr_lst['it_occ_cnt']

    # 3. Generate supply dictionary
    dc_supply_lgt, ar_splv_totl_acrs_i = cme_inpt_simu_supply.cme_simu_supply_params_lgt(
        it_worker_types=it_worker_types,
        it_occ_types=it_occ_types,
        fl_itc_min=fl_itc_min,
        fl_itc_max=fl_itc_max,
        fl_slp_min=fl_slp_min,
        fl_slp_max=fl_slp_max,
        it_seed=it_simu_supply_seed,
        bl_simu_params=bl_simu_params,
        verbose=False)

    if verbose_debug:
        pprint.pprint('d-69828-cme_inpt_gateway_gen_dicts-2:')
        pprint.pprint(dc_supply_lgt)
        pprint.pprint('d-69828-cme_inpt_gateway_gen_dicts-3:')
        print(f'{ar_splv_totl_acrs_i=}')

    return dc_ces_flat, dc_supply_lgt, ar_splv_totl_acrs_i


if __name__ == "__main__":

    from pathlib import Path

    ar_it_test = [1,2,3]
    for it_test in ar_it_test:

        if it_test == 1:
            cme_inpt_gateway_simu(it_fixed_group=101,
                                  verbose=True,
                                  verbose_debug=False)

        elif it_test == 2:
            # Get path to the data folder
            spt_prt_pathlib = Path(__file__).parent.parent.parent.resolve()
            spt_data = Path.joinpath(spt_prt_pathlib, "data")

            # Files to export to csv
            snm_file_demand = 'tb_ces_flat-test-c-2-6-out.csv'
            snm_file_supply = 'tb_supply_lgt-test-c-2-6-out.csv'

            cme_inpt_gateway_load(spt_path_demand=spt_data,
                                  snm_file_demand=snm_file_demand,
                                  spt_path_supply=spt_data,
                                  snm_file_supply=snm_file_supply,
                                  verbose=True)

        elif it_test == 3:
            # Generate skeleton without any simulated parameters
            cme_inpt_gateway_gen_dicts(
                ar_it_chd_tre=[2, 2, 3], ar_it_occ_lyr=[1],
                bl_simu_params=False,
                verbose=True, verbose_debug=True
            )