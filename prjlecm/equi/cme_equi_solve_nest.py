import pprint

import numpy as np
import pandas as pd

import prjlecm.demand.cme_dslv_eval as cme_dslv_eval
import prjlecm.equi.cme_equi_solve as cme_equi_solve
import prjlecm.equi.cme_equi_solve_gen_inputs as cme_equi_solve_gen_inputs
import prjlecm.input.cme_inpt_parse as cme_inpt_parse
import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply


def cme_equi_solve_nest_test(
        bl_simu_params_of_simu=False, verbose=False, verbose_debug=False):
    """Testing Structure

    Parameters
    ----------
    bl_simu_params_of_simu : bool, optional
        If True, will use default parameters, if False, randomly draw parameters, by default False
    """

    # 1. Simulated Nested Input Dictionary

    # 1.a. Parameters to simulate

    # 1.a.1. These two parameters matter for both demands and supply, defaults
    ar_it_chd_tre = [5, 2, 10]
    ar_it_occ_lyr = [1]

    # 1.a.2. Demand side parameter, defaults
    fl_output_target = 0.2
    it_simu_demand_seed = 123
    # the two parameters below are bounds, they do not need to be simulated.
    fl_simu_demand_power_min = -0.1
    fl_simu_demand_power_max = 0.9

    # 1.a.3. Supply side parameters, defauls
    fl_simu_supply_itc_min = -2
    fl_simu_supply_itc_max = 1
    fl_simu_supply_slp_min = 0.9
    fl_simu_supply_slp_max = 1.2
    it_simu_supply_seed = 456

    # 1.a.4. Simulate
    if bl_simu_params_of_simu:
        # Random output target
        fl_output_target = np.random.uniform(low=0.1, high=0.3, size=(1,))[0]
        # 2 to 5 CES layers, each
        ar_it_chd_tre = np.random.randint(
            2, 10, size=np.random.randint(2, 5, size=1))
        ar_it_occ_lyr = list(np.sort(np.random.choice(
            np.arange(len(ar_it_chd_tre)) + 1, size=np.random.randint(1, len(ar_it_chd_tre)), replace=False)))
        # Simulating random demand share and elasticity parameter seed
        it_simu_demand_seed = int(np.random.randint(1, 1e6, size=1))
        # Simulating random supply side "intercept" and "slope" parameters
        it_simu_supply_seed = int(np.random.randint(1, 1e6, size=1))

    # 1.a Demand
    dc_dc_ces_nested = cme_inpt_simu_demand.cme_simu_demand_params_ces_nested(
        ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=ar_it_occ_lyr,
        fl_power_min=fl_simu_demand_power_min,
        fl_power_max=fl_simu_demand_power_max,
        it_seed=it_simu_demand_seed,
        bl_simu_q=False,
        verbose=False, verbose_debug=False)
    dc_ces_flat = dc_dc_ces_nested['dc_ces_flat']
    if verbose_debug:
        pprint.pprint('d-69828 Step 1 dc_ces_flat:')
        pprint.pprint(dc_ces_flat)

    ar_pwr_across_layers = cme_inpt_parse.cme_parse_demand_lyrpwr(dc_ces_flat)
    dc_parse_occ_wkr_lst = cme_inpt_parse.cme_parse_occ_wkr_lst(dc_ces_flat)
    it_worker_types = dc_parse_occ_wkr_lst['it_wkr_cnt']
    it_occ_types = dc_parse_occ_wkr_lst['it_occ_cnt']

    # 1.b Supply
    dc_supply_lgt, ar_splv_totl_acrs_i = cme_inpt_simu_supply.cme_simu_supply_params_lgt(
        it_worker_types=it_worker_types,
        it_occ_types=it_occ_types,
        fl_itc_min=fl_simu_supply_itc_min,
        fl_itc_max=fl_simu_supply_itc_max,
        fl_slp_min=fl_simu_supply_slp_min,
        fl_slp_max=fl_simu_supply_slp_max,
        it_seed=it_simu_supply_seed,
        verbose=False)

    # Run the iterative nested solution structure
    cme_equi_solve_nest(dc_ces_flat, dc_supply_lgt, ar_splv_totl_acrs_i,
                        fl_output_target=fl_output_target,
                        it_iter_max=1e2, fl_iter_tol=1e-3,
                        fl_solu_tol=1e-2,
                        verbose=verbose, verbose_debug=verbose_debug)


def cme_equi_solve_nest(dc_ces_flat, dc_supply_lgt, ar_splv_totl_acrs_i,
                        fl_output_target,
                        it_iter_max=1e2, fl_iter_tol=1e-5,
                        fl_solu_tol=1e-3,
                        verbose=False, verbose_debug=False):
    failed = False
    failed_flag = 0
    it_iter_ctr = 0
    fl_diff_joint = 1e5
    ar_diff_joint = np.empty([0])
    ar_ces_output = np.empty([0])
    ls_dc_equi_solv_sfur = []

    # Iterating over sni
    while it_iter_ctr < it_iter_max and fl_diff_joint > fl_iter_tol:
        it_iter_ctr = it_iter_ctr + 1
        if verbose_debug:
            pprint.pprint('d-69828 Step 1 iterating:')
            print(f'{it_iter_ctr=}')

        # 2. Generate shc values in input dicts
        dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_mpl(dc_ces_flat)
        if verbose_debug:
            pprint.pprint('d-69828 Step 2 dc_ces_flat:')
            pprint.pprint(dc_ces_flat)

        # 3. Generate equilibrium inputs matrixes from dicts and shc (1st) or sni (later) key
        if (it_iter_ctr == 1):
            st_rela_shr_key = 'shc'
            # Do this only once:
            dc_sprl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_supply_dict_converter_nonest(
                dc_supply_lgt, verbose=verbose_debug)
        else:
            st_rela_shr_key = 'sni'

        dc_dmrl_intr_slpe = cme_equi_solve_gen_inputs.cme_equi_demand_dict_converter_nest(
            dc_ces_flat, st_rela_shr_key=st_rela_shr_key,
            verbose=verbose_debug)

        # 4. Solve the problem given input matrixes
        dc_equi_solve_sone = cme_equi_solve.cme_equi_solve_sone(
            dc_sprl_intr_slpe, dc_dmrl_intr_slpe,
            verbose=verbose_debug)
        fl_nu1_solved, dc_equi_solv_sfur = cme_equi_solve.cme_equi_solve(
            dc_sprl_intr_slpe, ar_splv_totl_acrs_i,
            dc_dmrl_intr_slpe,
            dc_equi_solve_sone,
            fl_output_target=fl_output_target,
            verbose_slve=False,
            verbose=verbose_debug)
        ls_dc_equi_solv_sfur.append(dc_equi_solv_sfur)
        if (dc_equi_solv_sfur is None):
            failed = True
            failed_flag = 1
            fl_ces_output = None
            break
        fl_ces_output = dc_equi_solv_sfur['fl_ces_output']
        if (np.abs(fl_ces_output - fl_output_target) > fl_solu_tol):
            failed = True
            failed_flag = 2
            break
        pd_qtlv_all = dc_equi_solv_sfur['pd_qtlv_all']
        pd_wglv_all = dc_equi_solv_sfur['pd_wglv_all']
        if (it_iter_ctr > 1):
            fl_diff_qtlv = np.sum(
                np.sum(np.abs(pd_qtlv_all - pd_qtlv_all_last)))
            fl_diff_wglv = np.sum(
                np.sum(np.abs(pd_wglv_all - pd_wglv_all_last)))
            fl_diff_joint = fl_diff_qtlv + fl_diff_wglv

        pd_qtlv_all_last = pd_qtlv_all
        pd_wglv_all_last = pd_wglv_all

        # 5. Update Input Dicts with bottom (highest-layer) Qs
        __, ls_maxlyr_key, __ = cme_inpt_parse.cme_parse_demand_tbidx(
            dc_ces_flat)
        for it_key_idx in ls_maxlyr_key:
            # Get wkr and occ index for current child
            it_wkr_idx = dc_ces_flat[it_key_idx]['wkr']
            it_occ_idx = dc_ces_flat[it_key_idx]['occ']

            # wrk index matches with rows, 1st row is first worker, wkr index 0
            # occ + 1 because first column is leisure quantityt
            fl_qtlv = pd_qtlv_all.iloc[it_wkr_idx, it_occ_idx + 1]
            # Replace highest/bottommost layer's qty terms with equi solution qs.s
            dc_ces_flat[it_key_idx]['qty'] = fl_qtlv

        if verbose_debug:
            pprint.pprint('d-69828 Step 5 dc_ces_flat:')
            pprint.pprint(dc_ces_flat)

        # 6. Populate higher level Qs in Input Dicts
        dc_ces_flat = cme_dslv_eval.cme_prod_ces_nest_output(
            dc_ces_flat, verbose=verbose_debug, verbose_debug=False)

        # 7. Go back to step 2 and 6, iterate over that, but now generate and use sni key rather than shc key
        if verbose_debug:
            print(f'{it_iter_ctr=} and {fl_diff_joint=}')

        # Collect results for tracking
        ar_diff_joint = np.append(ar_diff_joint, fl_diff_joint)
        ar_ces_output = np.append(ar_ces_output, fl_ces_output)

    if failed_flag == 0 and (fl_diff_joint > fl_iter_tol):
        failed_flag = 3

    if failed is False and verbose_debug is True:
        pprint.pprint('d-69828 Step 7 ls_dc_equi_solv_sfur:')
        pprint.pprint(ls_dc_equi_solv_sfur)

    # Iteration history
    mt_iter_track = np.column_stack((ar_diff_joint, ar_ces_output))
    pd_iter_track = pd.DataFrame(mt_iter_track,
                                 index=[
                                     'iter=' + str(i)
                                     for i in np.arange(np.shape(mt_iter_track)[0])],
                                 columns=["diff_joint", "ces_output"])

    # More detailed info output
    # print(
    #     f'CES dimension, nest layers:{ar_it_chd_tre=}, occ layers:{ar_it_occ_lyr}')
    # print(
    #     f'CES power coef, layer-specific:{np.round(ar_pwr_across_layers,3)=}')
    # print(
    #     f'Output target is {fl_output_target=}, actual is {fl_ces_output=}')
    # print(f'{it_simu_demand_seed=} and {it_simu_supply_seed=}')
    dc_equi_solve_nest_info = {
        'failed': failed,
        'failed_flag': failed_flag,
        # 'ar_it_chd_tre': ar_it_chd_tre,
        # 'ar_it_occ_lyr': ar_it_occ_lyr,
        # 'ar_pwr_across_layers': ar_pwr_across_layers,
        'fl_output_target': fl_output_target,
        'fl_ces_output': fl_ces_output,
        'pd_iter_track': pd_iter_track
    }
    if verbose:
        if dc_equi_solv_sfur is not None:
            for st_key, dc_val in dc_equi_solv_sfur.items():
                print('d-69828, dc_equi_solv_sfur, key:' + str(st_key))
                pprint.pprint(dc_val, width=10)
        for st_key, dc_val in dc_equi_solve_nest_info.items():
            print('d-69828, dc_equi_solve_nest_info, key:' + str(st_key))
            pprint.pprint(dc_val, width=10)

    return dc_equi_solv_sfur, dc_equi_solve_nest_info


if __name__ == "__main__":
    import timeit

    start = timeit.default_timer()
    cme_equi_solve_nest_test(bl_simu_params_of_simu=False, verbose=True)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
