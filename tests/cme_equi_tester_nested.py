# Testing nested problem, with I=2 and J=2.
# TODO: Rely on the existing tester structure in cme_equi_solve_nest, obtain outputs, and solves CES nested
# TODO: Given prices, what are optimal nested CES allocation choices. Aggregation prices, and layer-specific choices. 

import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

import prjlecm.demand.cme_dslv_opti as cme_dslv_opti
import prjlecm.equi.cme_equi_solve_nest as cme_equi_solve_nest
import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.input.cme_inpt_parse as cme_inpt_parse
import prjlecm.supply.cme_splv_opti as cme_splv_opti

verbose = True
bl_pandas_out = True

# 1. Solve the equilibrium problem
start = timeit.default_timer()
dc_ces_flat, dc_supply_lgt, \
    dc_equi_solv_sfur, dc_equi_solve_nest_info, \
    dc_supply_lgt, ar_splv_totl_acrs_i = \
    cme_equi_solve_nest.cme_equi_solve_nest_test(it_fixed_group=11, verbose=True)
stop = timeit.default_timer()
print('Time: ', stop - start)
# Get equilibium quantities and prices as matrixes
pd_qtlv_all = dc_equi_solv_sfur['pd_qtlv_all']
pd_wglv_all = dc_equi_solv_sfur['pd_wglv_all']

# 2. Given wages, solve optimal labor demands
dc_ces_flat_opti_demand = cme_dslv_opti.cme_prod_ces_nested_solver(
    dc_ces_flat, fl_Q_agg=None,
    verbose=False, verbose_debug=False)
# Get optimal demand quantites
pd_qtlv_all_demand, __ = cme_inpt_parse.cme_parse_qtwg_mat(dc_ces_flat, ar_splv_totl_acrs_i)
# difference between optimal demand given prices and equilibrium quantities
pd_qtlv_equi_vs_demand = pd_qtlv_all - pd_qtlv_all_demand
fl_diff_equi_demand = np.sum(np.sum(np.abs(pd_qtlv_equi_vs_demand)))
if verbose:
    print(f'{pd_qtlv_all_demand=}')
    print(f'{pd_qtlv_equi_vs_demand=}')
    print(f'{fl_diff_equi_demand=}')

# 3. Check on supply decisions
dc_supply_lgt, pd_qtlv_all_supply = cme_splv_opti.cme_supply_lgt_solver(dc_supply_lgt)
# difference between optimal supply given prices and equilibrium quantities
pd_qtlv_equi_vs_supply = pd_qtlv_all - pd_qtlv_all_supply
fl_diff_equi_supply = np.sum(np.sum(np.abs(pd_qtlv_equi_vs_supply)))
if verbose:
    print(f'{pd_qtlv_all_supply=}')
    print(f'{pd_qtlv_equi_vs_supply=}')
    print(f'{fl_diff_equi_supply=}')

# 4. Supply and Demand Dictionaries to Pandas Dataframes
# The pandas dataframes would be the input structures for estimation to obtain underlying parameters
# The pandas dataframes are also provide the table format to store estimates parameters to simulate
# equilibrium results
tb_supply_lgt = cme_inpt_convert.cme_convert_dc2pd(dc_supply_lgt, input_type="supply")
tb_ces_flat = cme_inpt_convert.cme_convert_dc2pd(dc_ces_flat, input_type="demand")
if verbose:
    pd.pandas.set_option('display.max_columns', None)
    print(f'{tb_supply_lgt=}')
    print(f'{tb_ces_flat=}')

# 5. Export files
if bl_pandas_out:
    srt_pydebug = Path.joinpath(Path.home(), "Downloads", "PrjLECM_Debug")
    srt_pydebug.mkdir(parents=True, exist_ok=True)
    # Files to export to csv
    spn_csv_path = Path.joinpath(srt_pydebug,
                                 f'{tb_supply_lgt=}'.split('=')[0] +
                                 '-' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
    tb_supply_lgt.to_csv(spn_csv_path, sep=",")
    print(f'{spn_csv_path=}')
    spn_csv_path = Path.joinpath(srt_pydebug,
                                 f'{tb_ces_flat=}'.split('=')[0] +
                                 '-' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
    tb_ces_flat.to_csv(spn_csv_path, sep=",")
    print(f'{spn_csv_path=}')
