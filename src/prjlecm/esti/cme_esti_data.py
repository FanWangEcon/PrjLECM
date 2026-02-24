# Adjust gender share parameter within flexible and non-flexible jobs.
# Adjust gender elasticity parameter, and age elasticity parameters.
# Grid-based exploratory exercises.
# 1. given original parameters, find elasticity of gender vs age
# run in it_which_calibrate=1
# res in it_param_simu=1
# 2. given (1), find female share parameter for flexilbe and inflexible
# run in it_which_calibrate=2
# res in it_param_simu=2

import os
import time
import timeit

import numpy as np
import pandas as pd
import scipy as sp

import prjlecm.equi.cme_equi_solve_nest as cme_equi_solve_nest
import prjlecm.esti.cme_esti_obj as cme_esti_obj
import prjlecm.input.cme_inpt_convert as cme_inpt_convert
import prjlecm.input.cme_inpt_gateway as cme_inpt_gateway

# The file here is identical to cme_esti_tester_a
# there are two parameters, the y/z share parameter value, and the year

# cb is calibration b
it_which_calibrate = 3
# for it_param_simu in [0, 1, 2, 3, 4]:
it_param_simu = 3
if it_which_calibrate == 0:
    st_file_prefix = "test-cdp" + str(it_param_simu) + "-2-2-2-31"
else:
    st_file_prefix = "test-cdc" + str(it_which_calibrate) + "-2-2-2-31"
st_demand_file_prefix = 'tb_ces_flat_' + st_file_prefix
st_supply_file_prefix = 'tb_supply_lgt_' + st_file_prefix
spt_data = 'C:/Users/fan/Documents/Dropbox (UH-ECON)/repos/PrjLECM/data/'

# ls_chd_uqe_idx_all : list of child unique index all
# 1. Create all combination of year + flx + sex + age
# unique elements
it_year_start = 1993
it_year_adj = 1992
it_year_end = 2017
# it_year_end = 1995
it_year_cnt = it_year_end - it_year_start + 1


# A. Define function
def ffi_esti_gender_params(
        ar_yr_solve=np.arange(it_year_start, it_year_end + 1, step=1),
        fl_wge_slp=0.998,
        fl_flx_power=1 / (1 - 0.337),
        fl_gender_power=1 / (1 - 0.700),
        fl_age_power=1 / (1 - 0.974),
        ar_alpha1=[-1.4379, -0.0361, 0.2447, -0.0450],
        fl_alpha2_female_inf=0.43974978,
        fl_alpha2_female_flx=0.418877039,
        ar_a_demand_age_flx_sex=[-0.000214152, 0.009805873, 0.099267859],
        ar_b_demand_age_flx_sex=[-0.000126304, 0.005343194, 0.101534043],
        ar_c_demand_age_flx_sex=[-0.000151123, 0.007077408, 0.12976847],
        ar_d_demand_age_flx_sex=[-3.96081E-05, 0.001619739, 0.125952938],
        ar_z_nonparam=[1.9783073, 1.88633003, 2.02298967, 1.57471996, 2.17705674,
                       2.3014033, 2.42368213, 2.52749721, 2.58223651, 2.67127097,
                       2.76696538, 2.14991199, 2.15558105, 2.99842456, 3.11969558,
                       2.35282069, 3.29131954, 3.46537009, 3.62964579, 3.8084693,
                       3.9078983, 3.95828349, 4.07143947, 4.18997515, 4.2301114],
        ar_a_supply_year_flx_sex=[0.004009086, -0.106047372, 0.312336957],
        ar_b_supply_year_flx_sex=[0.001557121, -0.057408455, 0.927357343],
        ar_c_supply_year_flx_sex=[0.006603487, -0.169550854, 0.024052461],
        ar_d_supply_year_flx_sex=[0.002740785, -0.061852106, 0.029642648],
        ar_a_supply_age_flx_sex=[0.001259519, -0.066358354, 0.943407018],
        ar_b_supply_age_flx_sex=[0.001318856, -0.053034669, 0.415056319],
        ar_c_supply_age_flx_sex=[0.001048941, -0.052986186, 0.62958407],
        ar_d_supply_age_flx_sex=[0.000117358, -0.005152183, -0.18923571],
        ar_a_supply_m_inf=[1.0232, 0.5212, 0, 0],
        ar_b_supply_f_inf=[0.1230, -0.0452, -0.2213, -0.0024],
        ar_c_supply_m_flx=[0.8260, 0.2372, 0, 0],
        ar_f_supply_f_flx=[-0.5480, -0.3263, -0.3860, 0.0225],
        verbose=False, verbose_debug=False, verbose_outfile=False):
    fl_yz_ratio_adj = 1.7
    fl_qtp_adj = 1e5
    # Assume homogeneous elasticity at each layer
    dc_pwr = {'lyr0': (fl_flx_power - 1) / fl_flx_power,
              'lyr1': (fl_gender_power - 1) / fl_gender_power,
              'lyr2': (fl_age_power - 1) / fl_age_power}

    # # year by year
    # ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=1)
    # # Every five years
    # ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=5)
    # one year
    # ar_yr_solve = np.arange(it_year, it_year+1, step=1)

    it_flx_cnt = 2
    it_edu_cnt = 2
    it_sex_cnt = 2

    it_age_start = 25
    it_age_adj = 24
    it_age_end = 55
    # it_age_end = 27
    it_age_cnt = it_age_end - it_age_start + 1

    ar_it_cnt = [it_year_cnt, it_flx_cnt, it_edu_cnt, it_sex_cnt, it_age_cnt]
    # ar_it_cnt = [it_sex_cnt, it_sex_cnt, it_flx_cnt, it_year_cnt]

    # mesh
    ls_chd_uqe_idx_all = [np.arange(it_cnt)
                          for it_cnt in ar_it_cnt]
    mt_yfsa_mesh = np.array(np.meshgrid(
        *ls_chd_uqe_idx_all)).T.reshape(-1, len(ar_it_cnt))
    if verbose:
        pd.set_option('display.max_rows', None)
        print(mt_yfsa_mesh)
        pd.set_option('display.max_rows', 10)

    # generate panda from matrix
    pd_yfsa_mesh = pd.DataFrame(mt_yfsa_mesh).applymap(str)
    # pd_yfsa_mesh = pd_yfsa_mesh.reindex(columns=[3,2,1,0])
    pd_yfsa_mesh = pd_yfsa_mesh.rename(
        columns={4: "age",
                 3: "female",
                 2: "graduate",
                 1: "flexible",
                 0: "year"})
    # Sort
    pd_yfsa_mesh["age"] = pd_yfsa_mesh.age.astype(int)
    pd_yfsa_mesh["female"] = pd_yfsa_mesh.female.astype(int)
    pd_yfsa_mesh["graduate"] = pd_yfsa_mesh.graduate.astype(int)
    pd_yfsa_mesh["flexible"] = pd_yfsa_mesh.flexible.astype(int)
    pd_yfsa_mesh["year"] = pd_yfsa_mesh.year.astype(int)
    pd_yfsa_mesh = pd_yfsa_mesh.sort_values(
        by=['year', 'flexible', 'graduate', 'female', 'age'])
    # Level correction
    pd_yfsa_mesh['year'] = pd_yfsa_mesh['year'] + it_year_start
    pd_yfsa_mesh['age'] = pd_yfsa_mesh['age'] + it_age_start
    # reset index
    pd_yfsa_mesh = pd_yfsa_mesh.reset_index(drop=True)

    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_yfsa_mesh)
        print(f'{mt_yfsa_mesh.shape=}')
        print(f'{pd_yfsa_mesh.shape=}')
        pd.set_option('display.max_rows', 10)

    # 2. Coefficient dataframes:
    # - Year-flx-sex-specific intercept
    # - Age-flx-sex-specific intercept
    # - flx-sex specific coef
    # - year-specifid demand z, and demand alpha1
    # - year-flx-sex-specific demand coefficent
    # - key-node identifier dataframe that is flx-sex-age-specific
    # Generate string concated multi-column names

    # 2.A. Year-flx-sex-specific intercept
    # Quadratic Fit with Three Points of Data:
    # Supply-side, Sex, Occupation, and Year-specific Coefficients
    # m_inf
    ar_a = ar_a_supply_year_flx_sex
    # ar_a = [0.004009086, -0.106047372, 0.312336957]
    # f_inf
    ar_b = ar_b_supply_year_flx_sex
    # ar_b = [0.001557121, -0.057408455, 0.927357343]
    # m_flx
    ar_c = ar_c_supply_year_flx_sex
    # ar_c = [0.006603487, -0.169550854, 0.024052461]
    # f_flx
    ar_d = ar_d_supply_year_flx_sex
    # ar_d = [0.002740785, -0.061852106, 0.029642648]

    st_coef_col_name = 'psi0_s_j_yr'

    ls_pd_coef = []
    for it_flx in np.arange(it_flx_cnt):
        # it_flx = 1 is flexible
        for it_sex in np.arange(it_sex_cnt):
            # it_sex = 1 is female
            if (it_flx == 0) and (it_sex == 0):
                ar_use = ar_a
            elif (it_flx == 0) and (it_sex == 1):
                ar_use = ar_b
            elif (it_flx == 1) and (it_sex == 0):
                ar_use = ar_c
            elif (it_flx == 1) and (it_sex == 1):
                ar_use = ar_d

            fl_quad = ar_use[0]
            fl_lin = ar_use[1]
            fl_intercept = ar_use[2]

            ar_yr = np.arange(it_year_start, it_year_cnt + it_year_start) - it_year_adj
            ar_coef = fl_intercept + ar_yr * fl_lin + ar_yr ** 2 * fl_quad
            pd_coef = pd.DataFrame({'year': ar_yr, st_coef_col_name: ar_coef})
            pd_coef['female'] = it_sex
            pd_coef['flexible'] = it_flx
            pd_coef['year'] = pd_coef['year'] + it_year_adj

            # append to list
            ls_pd_coef.append(pd_coef)

    pd_coef_psi0_s_j_yr = pd.concat(ls_pd_coef)
    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_coef_psi0_s_j_yr)
        print(f'{pd_coef_psi0_s_j_yr.shape=}')
        pd.set_option('display.max_rows', 10)

    # 2.B Age-flx-sex-specific intercept
    # Quadratic Fit with Three Points of Data:
    # Supply-side, Sex, Occupation, and Age-specific Coefficients
    # m_inf
    # ar_a = [0.001259519, -0.066358354, 0.943407018]
    ar_a = ar_a_supply_age_flx_sex
    # f_inf
    # ar_b = [0.001318856, -0.053034669, 0.415056319]
    ar_b = ar_b_supply_age_flx_sex
    # m_flx
    # ar_c = [0.001048941, -0.052986186, 0.62958407]
    ar_c = ar_c_supply_age_flx_sex
    # f_flx
    # ar_d = [0.000117358, -0.005152183, -0.18923571]
    ar_d = ar_d_supply_age_flx_sex
    st_coef_col_name = 'psi0_s_j_age'

    ls_pd_coef = []
    for it_flx in np.arange(it_flx_cnt):
        # it_flx = 1 is flexible
        for it_sex in np.arange(it_sex_cnt):
            # it_sex = 1 is female
            if (it_flx == 0) and (it_sex == 0):
                ar_use = ar_a
            elif (it_flx == 0) and (it_sex == 1):
                ar_use = ar_b
            elif (it_flx == 1) and (it_sex == 0):
                ar_use = ar_c
            elif (it_flx == 1) and (it_sex == 1):
                ar_use = ar_d

            fl_quad = ar_use[0]
            fl_lin = ar_use[1]
            fl_intercept = ar_use[2]

            ar_age = np.arange(it_age_start, it_age_cnt + it_age_start) - it_age_adj
            ar_coef = fl_intercept + ar_age * fl_lin + ar_age ** 2 * fl_quad
            pd_coef = pd.DataFrame({'age': ar_age, st_coef_col_name: ar_coef})
            pd_coef['female'] = it_sex
            pd_coef['flexible'] = it_flx
            pd_coef['age'] = pd_coef['age'] + it_age_adj

            # append to list
            ls_pd_coef.append(pd_coef)

    pd_coef_psi0_s_j_age = pd.concat(ls_pd_coef)

    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_coef_psi0_s_j_age)
        print(f'{pd_coef_psi0_s_j_age.shape=}')
        pd.set_option('display.max_rows', 10)

    # 2.C Quadratic Fit with Three Points of Data:
    # Demand-side, sex, occupation, and age-specific coefficients
    # m_inf
    # ar_a = [-0.000214152, 0.009805873, 0.099267859]
    ar_a = ar_a_demand_age_flx_sex
    # f_inf
    # ar_b = [-0.000126304, 0.005343194, 0.101534043]
    ar_b = ar_b_demand_age_flx_sex
    # m_flx
    # ar_c = [-0.000151123, 0.007077408, 0.12976847]
    ar_c = ar_c_demand_age_flx_sex
    # f_flx
    # ar_d = [-3.96081E-05, 0.001619739, 0.125952938]
    ar_d = ar_d_demand_age_flx_sex
    st_coef_alphal3 = 'alpha3'

    ls_pd_coef = []
    for it_flx in np.arange(it_flx_cnt):
        # it_flx = 1 is flexible
        for it_sex in np.arange(it_sex_cnt):
            # it_sex = 1 is female
            if (it_flx == 0) and (it_sex == 0):
                ar_use = ar_a
            elif (it_flx == 0) and (it_sex == 1):
                ar_use = ar_b
            elif (it_flx == 1) and (it_sex == 0):
                ar_use = ar_c
            elif (it_flx == 1) and (it_sex == 1):
                ar_use = ar_d

            fl_quad = ar_use[0]
            fl_lin = ar_use[1]
            fl_intercept = ar_use[2]

            ar_age = np.arange(it_age_start, it_age_cnt + it_age_start) - it_age_adj
            ar_coef = fl_intercept + ar_age * fl_lin + ar_age ** 2 * fl_quad
            pd_coef = pd.DataFrame({'age': ar_age, st_coef_alphal3: ar_coef})
            pd_coef['female'] = it_sex
            pd_coef['flexible'] = it_flx
            pd_coef['age'] = pd_coef['age'] + it_age_adj

            # append to list
            ls_pd_coef.append(pd_coef)

    pd_coef_alpha3 = pd.concat(ls_pd_coef)
    pd_temp = pd_coef_alpha3.groupby(['flexible', 'female']).sum()
    fl_alpha3_sum_inf_m = pd_temp.loc[0, 0][[st_coef_alphal3]].values[0]
    fl_alpha3_sum_inf_f = pd_temp.loc[0, 1][[st_coef_alphal3]].values[0]
    fl_alpha3_sum_flx_m = pd_temp.loc[1, 0][[st_coef_alphal3]].values[0]
    fl_alpha3_sum_flx_f = pd_temp.loc[1, 1][[st_coef_alphal3]].values[0]

    ar_bl_inf_m = (pd_coef_alpha3['flexible'] == 0) & (pd_coef_alpha3['female'] == 0)
    ar_bl_inf_f = (pd_coef_alpha3['flexible'] == 0) & (pd_coef_alpha3['female'] == 1)
    ar_bl_flx_m = (pd_coef_alpha3['flexible'] == 1) & (pd_coef_alpha3['female'] == 0)
    ar_bl_flx_f = (pd_coef_alpha3['flexible'] == 1) & (pd_coef_alpha3['female'] == 1)

    pd_coef_alpha3.loc[ar_bl_inf_m, st_coef_alphal3] = \
        pd_coef_alpha3.loc[ar_bl_inf_m, st_coef_alphal3] / fl_alpha3_sum_inf_m
    pd_coef_alpha3.loc[ar_bl_inf_f, st_coef_alphal3] = \
        pd_coef_alpha3.loc[ar_bl_inf_f, st_coef_alphal3] / fl_alpha3_sum_inf_f
    pd_coef_alpha3.loc[ar_bl_flx_m, st_coef_alphal3] = \
        pd_coef_alpha3.loc[ar_bl_flx_m, st_coef_alphal3] / fl_alpha3_sum_flx_m
    pd_coef_alpha3.loc[ar_bl_flx_f, st_coef_alphal3] = \
        pd_coef_alpha3.loc[ar_bl_flx_f, st_coef_alphal3] / fl_alpha3_sum_flx_f

    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_coef_alpha3)
        print(f'{pd_coef_alpha3.shape=}')
        pd.set_option('display.max_rows', 10)

    # 2.D alpha1 and Z over time, only time specific
    # ar_alpha1 = [-1.4379, -0.0361, 0.2447, -0.0450]
    # ar_z = [2.4897, 0.0059, 0.1428, -0.0657]
    ar_st_coef_cnm = ['alpha1', 'yzratio']

    ar_use = ar_alpha1
    fl_cubic = ar_use[3]
    fl_quad = ar_use[2]
    fl_lin = ar_use[1]
    fl_intercept = ar_use[0]
    ar_yr = np.arange(it_year_start, it_year_cnt + it_year_start) - it_year_adj
    ar_coef = fl_intercept + ar_yr * fl_lin + 0.01 * ar_yr ** 2 * fl_quad + 0.001 * ar_yr ** 3 * fl_cubic
    pd_coef_alpha1 = pd.DataFrame({'year': ar_yr, ar_st_coef_cnm[0]: np.exp(ar_coef)})
    pd_coef_alpha1['year'] = pd_coef_alpha1['year'] + it_year_adj
    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_coef_alpha1)
        print(f'{pd_coef_alpha1.shape=}')
        pd.set_option('display.max_rows', 10)

    # not used this component, replaced by ar_z_nonparam
    ar_z = [1, 1, 1, 1]
    ar_use = ar_z
    fl_cubic = ar_use[3]
    fl_quad = ar_use[2]
    fl_lin = ar_use[1]
    fl_intercept = ar_use[0]
    ar_yr = np.arange(it_year_start, it_year_cnt + it_year_start) - it_year_adj
    ar_coef = fl_intercept + ar_yr * fl_lin + 0.01 * ar_yr ** 2 * fl_quad + 0.001 * ar_yr ** 3 * fl_cubic
    ar_coef = 1 / ar_coef
    pd_coef_z = pd.DataFrame({'year': ar_yr, ar_st_coef_cnm[1]: (ar_coef)})
    pd_coef_z['year'] = pd_coef_z['year'] + it_year_adj
    pd_coef_z[ar_st_coef_cnm[1]] = (pd_coef_z[ar_st_coef_cnm[1]] / pd_coef_z.iloc[0, 1]) * fl_yz_ratio_adj

    # update values
    # ar_cali_z_max = np.array([
    #     2.1981192233698477, 2.095922260248987,
    #     2.2477663024211885, 1.7496888436175986,
    #     2.4189519334020724, 2.5571147747332783,
    #     2.692980149063945, 2.808330237593377,
    #     2.869151673443739, 2.968078854638065,
    #     3.0744059722696244, 2.3887911055246738,
    #     2.395090057642401, 3.3315828442507804,
    #     3.466328421495617, 2.6142452151281113,
    #     3.65702170558812, 3.850411210723524,
    #     4.032939767480171, 4.23163255688848,
    #     4.342109227244245, 4.3980927666371965,
    #     4.52382163025815, 4.655527943143686, 4.700123778488247])
    # ar_cali_z = ar_cali_z_max * 0.90
    pd_coef_z[ar_st_coef_cnm[1]] = ar_z_nonparam

    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_coef_z)
        print(f'{pd_coef_z.shape=}')
        pd.set_option('display.max_rows', 10)

    # 2.E
    # m_inf
    # ar_a = [1.0232, 0.5212, 0, 0]
    ar_a = ar_a_supply_m_inf
    # f_inf
    # ar_b = [0.1230, -0.0452, -0.2213, -0.0024]
    ar_b = ar_b_supply_f_inf
    # m_flx
    # ar_c = [0.8260, 0.2372, 0, 0]
    ar_c = ar_c_supply_m_flx
    # f_flx
    # ar_d = [-0.5480, -0.3263, -0.3860, 0.0225]
    ar_d = ar_f_supply_f_flx

    ar_st_coef2_cnm = ['pi2', 'pi3', 'gamma1', 'gamma2']

    ls_pd_coef = []
    for it_flx in np.arange(it_flx_cnt):
        # it_flx = 1 is flexible
        for it_sex in np.arange(it_sex_cnt):
            # it_sex = 1 is female
            if (it_flx == 0) and (it_sex == 0):
                ar_use = ar_a
            elif (it_flx == 0) and (it_sex == 1):
                ar_use = ar_b
            elif (it_flx == 1) and (it_sex == 0):
                ar_use = ar_c
            elif (it_flx == 1) and (it_sex == 1):
                ar_use = ar_d

            pd_coef = pd.DataFrame({ar_st_coef2_cnm[0]: ar_use[0],
                                    ar_st_coef2_cnm[1]: ar_use[1],
                                    ar_st_coef2_cnm[2]: ar_use[2],
                                    ar_st_coef2_cnm[3]: ar_use[3]},
                                   index=[1])
            pd_coef['female'] = it_sex
            pd_coef['flexible'] = it_flx

            # append to list
            ls_pd_coef.append(pd_coef)

    pd_coef_pi_gamma = pd.concat(ls_pd_coef)
    pd_coef_pi_gamma.reset_index(drop=True)

    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_coef_pi_gamma)
        print(f'{pd_coef_pi_gamma.shape=}')
        pd.set_option('display.max_rows', 10)

    # 3. Load in year-age-sex-specific observable dataframe with 5 variables
    # spt_prt_pathlib = Path(__file__).parent.parent.resolve()
    # spt_data = Path.joinpath(spt_prt_pathlib, "data")
    spn_path_dataset2 = os.path.join(os.sep, spt_data, 'dataset2_2merge.csv')
    spn_path_dataset2 = os.path.abspath(spn_path_dataset2)
    pd_dataset2 = pd.read_csv(spn_path_dataset2)

    spn_path_dataset2aux = os.path.join(os.sep, spt_data, 'dataset2_aux_2merge.csv')
    spn_path_dataset2aux = os.path.abspath(spn_path_dataset2aux)
    pd_dataset2aux = pd.read_csv(spn_path_dataset2aux)

    pd_dataset2 = pd.merge(pd_dataset2, pd_dataset2aux,
                           on=['group'])
    pd_dataset2 = pd_dataset2.fillna(0)

    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_dataset2)
        print(f'{pd_dataset2.shape=}')
        pd.set_option('display.max_rows', 10)

    # 4. Merge (1) with (2) and (3)
    pd_yfsa_mesh = pd.merge(pd_yfsa_mesh, pd_coef_psi0_s_j_yr,
                            on=['year', 'female', 'flexible'])
    pd_yfsa_mesh = pd.merge(pd_yfsa_mesh, pd_coef_psi0_s_j_age,
                            on=['age', 'female', 'flexible'])
    pd_yfsa_mesh = pd.merge(pd_yfsa_mesh, pd_coef_pi_gamma,
                            on=['female', 'flexible'])
    pd_yfsa_mesh = pd.merge(pd_yfsa_mesh, pd_dataset2,
                            on=['year', 'age', 'female'])
    pd_yfsa_mesh = pd.merge(pd_yfsa_mesh, pd_coef_alpha1,
                            on=['year'])
    pd_yfsa_mesh = pd.merge(pd_yfsa_mesh, pd_coef_alpha3,
                            on=['age', 'female', 'flexible'])
    pd_yfsa_mesh = pd.merge(pd_yfsa_mesh, pd_coef_z,
                            on=['year'])
    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_yfsa_mesh)
        print(f'{pd_yfsa_mesh.shape=}')
        pd.set_option('display.max_rows', 10)
        for i in pd_yfsa_mesh.columns.tolist():
            print(i)

    # 5. Generate supply side overall "INTERCEPT"
    pd_yfsa_mesh['sp_intercept'] = pd_yfsa_mesh['psi0_s_j_yr'] + pd_yfsa_mesh['psi0_s_j_age']
    pd_yfsa_mesh['sp_pi_x_data'] = \
        pd_yfsa_mesh['pi2'] * pd_yfsa_mesh['shchildrenu5'] + \
        pd_yfsa_mesh['pi3'] * pd_yfsa_mesh['shmarried']
    pd_yfsa_mesh['sp_gamma_x_data'] = \
        pd_yfsa_mesh['gamma1'] * pd_yfsa_mesh['childcare'] + \
        pd_yfsa_mesh['gamma2'] * pd_yfsa_mesh['childbnfts']
    pd_yfsa_mesh['sp_intr_joint'] = \
        pd_yfsa_mesh['sp_intercept'] + \
        pd_yfsa_mesh['sp_pi_x_data'] + \
        pd_yfsa_mesh['sp_gamma_x_data']

    # Print check
    if verbose:
        pd.set_option('display.max_rows', None)
        print(pd_yfsa_mesh)
        print(f'{pd_yfsa_mesh.shape=}')
        pd.set_option('display.max_rows', 10)
        for i in pd_yfsa_mesh.columns.tolist():
            print(i)

    if verbose_outfile:
        snm_file_demand = st_file_prefix + '_datasetZ.csv'
        spn_path_datasetZ = os.path.join(os.sep, spt_data, snm_file_demand)
        pd_yfsa_mesh.to_csv(spn_path_datasetZ, sep=",", index=False)

    # 6. Simulate with random parameters once to generate demand and supply frames
    # change to three tiers below, from two tiers prior
    ar_it_chd_tre = [it_flx_cnt, it_sex_cnt, it_age_cnt]
    ar_it_occ_lyr = [1]
    dc_ces_flat, dc_supply_lgt, ar_splv_totl_acrs_i = \
        cme_inpt_gateway.cme_inpt_gateway_gen_dicts(
            ar_it_chd_tre=ar_it_chd_tre, ar_it_occ_lyr=ar_it_occ_lyr,
            bl_simu_params=False,
            verbose=verbose, verbose_debug=verbose_debug)
    tb_supply_lgt = cme_inpt_convert.cme_convert_dc2pd(dc_supply_lgt, input_type="supply")
    tb_ces_flat = cme_inpt_convert.cme_convert_dc2pd(dc_ces_flat, input_type="demand")
    if verbose:
        pd.pandas.set_option('display.max_columns', None)
        print(f'tb_supply_lgt')
        print(f'{tb_supply_lgt}')
        print(f'tb_ces_flat')
        print(f'{tb_ces_flat}')

    # 7. Cut transform parameter file
    # What are merging keys, we have some att
    # Need to merge jointly by "occ" and "wkr" keys, but problem is that the "data", "param" file does not have
    # OCC and WRK keys. OCC is easy. Assignment of WRK key needs to be in proper sequence, following the design
    # of the skeleton hierachy, this allows for higher up elasticity parameter and ipt values to be correctly
    # matched up.

    # 7.1 Define a function to slice, merge and solve
    def cut_merge_solve_equi_annual(it_year=2005):
        if verbose:
            print(f'Starting with {it_year=}')

        # slice
        pd_yfsa_mesh_curyr = pd_yfsa_mesh[pd_yfsa_mesh["year"] == it_year]

        # A. select relevant colums
        pd_yfsa_mesh_curyr = pd_yfsa_mesh_curyr[
            ["flexible", "female", "age",
             "alpha1", "alpha3", ar_st_coef_cnm[1],
             "potentialworkers", "sp_intr_joint"]]

        # B sort by columns
        pd_yfsa_mesh_curyr = pd_yfsa_mesh_curyr.sort_values(
            ['flexible', 'female', 'age'])
        pd_yfsa_mesh_curyr = pd_yfsa_mesh_curyr.reset_index(drop=True)

        # C generate wkr and occ keys
        sr_occ_idx = pd_yfsa_mesh_curyr.groupby(["flexible"]).ngroup()
        sr_occ_idx.name = "occ"
        sr_wkr_idx = pd_yfsa_mesh_curyr.groupby(["female", "age"]).ngroup()
        sr_wkr_idx.name = "wkr"

        # D Series wrk and occ keys merged back
        pd_yfsa_mesh_curyr = pd_yfsa_mesh_curyr.merge(
            sr_occ_idx, left_index=True, right_index=True)
        pd_yfsa_mesh_curyr = pd_yfsa_mesh_curyr.merge(
            sr_wkr_idx, left_index=True, right_index=True)

        # E Re-index and add 1
        pd_yfsa_mesh_curyr = pd_yfsa_mesh_curyr.reset_index()
        pd_yfsa_mesh_curyr[["index"]] = pd_yfsa_mesh_curyr[["index"]] + 1
        pd_yfsa_mesh_curyr = pd_yfsa_mesh_curyr.rename(
            columns={"index": cme_inpt_convert.cme_dictkey_pdvarname()})

        # F Supply and demand relevant dataframes
        pd_yfsa_mesh_curyr_supply = pd_yfsa_mesh_curyr[
            ["wkr", "occ", "sp_intr_joint", "potentialworkers"]]
        pd_yfsa_mesh_curyr_supply = pd_yfsa_mesh_curyr_supply.rename(
            columns={"sp_intr_joint": "itc",
                     "potentialworkers": "qtp"})
        pd_yfsa_mesh_curyr_demand = pd_yfsa_mesh_curyr[
            ["wkr", "occ", "alpha3"]]
        # Sum check
        pd_yfsa_mesh_curyr_demand.sum()
        pd_yfsa_mesh_curyr_demand = pd_yfsa_mesh_curyr_demand.rename(
            columns={"alpha3": "shr"})

        # print(pd_yfsa_mesh_curyr)

        # G. Supply side parameter file with supply and demand tables
        # Intercept parameters, merged in
        tb_supply_lgt_filled = tb_supply_lgt.combine_first(
            pd_yfsa_mesh_curyr_supply[["itc", "qtp"]]
        )
        tb_supply_lgt_filled["qtp"] = tb_supply_lgt_filled["qtp"] / fl_qtp_adj
        # Slope parameter, same value all rows
        tb_supply_lgt_filled["slp"] = fl_wge_slp
        # get qtp as array, homogeneous within occ, already sorted correct
        ar_bl_occ0 = (tb_supply_lgt_filled['occ'] == 0)
        ar_splv_totl_acrs_i = tb_supply_lgt_filled.loc[ar_bl_occ0, 'qtp'].to_numpy()

        # H. Demand side parameters merge in
        # merge in share parameter at layer=highest
        tb_ces_flat_filled = tb_ces_flat.combine_first(
            pd_yfsa_mesh_curyr_demand[["shr"]]
        )
        # add in share at "lower" layers
        # manually here, need to generalize
        # female share, for inflexible
        # these differ by gender
        tb_ces_flat_filled.loc[124, "shr"] = 1 - fl_alpha2_female_inf
        tb_ces_flat_filled.loc[125, "shr"] = fl_alpha2_female_inf
        tb_ces_flat_filled.loc[126, "shr"] = 1 - fl_alpha2_female_flx
        tb_ces_flat_filled.loc[127, "shr"] = fl_alpha2_female_flx
        # alpha1 is homogeneous within t
        fl_alpha1 = pd_yfsa_mesh_curyr.loc[0, "alpha1"]
        tb_ces_flat_filled.loc[128, "shr"] = 1 - fl_alpha1
        tb_ces_flat_filled.loc[129, "shr"] = fl_alpha1
        # merge in power parameter
        it_lyr_max = max(tb_ces_flat_filled["lyr"])
        for it_lyr in np.arange(it_lyr_max):
            # Power coefficient
            fl_pwr = dc_pwr['lyr' + str(it_lyr)]
            # Fill in value
            ar_bl_lyr = (tb_ces_flat_filled['lyr'] == it_lyr)
            tb_ces_flat_filled.loc[ar_bl_lyr, "pwr"] = fl_pwr
        # merge in yz ratio
        fl_output_target = pd_yfsa_mesh_curyr.loc[0, ar_st_coef_cnm[1]]

        if verbose_debug:
            print(f'{fl_output_target=}')
            print(f'{ar_splv_totl_acrs_i=}')
            print(f'{tb_ces_flat_filled}')
            print(f'{tb_supply_lgt_filled}')

        # pd.pandas.set_option('display.max_rows', None)
        # pd.pandas.set_option('display.max_columns', None)

        # I. Inputs to dictionary
        ar_st_coln = [
            cme_inpt_convert.cme_dictkey_pdvarname(),
            'lyr', 'prt',
            'wkr', 'occ',
            'shr', 'pwr',
            'ipt',
            'qty', 'wge',
            'drv', 'drc', 'shc', 'sni']
        tb_ces_flat_filled = tb_ces_flat_filled.reindex(columns=ar_st_coln)

        tb_ces_flat_filled["drc"] = tb_ces_flat_filled["drc"].astype(float)
        tb_ces_flat_filled["drv"] = tb_ces_flat_filled["drv"].astype(float)
        tb_ces_flat_filled["shr"] = tb_ces_flat_filled["shr"].astype(float)
        tb_ces_flat_filled["pwr"] = tb_ces_flat_filled["pwr"].astype(float)
        tb_ces_flat_filled["qty"] = tb_ces_flat_filled["qty"].astype(float)
        tb_ces_flat_filled["wge"] = tb_ces_flat_filled["wge"].astype(float)

        ar_st_coln_supply = [
            cme_inpt_convert.cme_dictkey_pdvarname(),
            'wkr', 'occ',
            'itc', 'slp',
            'wge', 'qty',
            'qtp',
            'lyr']
        tb_supply_lgt_filled = tb_supply_lgt_filled.reindex(columns=ar_st_coln_supply)
        tb_supply_lgt_filled["qtp"] = tb_supply_lgt_filled["qtp"].astype(float)
        tb_supply_lgt_filled["itc"] = tb_supply_lgt_filled["itc"].astype(float)
        tb_supply_lgt_filled["slp"] = tb_supply_lgt_filled["slp"].astype(float)
        tb_supply_lgt_filled["wge"] = tb_supply_lgt_filled["wge"].astype(float)
        tb_supply_lgt_filled["qty"] = tb_supply_lgt_filled["qty"].astype(float)

        ar_splv_totl_acrs_i_float = ar_splv_totl_acrs_i.astype(float)

        dc_ces_flat_here = cme_inpt_convert.cme_convert_pd2dc(tb_ces_flat_filled, input_type='demand')
        dc_supply_lgt_here = cme_inpt_convert.cme_convert_pd2dc(tb_supply_lgt_filled, input_type='supply')
        dc_inpt_gateway = {'fl_output_target': fl_output_target,
                           'dc_ces_flat': dc_ces_flat_here,
                           'ar_splv_totl_acrs_i': ar_splv_totl_acrs_i_float,
                           'dc_supply_lgt': dc_supply_lgt_here}

        # J. Solve the model with paraemters loaded in from CSV file
        start = timeit.default_timer()
        dc_ces_flat, dc_supply_lgt, \
            dc_equi_solv_sfur, dc_equi_solve_nest_info, _, _ = \
            cme_equi_solve_nest.cme_equi_solve_nest_test(
                dc_inpt_gateway=dc_inpt_gateway, verbose=verbose_debug)
        stop = timeit.default_timer()

        # max-output given other parameters
        fl_ces_output_max = dc_equi_solve_nest_info["fl_ces_output_max"]
        bl_failed = dc_equi_solve_nest_info['failed']

        if verbose:
            print('Time: ', stop - start)
            print(f'Finished with {it_year=}')
            print(f'{fl_ces_output_max=}')

        # K. outputs as tables
        df_supply_lgt = cme_inpt_convert.cme_convert_dc2pd(
            dc_supply_lgt, input_type="supply")
        df_ces_flat = cme_inpt_convert.cme_convert_dc2pd(
            dc_ces_flat, input_type="demand")

        return df_supply_lgt, df_ces_flat, pd_yfsa_mesh_curyr, \
            fl_ces_output_max, bl_failed

    # 7.2 year by year solution
    ls_df_supply_lgt = []
    ls_df_ces_flat = []
    ls_bl_failed = []
    # for it_year in np.arange(it_year_start, it_year_end + 1, step=1):
    for it_year in ar_yr_solve:
        df_supply_lgt, df_ces_flat, pd_yfsa_mesh_curyr, \
            fl_ces_output_max, bl_failed = \
            cut_merge_solve_equi_annual(it_year=it_year)
        df_supply_lgt['year'] = it_year
        df_ces_flat['year'] = it_year
        ls_bl_failed.append(bl_failed)
        ls_df_supply_lgt.append(df_supply_lgt)
        ls_df_ces_flat.append(df_ces_flat)

    df_ces_flat_jnt = pd.concat(ls_df_ces_flat)
    df_supply_lgt_jnt = pd.concat(ls_df_supply_lgt)

    # 7.3. Store stacked dataframes to folder
    # Files to export to csv
    # to_csv index=False important to elimiate index column
    if verbose_outfile:
        snm_file_demand = st_demand_file_prefix + '_myear.csv'
        spn_csv_path = os.path.join(os.sep, spt_data, snm_file_demand)
        df_ces_flat_jnt.to_csv(spn_csv_path, sep=",", index=False)
        print(f'{spn_csv_path=}')
        snm_file_supply = st_supply_file_prefix + '_myear.csv'
        spn_csv_path = os.path.join(os.sep, spt_data, snm_file_supply)
        df_supply_lgt_jnt.to_csv(spn_csv_path, sep=",", index=False)
        print(f'{spn_csv_path=}')

    # 8. Merge supply solution with observed quantity and price file
    # Load in year-occ-edu-sex-age-specific observable dataframe with quantity and price
    spn_path_dataset1 = os.path.join(os.sep, spt_data, 'dataset1.csv')
    spn_path_dataset1 = os.path.abspath(spn_path_dataset1)
    pd_dataset1 = pd.read_csv(spn_path_dataset1)

    spn_path_dataset1aux = os.path.join(os.sep, spt_data, 'dataset1_aux_2merge.csv')
    spn_path_dataset1aux = os.path.abspath(spn_path_dataset1aux)
    pd_dataset1_aux = pd.read_csv(spn_path_dataset1aux)

    # We do not match at home data, it is the residual
    # NaN for wage if at home
    pd_dataset1 = pd.merge(pd_dataset1, pd_dataset1_aux,
                           on=['category'])
    # Delete at home data, keep only no at home (meaning wokring)
    ar_bl_notathome = pd_dataset1['home'] == 0
    pd_dataset1 = pd_dataset1.loc[ar_bl_notathome]
    pd_dataset1 = pd_dataset1.drop(["home"], axis=1)
    # Delete non-graduate data, keep graduates
    ar_bl_graduate = pd_dataset1['graduate'] == 1
    pd_dataset1 = pd_dataset1.loc[ar_bl_graduate]
    pd_dataset1 = pd_dataset1.drop(["graduate"], axis=1)
    # Number of workers size adjustments
    pd_dataset1["numberworkers"] = pd_dataset1["numberworkers"] / fl_qtp_adj

    # Add the occ and wkr keys to the dataset1 file
    df_ow_trans_aff = pd_yfsa_mesh_curyr[
        ["occ", "wkr", "age", "female", "flexible"]]
    pd_dataset1 = pd.merge(pd_dataset1, df_ow_trans_aff,
                           on=["age", "female", "flexible"])

    # Now we merge dataset1 (observed) with prediction data (supply-side)
    # use supply-side file because it has qtp
    # pvd = predict vs data
    df_pvd = pd.merge(df_supply_lgt_jnt, pd_dataset1,
                      on=["year", "occ", "wkr"])

    # Generate diff with objective year by year and accumulate up
    ar_agg_diff_all_years = np.full([len(ar_yr_solve) + 1, ], np.nan)
    mt_agg_diff = np.full([len(ar_yr_solve) + 1, 2], np.nan)
    fl_failed_diff = 1e3
    for it_ctr, it_year in enumerate(ar_yr_solve):
        df_pvd_curyear = df_pvd.loc[df_pvd['year'] == it_year]
        df_pvd_curyear = df_pvd_curyear.reset_index()
        fl_agg_diff, ar_agg_diff = \
            cme_esti_obj.cme_esti_obj_pvd(df_pvd_curyear, verbose=verbose_debug)
        bl_failed = ls_bl_failed[it_ctr]
        if bl_failed:
            ar_agg_diff_all_years[it_ctr] = fl_failed_diff
        else:
            ar_agg_diff_all_years[it_ctr] = fl_agg_diff
        mt_agg_diff[it_ctr,] = ar_agg_diff

    # all year residual added up
    fl_agg_diff_all_years = np.nansum(ar_agg_diff_all_years)
    ar_agg_diff_all_years[it_ctr + 1] = fl_agg_diff_all_years
    mt_agg_diff[it_ctr + 1,] = np.nansum(mt_agg_diff, axis=0)
    ar_yr_solve = np.append(ar_yr_solve, -1)
    if verbose:
        pd_diff = pd.DataFrame(
            np.column_stack((ar_yr_solve, ar_agg_diff_all_years, mt_agg_diff)),
            index=ar_yr_solve,
            columns=["year", "fl_agg_diff", "wge_d_ls_w", "qty_d_ls_we"],
        )

        print(f'pd_diff displayed below')
        print(f'{pd_diff}')
        print(f'{fl_agg_diff_all_years=}')

    # 11. export fit file
    if verbose_outfile:
        snm_file_supply_fit = st_supply_file_prefix + '_myear_fit.csv'
        spn_csv_path = os.path.join(os.sep, spt_data, snm_file_supply_fit)
        pd_diff.to_csv(spn_csv_path, sep=",", index=False)

        snm_file_supply_wthobs = st_supply_file_prefix + '_myear_wthobs.csv'
        spn_csv_path = os.path.join(os.sep, spt_data, snm_file_supply_wthobs)
        df_pvd.to_csv(spn_csv_path, sep=",", index=False)
        print(f'{spn_csv_path=}')

    return fl_agg_diff_all_years, mt_agg_diff


if it_which_calibrate == 0:
    # Simulate results, save to file
    ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=1)
    # ar_yr_solve = np.arange(it_year_start, it_year_start + 1, step=1)

    fl_wge_slp = 0.998
    fl_flx_power = 1 / (1 - 0.337)
    fl_gender_power = 1 / (1 - 0.700)
    fl_age_power = 1 / (1 - 0.974)
    ar_alpha1 = [-1.4379, -0.0361, 0.2447, -0.0450]
    fl_alpha2_female_inf = 0.43974978
    fl_alpha2_female_flx = 0.418877039
    ar_a_demand_age_flx_sex = [-0.000214152, 0.009805873, 0.099267859]
    ar_b_demand_age_flx_sex = [-0.000126304, 0.005343194, 0.101534043]
    ar_c_demand_age_flx_sex = [-0.000151123, 0.007077408, 0.12976847]
    ar_d_demand_age_flx_sex = [-3.96081E-05, 0.001619739, 0.125952938]

    ar_a_supply_year_flx_sex = [0.004009086, -0.106047372, 0.312336957]
    ar_b_supply_year_flx_sex = [0.001557121, -0.057408455, 0.927357343]
    ar_c_supply_year_flx_sex = [0.006603487, -0.169550854, 0.024052461]
    ar_d_supply_year_flx_sex = [0.002740785, -0.061852106, 0.029642648]

    if it_param_simu == 0:
        # These are suppose to be generate the same predictions as the 2-62 results
        fl_alpha2_female_inf = 0.43974978
        fl_alpha2_female_flx = 0.418877039
        fl_gender_power = 0.974
        fl_age_power = 0.974

    elif it_param_simu == 1:
        # run "it_which_calibrate=1" first time
        # Based on results from test-cdc1-2-2-31-agg-diff-genderagepower-20230426-200709
        fl_alpha2_female_inf = 0.45
        fl_alpha2_female_flx = 0.416666667
        fl_gender_power = 0.99
        fl_age_power = 0.99

    elif it_param_simu == 2:
        # run "it_which_calibrate=2" first time given 1 "it_which_calibrate=1" result
        # Based on results from test-cdc2-2-2-31-agg-diff-gendershare-20230426-221639
        # obj = 0.499451204, not an improvement from baseline simu

        fl_alpha2_female_inf = 0.45
        fl_alpha2_female_flx = 0.383333333
        fl_gender_power = 0.99
        fl_age_power = 0.19

    elif it_param_simu == 3:
        # Failed to run, yz problem error break

        fl_alpha2_female_inf = 4.470e-01
        fl_alpha2_female_flx = 3.839e-01
        fl_gender_power = 0.99
        fl_age_power = 1.326e-01

        ar_params = np.array(
            [1.30407872e+00, 1.41296584e+00, 4.89551827e+00, 4.18073521e+01,
             -1.54620587e+00, -3.40227748e-02, 2.57932441e-01, -4.05461321e-02,
             4.40883373e-01, 4.03566026e-01, -1.85635703e-04, 8.43651884e-03,
             9.04710458e-02, -1.20406849e-04, 4.77960181e-03, 1.08538928e-01,
             -1.53453661e-04, 6.34378587e-03, 1.64743872e-01, -3.62410113e-05,
             1.48088619e-03, 7.04108388e-02])
        fl_wge_slp = ar_params[0]
        fl_flx_power = ar_params[1]
        fl_gender_power = ar_params[2]
        fl_age_power = ar_params[3]
        ar_alpha1 = [ar_params[4], ar_params[5], ar_params[6], ar_params[7]]
        fl_alpha2_female_inf = ar_params[8]
        fl_alpha2_female_flx = ar_params[9]
        ar_a_demand_age_flx_sex = [ar_params[10], ar_params[11], ar_params[12]]
        ar_b_demand_age_flx_sex = [ar_params[13], ar_params[14], ar_params[15]]
        ar_c_demand_age_flx_sex = [ar_params[16], ar_params[17], ar_params[18]]
        ar_d_demand_age_flx_sex = [ar_params[19], ar_params[20], ar_params[21]]

    elif it_param_simu == 4:
        # run "it_which_calibrate=4", did not wait until estimation to finish

        fl_alpha2_female_inf = 0.43974978
        fl_alpha2_female_flx = 0.418877039
        fl_gender_power = 0.99
        fl_age_power = 0.99
        ar_a_supply_year_flx_sex = [0.00461428, -0.11471043, 0.32748848]
        ar_b_supply_year_flx_sex = [0.00169116, -0.04543103, 0.86851846]
        ar_c_supply_year_flx_sex = [0.00622025, -0.16343688, 0.03022742]
        ar_d_supply_year_flx_sex = [0.00226091, -0.05560411, 0.03151853]

    verbose = True
    verbose_debug = False
    verbose_outfile = True

    fl_agg_diff_all_years, mt_agg_diff = \
        ffi_esti_gender_params(
            ar_yr_solve=ar_yr_solve,
            fl_wge_slp=fl_wge_slp,
            fl_flx_power=fl_flx_power,
            fl_gender_power=fl_gender_power,
            fl_age_power=fl_age_power,
            ar_alpha1=ar_alpha1,
            fl_alpha2_female_inf=fl_alpha2_female_inf,
            fl_alpha2_female_flx=fl_alpha2_female_flx,
            ar_a_demand_age_flx_sex=ar_a_demand_age_flx_sex,
            ar_b_demand_age_flx_sex=ar_b_demand_age_flx_sex,
            ar_c_demand_age_flx_sex=ar_c_demand_age_flx_sex,
            ar_d_demand_age_flx_sex=ar_d_demand_age_flx_sex,
            ar_a_supply_year_flx_sex=ar_a_supply_year_flx_sex,
            ar_b_supply_year_flx_sex=ar_b_supply_year_flx_sex,
            ar_c_supply_year_flx_sex=ar_c_supply_year_flx_sex,
            ar_d_supply_year_flx_sex=ar_d_supply_year_flx_sex,
            verbose=verbose,
            verbose_debug=verbose_debug,
            verbose_outfile=verbose_outfile)

if it_which_calibrate == 1:
    # B. given prior paraemters, ddoes changing elasticity improve predictions
    # fixing age elasticity at prior estimates
    print(f'\n\n\nSimuate all years\n\n')
    ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=1)
    snm_file_agg_diff = \
        st_file_prefix + '-agg-diff-genderagepower-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
    print(f'{it_which_calibrate=}')
    print(f'Output file name = {snm_file_agg_diff}')

    # These are suppose to be generate the same predictions as the 2-62 results
    fl_alpha2_female_inf = 0.43974978
    fl_alpha2_female_flx = 0.418877039

    it_gender_power_grid = 10
    fl_gender_power_min = 0.09
    fl_gender_power_max = 0.99
    it_age_power_grid = 10
    fl_age_power_min = 0.09
    fl_age_power_max = 0.99

    ar_fl_gender_power_grid = np.linspace(
        fl_gender_power_min, fl_gender_power_max, num=it_gender_power_grid)
    ar_fl_age_power_grid = np.linspace(
        fl_age_power_min, fl_age_power_max, num=it_age_power_grid)

    mt_fl_agg_diff = np.random.rand(it_gender_power_grid, it_age_power_grid)
    mt_fl_agg_diff = mt_fl_agg_diff - mt_fl_agg_diff

    for it_gender_power, fl_gender_power in enumerate(ar_fl_gender_power_grid):
        for it_age_power, fl_age_power in enumerate(ar_fl_age_power_grid):
            print(f'idx={it_gender_power} and val={fl_gender_power}')
            print(f'idx={it_age_power} and val={fl_age_power}')

            # Solve
            fl_agg_diff_all_years, mt_agg_diff = \
                ffi_esti_gender_params(
                    ar_yr_solve=ar_yr_solve,
                    fl_alpha2_female_inf=fl_alpha2_female_inf,
                    fl_alpha2_female_flx=fl_alpha2_female_flx,
                    fl_gender_power=fl_gender_power,
                    fl_age_power=fl_age_power,
                    verbose=False, verbose_debug=False, verbose_outfile=False)

            print(f'{fl_agg_diff_all_years=}')

            # Store objective
            mt_fl_agg_diff[it_gender_power, it_age_power] = fl_agg_diff_all_years

            print(f'{mt_fl_agg_diff=}')
            spn_csv_path = os.path.join(os.sep, spt_data, snm_file_agg_diff)
            pd.DataFrame(mt_fl_agg_diff,
                         index=ar_fl_gender_power_grid,
                         columns=ar_fl_age_power_grid,
                         ).to_csv(spn_csv_path, sep=",", index=True)
            print(f'{spn_csv_path=}')

if it_which_calibrate == 2:
    # Adjust for gender share
    # Separate gender share parameters for flexible and inflexible works
    ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=1)
    # ar_yr_solve = np.arange(it_year_start, it_year_start + 1, step=1)
    snm_file_agg_diff = \
        st_file_prefix + '-agg-diff-gendershare-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
    print(f'{it_which_calibrate=}')
    print(f'Output file name = {snm_file_agg_diff}')

    it_alpha2_grid = 10
    fl_alpha2_min = 0.35
    fl_alpha2_max = 0.65

    if it_which_calibrate == 2:
        fl_gender_power = 0.99
        fl_age_power = 0.99

    ar_fl_alpha2_female_inf_grid = np.linspace(
        fl_alpha2_min, fl_alpha2_max, num=it_alpha2_grid)
    ar_fl_alpha2_female_flx_grid = np.linspace(
        fl_alpha2_min, fl_alpha2_max, num=it_alpha2_grid)
    mt_fl_agg_diff = np.random.rand(it_alpha2_grid, it_alpha2_grid)
    mt_fl_agg_diff = mt_fl_agg_diff - mt_fl_agg_diff

    for it_alpha2_female_inf, fl_alpha2_female_inf in enumerate(ar_fl_alpha2_female_inf_grid):
        for it_alpha2_female_flx, fl_alpha2_female_flx in enumerate(ar_fl_alpha2_female_flx_grid):
            print(f'Row idx={it_alpha2_female_inf} and '
                  f'Col idx={it_alpha2_female_flx}')

            # Solve
            fl_agg_diff_all_years, mt_agg_diff = \
                ffi_esti_gender_params(
                    ar_yr_solve=ar_yr_solve,
                    fl_alpha2_female_inf=fl_alpha2_female_inf,
                    fl_alpha2_female_flx=fl_alpha2_female_flx,
                    fl_gender_power=fl_gender_power,
                    fl_age_power=fl_age_power,
                    verbose=False, verbose_debug=False, verbose_outfile=False)
            print(f'{fl_agg_diff_all_years=}')

            # Store objective
            mt_fl_agg_diff[it_alpha2_female_inf, it_alpha2_female_flx] = fl_agg_diff_all_years
            print(f'{mt_fl_agg_diff=}')

            # Save to table
            spn_csv_path = os.path.join(os.sep, spt_data, snm_file_agg_diff)
            pd.DataFrame(mt_fl_agg_diff,
                         index=ar_fl_alpha2_female_inf_grid,
                         columns=ar_fl_alpha2_female_flx_grid
                         ).to_csv(spn_csv_path, sep=",", index=True)
            print(f'{spn_csv_path=}')

if it_which_calibrate == 3:
    # Search over all demand side parameters, jointly (except for y/z ratio)
    ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=1)

    def ffi_fh_esti_gender_params(ar_params):
        fl_wge_slp = ar_params[0]
        fl_flx_power = ar_params[1]
        fl_gender_power = ar_params[2]
        fl_age_power = ar_params[3]
        ar_alpha1 = [ar_params[4], ar_params[5], ar_params[6], ar_params[7]]
        fl_alpha2_female_inf = ar_params[8]
        fl_alpha2_female_flx = ar_params[9]
        ar_a_demand_age_flx_sex = [ar_params[10], ar_params[11], ar_params[12]]
        ar_b_demand_age_flx_sex = [ar_params[13], ar_params[14], ar_params[15]]
        ar_c_demand_age_flx_sex = [ar_params[16], ar_params[17], ar_params[18]]
        ar_d_demand_age_flx_sex = [ar_params[19], ar_params[20], ar_params[21]]
        ar_z_nonparam = [
            ar_params[22], ar_params[23], ar_params[24], ar_params[25], ar_params[26],
            ar_params[27], ar_params[28], ar_params[29], ar_params[30], ar_params[31],
            ar_params[32], ar_params[33], ar_params[34], ar_params[35], ar_params[36],
            ar_params[37], ar_params[38], ar_params[39], ar_params[40], ar_params[41],
            ar_params[42], ar_params[43], ar_params[44], ar_params[45], ar_params[46]]

        # fl_alpha2_female_inf = 0.43974978
        # fl_alpha2_female_flx = 0.418877039
        # fl_gender_power = 0.99
        print(f'{ar_params=}')

        fl_agg_diff_all_years, mt_agg_diff = \
            ffi_esti_gender_params(
                ar_yr_solve=ar_yr_solve,
                fl_wge_slp=fl_wge_slp,
                fl_flx_power=fl_flx_power,
                fl_gender_power=fl_gender_power,
                fl_age_power=fl_age_power,
                ar_alpha1=ar_alpha1,
                fl_alpha2_female_inf=fl_alpha2_female_inf,
                fl_alpha2_female_flx=fl_alpha2_female_flx,
                ar_a_demand_age_flx_sex=ar_a_demand_age_flx_sex,
                ar_b_demand_age_flx_sex=ar_b_demand_age_flx_sex,
                ar_c_demand_age_flx_sex=ar_c_demand_age_flx_sex,
                ar_d_demand_age_flx_sex=ar_d_demand_age_flx_sex,
                ar_z_nonparam = ar_z_nonparam,
                verbose=False, verbose_debug=False, verbose_outfile=False)

        print(f'{fl_agg_diff_all_years=}')

        return fl_agg_diff_all_years


    fl_wge_slp = 0.998
    fl_flx_power = 1 / (1 - 0.337)
    fl_gender_power = 1 / (1 - 0.700)
    fl_age_power = 1 / (1 - 0.974)
    ar_alpha1 = [-1.4379, -0.0361, 0.2447, -0.0450]
    fl_alpha2_female_inf = 0.43974978
    fl_alpha2_female_flx = 0.418877039
    ar_a_dmd_ageflxsex = [-0.000214152, 0.009805873, 0.099267859]
    ar_b_dmd_ageflxsex = [-0.000126304, 0.005343194, 0.101534043]
    ar_c_dmd_ageflxsex = [-0.000151123, 0.007077408, 0.12976847]
    ar_d_dmd_ageflxsex = [-3.96081E-05, 0.001619739, 0.125952938]
    ar_z_npar = [1.9783073, 1.88633003, 2.02298967, 1.57471996, 2.17705674,
                 2.3014033, 2.42368213, 2.52749721, 2.58223651, 2.67127097,
                 2.76696538, 2.14991199, 2.15558105, 2.99842456, 3.11969558,
                 2.35282069, 3.29131954, 3.46537009, 3.62964579, 3.8084693,
                 3.9078983, 3.95828349, 4.07143947, 4.18997515, 4.2301114]

    ar_params_init = (
        fl_wge_slp,
        fl_flx_power, fl_gender_power, fl_age_power,
        ar_alpha1[0], ar_alpha1[1], ar_alpha1[2], ar_alpha1[3],
        fl_alpha2_female_inf, fl_alpha2_female_flx,
        ar_a_dmd_ageflxsex[0], ar_a_dmd_ageflxsex[1], ar_a_dmd_ageflxsex[2],
        ar_b_dmd_ageflxsex[0], ar_b_dmd_ageflxsex[1], ar_b_dmd_ageflxsex[2],
        ar_c_dmd_ageflxsex[0], ar_c_dmd_ageflxsex[1], ar_c_dmd_ageflxsex[2],
        ar_d_dmd_ageflxsex[0], ar_d_dmd_ageflxsex[1], ar_d_dmd_ageflxsex[2],
        ar_z_npar[0], ar_z_npar[1], ar_z_npar[2], ar_z_npar[3], ar_z_npar[4],
        ar_z_npar[5], ar_z_npar[6], ar_z_npar[7], ar_z_npar[8], ar_z_npar[9],
        ar_z_npar[10], ar_z_npar[11], ar_z_npar[12], ar_z_npar[13], ar_z_npar[14],
        ar_z_npar[15], ar_z_npar[16], ar_z_npar[17], ar_z_npar[18], ar_z_npar[19],
        ar_z_npar[20], ar_z_npar[21], ar_z_npar[22], ar_z_npar[23], ar_z_npar[24])

    res = sp.optimize.minimize(
        ffi_fh_esti_gender_params,
        ar_params_init,
        method='Nelder-Mead',
        tol=1e-4)
    print(res)

if it_which_calibrate == 4:
    # Search over four parameters, using optimization routine
    ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=1)

    def ffi_fh_esti_gender_params(ar_params):
        fl_gender_power = 0.99
        fl_age_power = 0.99
        fl_alpha2_female_inf = 0.43974978
        fl_alpha2_female_flx = 0.418877039

        ar_a_supply_year_flx_sex = [ar_params[0], ar_params[1], ar_params[2]]
        ar_b_supply_year_flx_sex = [ar_params[3], ar_params[4], ar_params[5]]
        ar_c_supply_year_flx_sex = [ar_params[6], ar_params[7], ar_params[8]]
        ar_d_supply_year_flx_sex = [ar_params[9], ar_params[10], ar_params[11]]

        # ar_a_supply_year_flx_sex = [0.004009086, -0.106047372, 0.312336957],
        # ar_b_supply_year_flx_sex = [0.001557121, -0.057408455, 0.927357343],
        # ar_c_supply_year_flx_sex = [0.006603487, -0.169550854, 0.024052461],
        # ar_d_supply_year_flx_sex = [0.002740785, -0.061852106, 0.029642648],

        # fl_alpha2_female_inf = 0.43974978
        # fl_alpha2_female_flx = 0.418877039
        # fl_gender_power = 0.99
        print(f'{ar_params=}')

        fl_agg_diff_all_years, mt_agg_diff = \
            ffi_esti_gender_params(
                ar_yr_solve=ar_yr_solve,
                fl_alpha2_female_inf=fl_alpha2_female_inf,
                fl_alpha2_female_flx=fl_alpha2_female_flx,
                fl_gender_power=fl_gender_power,
                fl_age_power=fl_age_power,
                ar_a_supply_year_flx_sex=ar_a_supply_year_flx_sex,
                ar_b_supply_year_flx_sex=ar_b_supply_year_flx_sex,
                ar_c_supply_year_flx_sex=ar_c_supply_year_flx_sex,
                ar_d_supply_year_flx_sex=ar_d_supply_year_flx_sex,
                verbose=False, verbose_debug=False, verbose_outfile=False)

        print(f'{fl_agg_diff_all_years=}')

        return fl_agg_diff_all_years


    ar_parms_init = (
        0.004009086, -0.106047372, 0.312336957,
        0.001557121, -0.057408455, 0.927357343,
        0.006603487, -0.169550854, 0.024052461,
        0.002740785, -0.061852106, 0.029642648)

    res = sp.optimize.minimize(
        ffi_fh_esti_gender_params,
        ar_parms_init,
        method='Nelder-Mead',
        tol=1e-6)

    print(res)

# if it_which_calibrate == 3:
#     # C. Iterate and find best power parameter, across male and female, and across age groups
#     snm_file_agg_diff = st_file_prefix + '-agg-diff-genderagepower-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
#     spn_csv_path = os.path.join(os.sep, spt_data, snm_file_agg_diff)
#     print(f'{spn_csv_path=}')
#
#     print(f'\n\n\nSimuate all years\n\n')
#     ar_yr_solve = np.arange(it_year_start, it_year_end + 1, step=1)
#     # print(f'\n\n\nSimuate less years\n\n')
#     # ar_yr_solve = np.arange(it_year_start, it_year_start + 1, step=1)
#
#     fl_alpha2_female_inf = 0.50
#     fl_alpha2_female_flx = 0.50
#
#     it_gender_power_grid = 4
#     fl_gender_power_min = 0.985
#     fl_gender_power_max = 0.999
#     it_age_power_grid = 4
#     fl_age_power_min = 0.40
#     fl_age_power_max = 0.60
#
#     ar_fl_gender_power_grid = np.linspace(
#         fl_gender_power_min, fl_gender_power_max, num=it_gender_power_grid)
#     ar_fl_age_power_grid = np.linspace(
#         fl_age_power_min, fl_age_power_max, num=it_age_power_grid)
#     mt_fl_agg_diff = np.random.rand(it_gender_power_grid, it_age_power_grid)
#     mt_fl_agg_diff = mt_fl_agg_diff - mt_fl_agg_diff
#
#     it_ctr = 0
#     for it_gender_power, fl_gender_power in enumerate(ar_fl_gender_power_grid):
#         for it_age_power, fl_age_power in enumerate(ar_fl_age_power_grid):
#             it_ctr = it_ctr + 1
#             start = timeit.default_timer()
#             print(f'{it_gender_power=} and {fl_gender_power=}\n'
#                   f'{it_age_power=} and {fl_age_power=}')
#
#             # Solve
#             fl_agg_diff, ar_agg_diff, fl_ces_output_max = \
#                 ffi_esti_gender_params(
#                     ar_yr_solve=ar_yr_solve,
#                     fl_alpha2_female_inf=fl_alpha2_female_inf,
#                     fl_alpha2_female_flx=fl_alpha2_female_flx,
#                     fl_gender_power=fl_gender_power,
#                     fl_age_power=fl_age_power,
#                     verbose=False, verbose_debug=False, verbose_outfile=False)
#
#             print(f'{fl_agg_diff=}')
#
#             # Store objective
#             mt_fl_agg_diff[it_gender_power, it_age_power] = fl_agg_diff
#
#             print(f'Display mt_fl_agg_diff\n'
#                   f'{np.round(mt_fl_agg_diff, 4)}')
#
#             pd.DataFrame(mt_fl_agg_diff,
#                          index=ar_fl_gender_power_grid,
#                          columns=ar_fl_age_power_grid
#                          ).to_csv(spn_csv_path, sep=",", index=True)
#
#             stop = timeit.default_timer()
#             print(f'Time={stop - start}, completed {it_ctr} of {mt_fl_agg_diff.size}')
#
#     print(f'{spn_csv_path=}')
