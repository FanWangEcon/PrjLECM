# Optimizer Support functions
import numpy as np


def cme_opti_nu_func(slope, omega, nu):
    return slope*np.log(1-nu) - np.log(nu) - omega


def cme_opti_nu_bisect(fl_slope, fl_omega,
                       fl_nu_min=1e-5, fl_nu_max=1-1e-5,
                       fl_nu_tol=1e-6, it_n_max=20):
    it_n = 1
    fl_nu_low = fl_nu_min
    fl_nu_high = fl_nu_max
    while (it_n <= it_n_max):
        fl_nu_new = (fl_nu_low + fl_nu_high)/2
        fl_obj = cme_opti_nu_func(fl_slope, fl_omega, fl_nu_new)
        if fl_obj == 0 or (fl_nu_high - fl_nu_low)/2 < fl_nu_tol:
            break
        else:
            it_n = it_n + 1
            if (fl_obj > 0):
                fl_nu_low = fl_nu_new
            else:
                fl_nu_high = fl_nu_new

    return fl_nu_new, fl_obj


def cme_opti_nu1_bisect(ffi_output, fl_output_target,
                        fl_nu1_min=1e-5,
                        fl_nu1_tol=1e-6, it_n_max=20):
    # note nu is leisure time now, nu_{i=1} specifically, low measures more work
    it_n = 1
    fl_nu1_max = 1 - fl_nu1_min
    fl_nu_low = fl_nu1_min
    fl_nu_high = fl_nu1_max
    while (it_n <= it_n_max):
        fl_nu1_new = (fl_nu_low + fl_nu_high)/2
        dc_equi_solv_sfur = ffi_output(fl_nu1_new)
        fl_ces_output = dc_equi_solv_sfur["fl_ces_output"]
        fl_obj = fl_ces_output - fl_output_target
        if fl_obj == 0 or (fl_nu_high - fl_nu_low)/2 < fl_nu1_tol:
            break
        else:
            it_n = it_n + 1
            if (fl_obj > 0):
                # higher than target, working too much, leisure too low
                # increase leisure time
                fl_nu_low = fl_nu1_new
            else:
                fl_nu_high = fl_nu1_new

    return fl_nu1_new, fl_obj, dc_equi_solv_sfur


if __name__ == "__main__":
    ar_omega = np.array([0.5, 1, 3, 100])
    ar_coef_ln1misnui = np.array([0.1, 0.6, 0.3, 0.5])
    it_wkr_cnt = len(ar_omega)
    for it_wkr_ctr in np.arange(it_wkr_cnt):
        fl_omega = ar_omega[it_wkr_ctr]
        fl_slope = ar_coef_ln1misnui[it_wkr_ctr]
        fl_nu_new, fl_obj = cme_opti_nu_bisect(fl_slope, fl_omega)
        print(f'{fl_nu_new=} and {fl_obj=}')
