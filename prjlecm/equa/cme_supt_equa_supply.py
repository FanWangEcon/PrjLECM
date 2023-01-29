import numpy as np
# supply related functions

# sply = supply
# lgt = logit


def cme_hh_lgt_sprl_intr(alpha_i_j, alpha_i_1):
    return alpha_i_j - alpha_i_1


def cme_hh_lgt_sprl_slpe(beta_i):
    return beta_i


def cme_hh_lgt_wglv_j1(splv_i_j1, splv_i_j0, alpha_i_j1, beta_i):
    ffi_part1 = np.log(splv_i_j1/splv_i_j0) - alpha_i_j1
    wage_i_j1 = np.exp(ffi_part1/beta_i)
    return wage_i_j1


def cme_agg_splv_j1(splv_totl_wrki, nu_i, spsh_j1_of_totw):
    """Occupation 1 supply level for household type
    Aggregate supply level worker of type i in occupation j=1

    splv_totl_wrki = \hat{L}_i

    nu_i = share in leisure househoold type i
    """
    splv_working = splv_totl_wrki * (1-nu_i)
    sply_wrki_j1 = splv_working*spsh_j1_of_totw

    splv_wkri_j0 = splv_totl_wrki * nu_i

    return sply_wrki_j1, splv_wkri_j0


def cme_splv_lgt_solver(ar_price, ar_alpha_i,
                        fl_beta_i, fl_splv_totl_i):
    ar_v_hat = ar_alpha_i + fl_beta_i*np.log(ar_price)
    fl_denominator = 1 + np.sum(np.exp(ar_v_hat))
    ar_prob_occj = np.exp(ar_v_hat)/fl_denominator
    ar_splv_occj = fl_splv_totl_i*ar_prob_occj
    fl_splv_occ0 = fl_splv_totl_i*(1-np.sum(ar_prob_occj))
    ar_splv_i = np.insert(ar_splv_occj, 0, fl_splv_occ0)

    return ar_splv_i


if __name__ == "__main__":
    ar_splv_totl_wrki = np.array([1, 2, 3])
    ar_nu_i = np.array([0.4, 0.6, 0.11])
    ar_spsh_j1_of_totw = np.array([0.4, 0.5, 0.6])
    sply_wrki_j1 = cme_agg_splv_j1(
        ar_splv_totl_wrki, ar_nu_i, ar_spsh_j1_of_totw)
    print(f'{sply_wrki_j1=}')

    ar_price = np.array([1, 2, 3])
    ar_alpha_i = np.array([-1, 0.5, -0.25])
    fl_beta_i = 0.8
    fl_splv_totl_i = 1
    ar_splv_i = cme_splv_lgt_solver(ar_price, ar_alpha_i,
                                    fl_beta_i, fl_splv_totl_i)

    ar_price = np.array([1.61381634, 1.36445694, 1.48217059, 1.4994783 ])
    ar_alpha_i = np.array([-1.23612334, -0.88511407, -0.29177873, -1.13731607])
    fl_beta_i = 0.5254217290179863
    fl_splv_totl_i = 1.8087352766430267
    ar_splv_i = cme_splv_lgt_solver(ar_price, ar_alpha_i,
                                    fl_beta_i, fl_splv_totl_i)
