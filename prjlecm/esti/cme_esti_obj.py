import numpy as np


def cme_esti_obj_pvd(df_pvd, verbose=True):
    """Evaluate data and model prediction fit

    Year by year. Model and data wages relative to one model and one data
    wage relatively, comparison of relative ratios. Wage scale is not
    determined. Quantity scale is determined through potential labor
    level, model and data relative to one data quantity level.

    Relative ratio for quantity and wages, putting them in the same scale.

    Parameters
    ----------
    df_pvd:
        Dataframe including all model prediction and observabed data
    verbose

    Returns
    -------

    """
    df_pvd[["qty_r"]] = \
        df_pvd[["qty"]] / float(df_pvd.loc[0, ["numberworkers"]])
    df_pvd[["numberworkers_r"]] = \
        df_pvd[["numberworkers"]] / float(df_pvd.loc[0, ["numberworkers"]])
    df_pvd[["qty_d_l"]] = np.abs(
        np.array(df_pvd[["qty_r"]]) - np.array(df_pvd[["numberworkers_r"]]))
    # d_l still called as this, but here measn relative difference level
    # relative to value in row one. rescale all
    df_pvd[["wge_r"]] = \
        df_pvd[["wge"]] / float(df_pvd.loc[0, ["wge"]])
    df_pvd[["meanwage_r"]] = \
        df_pvd[["meanwage"]] / float(df_pvd.loc[0, ["meanwage"]])
    df_pvd[["wge_d_l"]] = np.abs(
        np.array(df_pvd[["wge_r"]]) - np.array(df_pvd[["meanwage_r"]]))

    # Get total mass
    fl_total_workers = np.sum(df_pvd[["numberworkers"]])

    # weighted d_ls_w = weighed differenced in levels squares
    df_pvd[["wge_d_ls_w"]] = (
            (np.array(df_pvd[["numberworkers"]]) / np.array(fl_total_workers)) *
            (np.array(df_pvd[["wge_d_l"]]) ** 2))
    df_pvd[["qty_d_ls_w"]] = (
            (np.array(df_pvd[["numberworkers"]]) / np.array(fl_total_workers)) *
            (np.array(df_pvd[["qty_d_l"]]) ** 2))
    # quantity does not really need to be quantity weighted, but needs even weight
    df_pvd[["qty_d_ls_we"]] = (
            (1 / len(df_pvd[["numberworkers"]])) *
            (np.array(df_pvd[["qty_d_l"]]) ** 2))

    # qty_d_ls_w sum by subgroups
    # by weight by population,
    # differences for more population groups matters more
    df_agg_diff_flx_sex = df_pvd[
        ["flexible", "female", "wge_d_ls_w", "qty_d_ls_w", "qty_d_ls_we"]].groupby(
        ["flexible", "female"]
    ).sum()
    df_agg_diff_age = df_pvd[
        ["age", "wge_d_ls_w", "qty_d_ls_w", "qty_d_ls_we"]].groupby(
        ["age"]
    ).sum()
    df_agg_diff_year = df_pvd[
        ["year", "wge_d_ls_w", "qty_d_ls_w", "qty_d_ls_we"]].groupby(
        ["year"]
    ).sum()

    if verbose:
        print(df_agg_diff_flx_sex)
        print(df_agg_diff_age)
        print(df_agg_diff_year)

    # 10. Total difference objective
    ar_agg_diff = np.array(df_pvd[["wge_d_ls_w", "qty_d_ls_we"]].sum())
    fl_agg_diff = np.sum(ar_agg_diff)
    if verbose:
        print(f'{ar_agg_diff=}')
        print(f'{fl_agg_diff=}')

    return fl_agg_diff, ar_agg_diff
