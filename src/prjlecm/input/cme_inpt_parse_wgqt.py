import numpy as np
import pandas as pd
import prjlecm.input.cme_inpt_parse as cme_inpt_parse

def cme_parse_wgqt_pd2dc(fl_aggregate_output, dc_demand_ces, dc_supply_lgt, pd_wglv_all, pd_qtlv_all):
    """
    Update demand and supply dictionaries with equilibrium wages and quantities.
    After solving for equilibrium prices and quantities, this function extracts the 
    equilibrium wage and quantity outcomes from the solution dataframes and populates 
    them into the corresponding demand and supply nested CES/logit dictionaries.
    Parameters
    ----------
    dc_demand_ces : dict
        Nested dictionary structure containing demand-side CES parameters and specifications.
        Updated in-place with equilibrium wages at the lowest layer.
    dc_supply_lgt : dict
        Nested dictionary structure containing supply-side logit parameters and specifications.
        Updated in-place with equilibrium wages and quantities at the lowest layer.
    pd_wglv_all : pd.DataFrame
        DataFrame containing equilibrium wage levels indexed by worker (rows) and 
        occupation (columns).
    pd_qtlv_all : pd.DataFrame
        DataFrame containing equilibrium quantity levels indexed by worker (rows) and 
        occupation (columns).
    Returns
    -------
    tuple of (dict, dict)
        Updated demand and supply dictionaries with equilibrium wages and quantities populated:
        - dc_demand_ces : dict with 'wge' key updated at leaf nodes
        - dc_supply_lgt : dict with 'wge' and 'qty' keys updated at leaf nodes
    """
    
    # After solving for equilibrium prices and quantities, the equilibrium outcomes are stores
    # in pd_wglv_all and pd_qtlv_all dataframes. 
    # Grab out the equilibrium wage and quantities, and update the demand and supply dictionaries with these equilibrium outcomes.
     
    # Update demand and supply dictionaries with equilibrium wage
    # if solutions are successful
    # DEMAND DICT: Populate solution wage to lowest level
    __, ls_maxlyr_demand_key, it_lyr0_key = cme_inpt_parse.cme_parse_demand_tbidx(
        dc_demand_ces)
    # fill aggregate output rquirement
    dc_demand_ces[it_lyr0_key]['qty'] = fl_aggregate_output
    for it_key_idx in ls_maxlyr_demand_key:
        # Get wkr and occ index for current child
        it_wkr_idx = dc_demand_ces[it_key_idx]['wkr']
        it_occ_idx = dc_demand_ces[it_key_idx]['occ']

        # wrk index matches with rows, 1st row is first worker, wkr index 0
        # occ + 0 because first column is NOT leisure
        fl_wglv = pd_wglv_all.iloc[it_wkr_idx, it_occ_idx + 0]
        # Replace highest/bottommost layer's qty terms with equi solution qs.s
        dc_demand_ces[it_key_idx]['wge'] = fl_wglv

    # SUPPLY DICT: population solution wage
    __, ls_maxlyr_supply_key, __ = cme_inpt_parse.cme_parse_demand_tbidx(
        dc_supply_lgt)
    for it_key_idx in ls_maxlyr_supply_key:
        it_wkr_idx = dc_supply_lgt[it_key_idx]['wkr']
        it_occ_idx = dc_supply_lgt[it_key_idx]['occ']
        fl_wglv = pd_wglv_all.iloc[it_wkr_idx, it_occ_idx + 0]
        fl_qtlv = pd_qtlv_all.iloc[it_wkr_idx, it_occ_idx + 1]
        dc_supply_lgt[it_key_idx]['wge'] = fl_wglv
        dc_supply_lgt[it_key_idx]['qty'] = fl_qtlv

    return dc_demand_ces, dc_supply_lgt

def cme_parse_qtwg_dc2pd_demand(dc_demand_ces, ar_splv_totl_acrs_i=None):
    """
    Obtain from the bottom-most layer, matrix of qty and wge.

    Rows are worker types and columns are occupation types.

    :param dc_ces_flat:
    :return:
    """

    __, ls_maxlyr_key, __ = cme_inpt_parse.cme_parse_demand_tbidx(
        dc_demand_ces)
    dc_parse_occ_wkr_lst = cme_inpt_parse.cme_parse_occ_wkr_lst(dc_demand_ces)
    it_worker_types = dc_parse_occ_wkr_lst['it_wkr_cnt']
    it_occ_types = dc_parse_occ_wkr_lst['it_occ_cnt']

    mt_qtlv_all = np.zeros([it_worker_types, it_occ_types + 1], dtype=float)
    mt_wglv_all = np.zeros([it_worker_types, it_occ_types], dtype=float)
    for it_key_idx in ls_maxlyr_key:
        it_occ = dc_demand_ces[it_key_idx]['occ']
        it_wkr = dc_demand_ces[it_key_idx]['wkr']

        fl_qty = dc_demand_ces[it_key_idx]['qty']
        mt_qtlv_all[it_wkr, it_occ + 1] = fl_qty
        fl_wge = dc_demand_ces[it_key_idx]['wge']
        mt_wglv_all[it_wkr, it_occ] = fl_wge

    if ar_splv_totl_acrs_i is not None:
        mt_qtlv_all[:, 0] = ar_splv_totl_acrs_i - np.sum(mt_qtlv_all, 1)

    pd_qtlv_all = pd.DataFrame(
        mt_qtlv_all,
        index=["i" + str(i + 1)
               for i in np.arange(np.shape(mt_qtlv_all)[0])],
        columns=["j" + str(j) for j in np.arange(np.shape(mt_qtlv_all)[1])]
    )

    pd_wglv_all = pd.DataFrame(
        mt_wglv_all,
        index=["i" + str(i + 1)
               for i in np.arange(np.shape(mt_wglv_all)[0])],
        columns=["j" + str(j + 1)
                 for j in np.arange(np.shape(mt_wglv_all)[1])]
    )

    return pd_qtlv_all, pd_wglv_all

