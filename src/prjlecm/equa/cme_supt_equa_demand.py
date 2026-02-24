import numpy as np
# import pprint
# Production function related functions, hard-coding of various aspects of production funciton


def cme_prod_ces_dmrl_intr(psi, theta_i_j, theta_i_1):
    fl_intr = (1/(1-psi))*np.log(theta_i_j/theta_i_1)
    return fl_intr


def cme_prod_ces_dmrl_slpe(psi):
    return -1*(1/(1-psi))


def cme_prod_ces(fl_elas, ar_share, ar_input):
    """Evaluate CES Production Function

    Compute aggregate outputs given shares, inputs, and the elasticity parameter

    Parameters
    ----------
    fl_elas : _type_
        The parameter value that determines elasticity of substitution, not the
        elasticity of substitution itself itself.
    ar_share : _type_
        _description_
    ar_input : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    fl_output = np.dot(ar_share, ar_input**fl_elas)**(1/fl_elas)

    return fl_output


# Note should move the derivative function here, from cme_solve_demand_eval
def cme_prod_ces_deri(fl_prt_qty, fl_prt_pwr, fl_chd_shr, fl_chd_qty, fl_prt_drc):
    """Current Segment derivative

    drc: which is MPL for the current input at the current layer
    drv: which is the contribution of the node at the current layer to the cumulative MPL

    Note the CES derivative structure is, we are abusing notations, now indexing children within layer:
      y0 = (a_1 ( y1 )^{b_1} + ... )^(1/b_1)
      y1 = (a_2 ( y2 )^{b_2} + ... )^{1/b_2}
      y2 = (a_3 ( y3 )^{b_3} + ... )^{1/b_3}
      and corresponding derivatives are:
      dy0/dy1 = (1/b_1) * (a_1 ( y1 )^{b_1} + ... )^(1/b_1 - 1) * (a_1 * b_1 * (y1)^{b_1 - 1} )
      dy1/dy2 = (1/b_2) * (a_2 ( y2 )^{b_2} + ... )^(1/b_2 - 1) * (a_2 * b_2 * (y2)^{b_2 - 1} )
      dy2/dy3 = (1/b_3) * (a_3 ( y3 )^{b_3} + ... )^(1/b_3 - 1) * (a_3 * b_3 * (y3)^{b_3 - 1} )
      simplifying, we have:
      layer 1 nodes: dy0/dy1 = (1/b_1) * (y0)^(1-b_1) * (a_1 * b_1 * (y1)^{b_1 - 1} )
                             = (y0)^(1-b_1) * a_1 * (y1)^{b_1 - 1}
      layer 2 nodes: dy1/dy2 = (1/b_2) * (y1)^(1-b_2) * (a_2 * b_2 * (y2)^{b_2 - 1} )
                             = (y1)^(1-b_2) * a_2 * (y2)^{b_2 - 1}
      layer 3 nodes: dy2/dy3 = (1/b_3) * (y2)^(1-b_3) * (a_3 * b_3 * (y3)^{b_3 - 1} )
                             = (y2)^(1-b_3) * a_3 * (y3)^{b_3 - 1}

    Parameters
    ----------
    fl_prt_qty : _type_
        _description_
    fl_prt_pwr : _type_
        _description_
    fl_chd_shr : _type_
        _description_
    fl_chd_qty : _type_
        _description_
    """
    # part 1: (y0)^(1-b_1)
    fl_chd_drv_p1 = (fl_prt_qty)**(1-fl_prt_pwr)
    # part 2: (a_1 * (y1)^{b_1 - 1} )
    fl_chd_drv_p2 = fl_chd_shr * (fl_chd_qty)**(fl_prt_pwr-1)
    # combine for current
    fl_chd_drv = fl_chd_drv_p1 * fl_chd_drv_p2

    # cumulative
    fl_chd_drc = fl_prt_drc * fl_chd_drv

    return fl_chd_drv, fl_chd_drc, fl_chd_drv_p1