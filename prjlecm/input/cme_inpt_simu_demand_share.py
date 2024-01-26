"""
Simulate share parameters in nested CES system with multiple production functions


see: https://github.com/FanWangEcon/PrjLECM/issues/2
"""

import pprint as pprint

import ast as ast
import numpy as np
import pandas as pd

import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand

def cme_simu_ces_shrparam_poly_coef(
    ar_it_poly_order = [3, 3, 1],
    ar_it_poly_seed = [123, 456, 445],
    dc_ar_fl_poly_scale = {0:[5e0, 1e-1, 1e-2, 1e-3], 1:[3e0, 1e0, 1e-2, 1e-2], 2:[1e0, 1e0]},
    dc_ar_bl_poly_pos = {0:[True, True, False, True], 1:[True, True, False, True], 2:[True, True]},
    verbose = False        
):
    """Generate dict of polynomial coefficient array for constructing share parameters

    SHARE = MT_X_POLY x AR_POLY_COEF, we generate AR_POLY_COEF here. 

    This function carries out "Implementation Function 3", Generate polynomial coefficients.
    From https://github.com/FanWangEcon/PrjLECM/issues/2.

    Parameters
    ----------
    ar_it_poly_order : list, optional
        Polynomial order for each nest, by default [3, 3, 0]
    ar_it_poly_seed : list, optional
        Coefficient seed per nest, by default [123, 345, 445]
    dc_ar_fl_poly_scale : dict, optional
        Coefficient scaling per nest and child, 
        by default {0:[5e0, 1e-1, 1e-2, 1e-3], 1:[3e0, 1e0, 1e-2, 1e-2], 2:[1e0]}
    dc_ar_bl_poly_pos : dict, optional
        Coefficient pos or neg per nest and child, 
        by default {0:[True, True, False, True], 1:[True, True, False, True], 2:[True]}
    verbose : bool, optional
        print more, by default False

    Returns
    -------
    dict of numpy.array
        Each element of the dictionary is for a particular nest. 1 by K. 
        Each column is coefficient for an observable that is nest and child 
        specific. The random polynomial parameters are nest specific but not child
        specific. K is the polynomial order + 1. 
        is a polynomial expanded value based on an underlying scalar value
        from `dc_ar_fl_x_obs`, number of polynomial-expansion order 
        per nest is determined by elements of `ar_it_poly_order`
    """
    # Number nests
    it_nest_count = len(ar_it_poly_order)
    # # Count dict size, this must match
    # it_nest_count_dc = len(dc_ar_fl_x_obs)

    # Storage output
    dc_ar_poly_coef = {}
    # loop over nest and element
    for it_nest_ctr in range(it_nest_count):

        # Polynomial order and seed
        it_poly_order = ar_it_poly_order[it_nest_ctr]
        it_poly_seed = ar_it_poly_seed[it_nest_ctr]

        # Polynomial scaling array and pos boolean array
        ar_fl_poly_scale = dc_ar_fl_poly_scale[it_nest_ctr]
        ar_bl_poly_pos = dc_ar_bl_poly_pos[it_nest_ctr]
        
        # Set seed
        np.random.seed(it_poly_seed)

        # generate random coefficients
        ar_fl_poly_coef_unif0t1 = np.random.rand(it_poly_order+1)

        # Consistent with `cme_simu_ces_shrparam_poly_obs` return, columns are polynomial count
        ar_fl_poly_coef = np.empty([1, it_poly_order+1])
        # Generate coefficient by coefficient
        for it_step_order in np.arange(it_poly_order+1):

            # Random draw
            fl_poly_coef_unif0t1 = ar_fl_poly_coef_unif0t1[it_step_order]
            # Polynomial nest- and child-specific scaler and pos/neg
            fl_poly_scale = ar_fl_poly_scale[it_step_order]
            bl_poly_pos = ar_bl_poly_pos[it_step_order]

            fl_poly_coef = fl_poly_coef_unif0t1*fl_poly_scale
            if bl_poly_pos is False:
                fl_poly_coef = (-1)*fl_poly_coef

            ar_fl_poly_coef[0, it_step_order] = fl_poly_coef
        
        # Store 
        dc_ar_poly_coef[it_nest_ctr] = ar_fl_poly_coef
            
    # print
    if verbose:
        print(f'F-714190, A, {dc_ar_poly_coef=}')

    return dc_ar_poly_coef

def cme_simu_ces_shrparam_poly_obs(
        ar_it_poly_order = [3, 3, 1],
        dc_ar_fl_x_obs = {0:[25, 35, 45], 1:[25,35,45], 2:[0,1]}, 
        verbose = False        
):    
    """Generate polynomial expanded X input matrix for share parameters

    SHARE = MT_X_POLY x AR_POLY_COEF

    This function carries out "Implementation Function 2", Polynomial "observable" matrix generator. 
    From https://github.com/FanWangEcon/PrjLECM/issues/2.

    Each share parameter in a particular nest is the vector product 
    of parameters and observables. When dealing with polynomials, the observables
    are polynomial expanded based on the same scalar value. 

    In the example here, we have a doubly-nested CES system, at the top layer
    we aggregate across genders: male = 0, female = 1. At the bottom layer, 
    we have a female nest and a male nest, inside each of these, we aggregate 
    over three ages, age = 25, 35, and 45. 

    In the age = 25 subnest, we have a 3rd order polynomial, `ar_it_poly_order[0]=3`. 
    This means that we will have four terms for the polynomial expansion: 
    x_{age=25, female=0} = [0*25, 1*25, 2*25, 3*25].

    In this function we generate a dictionary of these x vectors. Which will be 
    multiplied by parameters to generate age- and gender-specific share parameters.

    Parameters
    ----------
    ar_it_poly_order : list, optional
        Each element of list is polynomial order for a nest, by default [3, 3, 0]
    dc_ar_fl_x_obs : dict, optional
        Dict is indexed by nest index, elements in each dict-value-list corresponds
        to the number of children in each nest, by default {0:[25, 35, 45], 1:[25,35,45], 2:[0,1]}
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    dict of numpy.array
        Each element of the dictionary is for a particular nest. M by K.
        M is number of children per nest, K is number of observables determining
        share parameter per nest and child. K is polynomial order + 1.
        Each row is for a particular parameter in the nest/subnest. Each column
        is a polynomial expanded value based on an underlying scalar value
        from `dc_ar_fl_x_obs`, number of polynomial-expansion order 
        per nest is determined by elements of `ar_it_poly_order`
    """

    # Number nests
    it_nest_count = len(ar_it_poly_order)
    # # Count dict size, this must match
    # it_nest_count_dc = len(dc_ar_fl_x_obs)

    # storage output 
    dc_mt_poly_x = {}
    # loop over nest and element
    for it_nest_ctr in range(it_nest_count):

        # polynomial expansion order
        it_poly_order = ar_it_poly_order[it_nest_ctr]
        # nest-specific array of values
        ar_fl_x_obs = dc_ar_fl_x_obs[it_nest_ctr]

        # polynomial expansion
        mt_fl_x_poly = np.empty((len(ar_fl_x_obs), it_poly_order+1))
        for it_x_obs_ctr in range(len(ar_fl_x_obs)):

            fl_x_obs = ar_fl_x_obs[it_x_obs_ctr]
            for it_poly_ctr in range(it_poly_order + 1):
                
                # polynomial expand element-wise
                fl_x_poly = fl_x_obs**it_poly_ctr

                mt_fl_x_poly[it_x_obs_ctr, it_poly_ctr] = fl_x_poly

        # Store to dictionary
        dc_mt_poly_x[it_nest_ctr] = mt_fl_x_poly

    # print
    if verbose:
        print(f'F-714190, B, {dc_mt_poly_x=}')

    # return 
    return(dc_mt_poly_x)

def cme_simu_ces_shrparam_param_gen(
    dc_mt_poly_x = cme_simu_ces_shrparam_poly_obs(verbose=True),
    dc_ar_poly_coef = cme_simu_ces_shrparam_poly_coef(verbose=True),
    verbose = False        
):
    """Generate prod-nest share parameters, given poly obs and coefficients

    Generates production-function and nest-specific share parameters across
    all children within nest. 

    Parameters
    ----------
    dc_mt_poly_x : _type_, optional
        _description_, by default cme_simu_ces_shrparam_poly_obs(verbose=True)
    dc_ar_poly_coef : _type_, optional
        _description_, by default cme_simu_ces_shrparam_poly_coef(verbose=True)
    verbose : bool, optional
        _description_, by default False
    """

    # Number nests
    it_nest_count = len(dc_mt_poly_x)    

    # storage output 
    dc_ar_shares = {}
    dc_fl_normalize = {}
    # loop over nest and element
    for it_nest_ctr in range(it_nest_count):

        # coefficients and polynomials
        mt_poly_x = dc_mt_poly_x[it_nest_ctr]
        ar_poly_coef = dc_ar_poly_coef[it_nest_ctr]

        # nest- and child-specific share parameters
        ar_coef_child = np.matmul(mt_poly_x, np.transpose(ar_poly_coef))

        # Store to dictionary
        dc_ar_shares[it_nest_ctr] = ar_coef_child

    if verbose:
        print(f'F-714190, C, {dc_ar_shares=}')

    return(dc_ar_shares)

def cme_simu_ces_shrparam_normalize(
    dc_ar_shares = cme_simu_ces_shrparam_param_gen(verbose=True),
    fl_share_sum = 1,
    verbose = False        
):
    """Normalize share parameters across nests

    Share parameters need to be normalized, by default, share parameters within
    nest sums to one. And we keep the normalizing constant, and the adjusted
    coefficients. 

    Parameters
    ----------
    dc_ar_share : _type_, optional
        output of `cme_simu_ces_shrparam_param_gen`. 
    fl_share_sum : float, optional
        Sum of share parameters within nest.
    verbose : bool, optional
        _description_, by default False
    """

    # Number nests
    it_nest_count = len(dc_ar_shares)    

    # storage output 
    dc_ar_shares_normalized = {}
    dc_fl_normalize = {}
    # loop over nest and element
    for it_nest_ctr in range(it_nest_count):

        # nest- and child-specific share parameters
        ar_coef_child = dc_ar_shares[it_nest_ctr]

        # Adjustment ratio
        fl_existing_sum = sum(ar_coef_child)
        fl_adj_ratio = fl_share_sum/fl_existing_sum

        # Adjust
        ar_coef_child_adj = ar_coef_child*fl_adj_ratio

        # Store to dictionary
        dc_ar_shares_normalized[it_nest_ctr] = ar_coef_child_adj
        dc_fl_normalize[it_nest_ctr] = fl_adj_ratio

    if verbose:
        print(f'F-714190, D1, {dc_ar_shares_normalized=}')
        print(f'F-714190, D2, {dc_fl_normalize=}')

    return dc_ar_shares_normalized, dc_fl_normalize
