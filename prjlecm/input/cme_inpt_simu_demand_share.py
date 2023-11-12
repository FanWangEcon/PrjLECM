"""
Simulate share parameters in nested CES system with multiple production functions


see: https://github.com/FanWangEcon/PrjLECM/issues/2
"""

import pprint as pprint

import ast as ast
import numpy as np
import pandas as pd

import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand


def cme_simu_ces_shrparam_poly_obs(
        ar_it_poly_order = [3, 3, 0],
        dc_ar_fl_x_obs = {0:[25, 35, 45], 1:[25,35,45], 2:[0,1]}, 
        verbose = False        
):    
    """Generate polynomial expanded X input matrix for share parameters

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
        _description_, by default [3, 3, 0]
    dc_ar_fl_x_obs : dict, optional
        _description_, by default {0:[25, 35, 45], 1:[25,35,45], 2:[0,1]}
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    dict of numpy.array
        Each element of the dictionary is for a particular nest. Each row
        is for a particular parameter in the nest/subnest. Each column
        is a polynomial expanded value based on an underlying scalar value
        from `dc_ar_fl_x_obs`, number of polynomial-expansion order 
        per nest is determined by elements of `ar_it_poly_order`
    """

    # Number nests
    it_nest_count = len(ar_it_poly_order)
    # # Count dict size, this must match
    # it_nest_count_dc = len(dc_ar_fl_x_obs)

    # Storage output 
    dc_mt_x_poly = {}
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
        dc_mt_x_poly[it_nest_ctr] = mt_fl_x_poly

    # print
    if verbose:
        print(f'F-714190, {dc_mt_x_poly=}')

    # return 
    return(dc_mt_x_poly)



if __name__ == "__main__":

    # Default call
    cme_simu_ces_shrparam_poly_obs(verbose=True)



