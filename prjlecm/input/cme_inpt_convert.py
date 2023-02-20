# There are two ways of providing inputs, either via a pandas dataframe or via a dictionary
# Inputs to the solution programs are dictionaries, so panda dataframes inputs will be converted
# to nested dictionaries. Outputs hard to read as dictionary, so will be outputted optionally
# also as pandas dataframes, to be stored as a csv file with maybe key/variable information
# as outputs to the equilibrium solution problem. 

import pprint
import pandas as pd

def cme_dictkey_pdvarname():
    return "key_node"

def cme_convert_pd2dc(df_demand_or_supply, verbose=False):

    # 1. move key_node column to index   
    st_key_var_name = cme_dictkey_pdvarname()
    df_from_nested = df_from_nested.set_index(st_key_var_name)

    # 2. Convert to dictionary
    dc_from_df = df_from_nested.to_dict(orient="index")
    
    # print
    if verbose:
        print('d-129941 dc_from_df:')
        pprint.pprint(dc_from_df)
    
    return dc_from_df


def cme_convert_dc2pd(dc_demand_or_supply, verbose=False):

    # 1. convert to dataframe
    st_key_var_name = cme_dictkey_pdvarname()
    df_from_dc = pd.DataFrame.from_dict(dc_demand_or_supply, orient='index')

    # 2. keys from top nest as variable and rename as key_node
    df_from_dc = df_from_dc.reset_index()
    df_from_dc.rename(columns={'index':st_key_var_name}, inplace=True)

    # Print
    if verbose:
        print('d-129941 df_from_dc:')
        print(df_from_dc)

    return df_from_dc