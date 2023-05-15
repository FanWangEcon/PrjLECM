# There are two ways of providing inputs, either via a pandas dataframe or via a dictionary
# Inputs to the solution programs are dictionaries, so panda dataframes inputs will be converted
# to nested dictionaries. Outputs hard to read as dictionary, so will be outputted optionally
# also as pandas dataframes, to be stored as a csv file with maybe key/variable information
# as outputs to the equilibrium solution problem. 

import pprint
import pandas as pd

import prjlecm.input.cme_inpt_simu_demand as cme_inpt_simu_demand
import prjlecm.input.cme_inpt_simu_supply as cme_inpt_simu_supply

def cme_dictkey_pdvarname():
    return "key_node"

def cme_convert_pd2dc(df_demand_or_supply, input_type='demand', verbose=False):

    # 1. move key_node column to index   
    st_key_var_name = cme_dictkey_pdvarname()
    df_from_nested = df_demand_or_supply.set_index(st_key_var_name)

    # 2. Convert to dictionary
    dc_from_df = df_from_nested.to_dict(orient="index")

    # 3. Value types conversion 
    if input_type == "demand":
        __, __ , dc_type_return = cme_inpt_simu_demand.cme_simu_demand_ces_inner_dict()
    elif input_type == "supply":
        __ , dc_type_return = cme_inpt_simu_supply.cme_simu_supply_lgt_dict()

    # Int should be int, list should be list, etc.
    for it_key, dc_val in dc_from_df.items():
        for st_key, cell_val in dc_val.items():
            if (st_key == "ipt") and (type(cell_val) == list):
                pass
            else:
                if cell_val is None or \
                        pd.isna(cell_val) or \
                        str(cell_val).strip() == "":
                    dc_from_df[it_key][st_key] = None
                else:
                    dc_from_df[it_key][st_key] = dc_type_return[st_key](cell_val)

    # print
    if verbose:
        print('d-129941 dc_from_df:')
        pprint.pprint(dc_from_df)
    
    return dc_from_df


def cme_convert_dc2pd(dc_demand_or_supply, input_type='demand', verbose=False):

    # 1. convert to dataframe
    st_key_var_name = cme_dictkey_pdvarname()
    df_from_dc = pd.DataFrame.from_dict(dc_demand_or_supply, orient='index')

    # 2. keys from top nest as variable and rename as key_node
    df_from_dc = df_from_dc.reset_index()
    df_from_dc.rename(columns={'index':st_key_var_name}, inplace=True)

    # 3. Value types conversion
    # Convert int columns to int
    if input_type == "demand":
        __, __ , dc_type_return = cme_inpt_simu_demand.cme_simu_demand_ces_inner_dict()
    elif input_type == "supply":
        __ , dc_type_return = cme_inpt_simu_supply.cme_simu_supply_lgt_dict()
    for st_col, fc_datatype in dc_type_return.items():
        if (fc_datatype is int) and (st_col in df_from_dc.columns):
            # df_from_dc = df_from_dc.astype({st_col: 'int'})
            df_from_dc[st_col] = df_from_dc[st_col].astype(float).astype('Int64')

    # Print
    if verbose:
        print('d-129941 df_from_dc:')
        print(df_from_dc)

    return df_from_dc