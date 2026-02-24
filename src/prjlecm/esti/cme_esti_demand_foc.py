"""Demand-side estimation based on FOC
The :mod:`prjlecm.esti.cmeesti_demand_ofc` takes a dataframe with labor quantities and 
wages and estimates share and elasticity parameters on the demand side based on FOC. 

Note that at the lowest nest, we have the standard FOC. But at highest levels, because
different parent nests can not be canceled out, the FOC is more complicated. 

1. Input data file

The existing dict structure. The dict structure contains all relevant information.
there is already quantity and prices, and construct the relative quantity and 
price equations

Several dict files from different years. 

INPUT files

- W and Q stored in dataframe version of DICS, with full upward nesting information, data for separate 
    sets of things. layer within, and dict key across
- table Matching of key_node id to observable, observabley at top top layer is equivalent to dict-level observable

2. Estimation inputs from within year

- Start at layer 1
- Iterate over each nest, get all keys within nest
- Select from full dataframe wll rows across "years" with the same keys
- Use the initial key as the base, extract dataframe with year key
- merge back
- Divide all w and q levels by base. 
- Keep all rows except for base
- Get regression information from the full DICT 
- regression with appropriate polynomials, either across time, and within time. 
- store estimation coefficients back to DICT
- based on DICT normalization instructions, generate appropriate share parameters store to DICT
- DICT generates higher layer aggreagte Q and P, iterate forward.

At each tier, we have to construct the relevant 
Y variable, and X variable, and run regression. 

At each tier, the x and y variables are different, and then 
also perhaps how to interpret the intercept. Or maybe ned to adjust the Y-LHS. 

- relative Q and relative W.

3. Stack estimation inputs from multiple years

4. Estimate with multiple elements within year and from across years

5. Store estimated results into the dictionary using new aux dict structure. 

Includes method :func:`cme_`, 
"""