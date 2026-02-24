# Generating Share Parameters

## Share parameter generator

I developed a fairly convoluted way of generating random share parameters for neseted CES systems, this should be dramatically simplified, but is that possible?

What are the programming requirements?

- Pure random parameters
- Parameters that are functions of observables
  - observables might be categorical
  - observables might be time-based (time polynomial)

Treat parameters as linear predictions.
$$
\alpha = \exp\left( a + \beta \cdot X + \epsilon\right) \cdot \Omega
$$

I will need to run this several times. for different sets of parameters across the system across subnests and times.

THe exponential is needed to keep everything positive.

Note that all share parameters are relative. So we are generating proportional, relative values, can be arbitrarily scaled. Where, the $\Omega>0$ is the arbitrary scaler.

In the no shock setting,  we need to specifiy a set of
$\left\{a_j, \beta_j, X_j\right\}_{j=1}^J$ for the $J$ sets of share parameters to draw.

Suppose we have a two-layer nested, with 3 share parameters, then $J=3$. In each sub-nest the children are ages, so the $X_1=X_2$, but $\beta_1 \neq \beta_2$. The outer share is by calendar year, which does not impact the inner results.

So the job of this function is simply to take a, beta, and X, and generate output shares, it does not care about if the shares should sum up to one (the might not if different with shares across "years"), and also not worry about which nest-layer we are talking about

## Share parameter plug into system

There are three steps in this process,

1. Generate J sets of a, beta, X
2. Call the share parameter generating function
3. Normalize, with-in sum to 1, or bounded below and above
4. Insert into dict system

## Updated structure (2023-10-04 19:19:07, updated again on 2023-10-15 14:10:40)

Prior components do not made sense to me, re-think generalization.

The framework below works with one production function (at one moment in time for example, or for one firm).

Let $X$ be a $1$ by $K$ vector, and let $\beta$ be a $1$ by $K$ vector of coefficients. Each share parameter is always:

$$
\alpha = \exp(
    X\cdot \beta' + \epsilon
)\cdot \Omega
$$

For example, suppose we have an age-based polynomial, and $X=\left[\text{age}^0,  \text{age}^1, \text{age}^2, \text{age}^3\right]$, $\beta = \left[0.3, +0.2, -0.01, +0.002\right]$, $\epsilon=0$, and $\Omega$. But note that there is a problem with $\Omega$ equal to an arbitrary value, it is a normalizer potentially, that we need to determine based on different share parameters.

$\Omega$ is the normalizing scaler, $\epsilon$ is the shock uncertainty value, $X$ are "observables" including a constant, and $\beta$ are parameters.

There are different "sets" of parameters, denote these as $S$ sets with index $s$. Each set shares the same $\beta_s$ and $\Omega_s$, but there are $M-1$ (last one is residual) share parameters in each set, with $X$ now is a $M$ by $K$ matrix, where each "row" is a different parameter, and columns are "observables". 

Note: the idea is that across share parameters within each $s$ (or nest), we have the same "parameters" (e.g., polynomial parameters), that will deliver differing share parameters when they are multiplied with different observables vectors $X_{m,s}$.

So to begin simulation, we need to have:
$$
\Lambda =
\left\{
    \beta_s, \left\{
        \epsilon_{m,s},
        X_{m,s}
    \right\}_{m=1}^{M_s-1},
    \Omega_s
\right\}_{s=1}^S
$$

Where $\beta_s$ is a $1$ by $K$ vector, $\epsilon_{ms}$ is a scalar that is $m$ and $s$ specific, and $X_{ms}$ is $1$ by $K$ that is also $m$ and $s$ specific.

Note ${M_s-1}$, residual is the "-1".

IN another word:

- nest-specific: $\beta$ vector and $\Omega$ scalar
- share-specific (within-nest): $X_{m,s}$ vector and $\epsilon$ scalar

Which in each "set", parameters are defined in the "same way", what does that mean? that means share parameters are
For each share parameter, we need a vector of $X_{1}$

1. Generating $\Lambda$
2. Evaluate loop through to generate shares, with $\Omega_s=1$
3. Normalization schemes with $\Omega_s$ changes to meet normalization objectives? those have to be some how specified.

<!-- TODO actually what should omega be it is endogeneous, so there should be other nest specific things each nest must sum to 1 -->
how to deal wit $\Omega$?

## More Generalized Structure

Now we describe a even more generalized parameter structure. Suppose we have multiple production functions, perhaps specified across different time periods, that rely on the same parameter structure. 

We need to specify them jointly, because the $\Omega$ normalizer for some nests might need to have consistency across production functions, if the vector of $X_{m,s}$ attributes not only contain elements that vary across $m$, but also contain components shared acorss $m$, but differ across $w$, where $w$ are production functions. 

We need to have:
$$
\Lambda =
\left\{
    \left\{
        \beta_{s,w}, \left\{
            \epsilon_{m,s,w},
            X_{m,s,w}
        \right\}_{m=1}^{M_s-1},
        \Omega_{s,w}
    \right\}_{s=1}^S
\right\}_{w=1}^W
$$

- $w$ are different productions 
- $s$ are different nests
- $m$ are different share parameters within next
- $K_{m,s,w}$ is the number of "observables" that jointly determine $m$, note that $X_{m,s,w}$ and $\beta_{s,w}$ are both $1\times K$ vectors.

For some subnests, it could be the case that $\Omega_{s,w}$ is homogeneous across $w$, so the same normalizer is used, when $X_{m,s,w}$ have attributes that are common within $m,s$, but differ across $w$. 

### Implementation Function 1: Share parameter generate

- Input: $s$ and $w$ specific set of parameters (from function 3) and $X$ (from function 2)
- Output: $M$ share parameters
- Given parameters, and "observables", evaluate function, generate shares

### Implementation Function 2: Polynomial "observable" matrix generator 

Note these are generated independent of the polynomial parameters that they will be multiplied by. 

- Input: 
    - $\left\{K_{s,w}\right\}_{s}^{S}$ vector of polynomial order for $S$ polynomial parameters
    - $\left\{\left\{\hat{x}_{m,s}\right\}_{m=1}^{M_s}\right\}_{s=1}^S$ is not a matrix (because of potentially different $M_s$), dictionary of arrays, where each element of the dictionary is a different nest, and each element of each array is a scalar value that will be polynomial expanded
- Output
    - $\left\{\left\{X_{m,s}\right\}_{m=1}^{M_s}\right\}_{s=1}^S$, this will be a dictionary of matrixes, the dictionary is over $s$, the nests, and the matrixes are expanded $X_s$ matrix that will be $M_{s}\times K_{s}$ in size, where $K_s$ is the number of polynomial terms, and $M_s$ is the number of share parameters within a nest. 

### Implementation Function 3: Generate polynomial coefficients

- Input:
    - $\left\{K_{s,w}\right\}_{s=1}^{S}$ vector of polynomial order for $S$ polynomial parameters
    - Random seeds
- Output
    - $\left\{\beta_s\right\}_{s=1}^S$ is the dictionary of parameter arrays, each $s$ specific to a different nest. 
