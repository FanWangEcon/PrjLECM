import numpy as np


def cme_math_lin2n(a, b, c, d, verbose=False):
    """Solve for intersection of two lines
    """
    # Loop over individuals
    # We have two equations:
    # ln(y) = a + b ln(x)
    # ln(y) = c + d ln(x)
    # which becomes
    # a + b ln(x) = c + d ln(x)
    # which means
    # ln(x) = (a - c)/(d - b)
    # and we also have
    # ln(y) = a + b ln(x)

    ln_of_x = (a-c)/(d-b)
    ln_of_y = a + b * ln_of_x

    return ln_of_x, ln_of_y


def cme_math_lininv(a, b, verbose=False):
    # Intercept and slope inverter
    # we have y(x)
    # y = a + b * x
    # inverting x and y, we can have x(y)
    # y - a = b * x
    # (1/b) * y - (a/b) = x
    # x = (-a/b) + (1/b)*y
    # inv_intr = inverted intercept
    # inv_slpe = inverted slope
    inv_intercept = (-a/b)
    inv_slope = (1/b)
    return inv_intercept, inv_slope


if __name__ == "__main__":

    shr_i1_j1 = 0.32433331
    shr_i2_j1 = 1 - shr_i1_j1
    elas_pwr = 0.2
    # a and b
    intr = np.log(shr_i2_j1/shr_i1_j1)*(1/(1-elas_pwr))
    slpe = -1/(1-elas_pwr)

    # inv_intr and inv_slpe
    inv_intr = np.log(shr_i2_j1/shr_i1_j1)
    inv_slpe = (elas_pwr-1)

    # Check
    inv_intr_alt, inv_slpe_alt = cme_math_lininv(intr, slpe)
    # print
    print(f'{inv_intr=} and {inv_intr_alt}')
    print(f'{inv_slpe=} and {inv_slpe_alt}')
