# Code for calculating the robust sigma in the tables.

def robust_sigma(in_y, zero=0):
    """
    Compute a robust (outlier-resistant) estimate of the dispersion (sigma)
    of a 1-D distribution using:
      - Median Absolute Deviation (MAD) for an initial scale estimate
      - Tukey biweight (bisquare) reweighting for a robust variance

    Canonical References:
        * Hoaglin, D. C., Mosteller, F., & Tukey, J. W. (1983).
          "Understanding Robust and Exploratory Data Analysis."
          New York: Wiley.
        * Beers, T. C., Flynn, K., & Gebhardt, K. (1990),
          AJ, 100, 1.
          Common astronomical implementation of biweight estimators.

    Args:
        in_y : numpy array
            Input values.
        zero : bool
            If True, center at zero; otherwise use sample median.

    Returns:
        float : Robust sigma estimate.
    """
    y = in_y.reshape(in_y.size, )
    eps = 1.0E-20
    c1 = 0.6745   # MAD → σ scale factor
    c2 = 0.80
    c3 = 6.0      # Tukey biweight tuning constant
    c4 = 5.0
    c_err = -1.0
    min_points = 3

    y0 = 0.0 if zero else np.median(y)

    dy = y - y0
    del_y = abs(dy)

    mad = np.median(del_y) / c1
    if mad < eps:
        mad = np.mean(del_y) / c2
    if mad < eps:
        return 0.0

    u = dy / (c3 * mad)
    uu = u * u

    q = np.where(uu <= 1.0)
    if len(q[0]) < min_points:
        print('ROBUST_SIGMA: This distribution is TOO WEIRD! Returning', c_err)
        return c_err

    numerator = np.sum((y[q]-y0)**2.0 * (1.0-uu[q])**4.0)
    n = y.size
    den1 = np.sum((1.0-uu[q]) * (1.0-c4*uu[q]))
    siggma = n * numerator / (den1 * (den1 - 1.0))

    if siggma > 0:
        return np.sqrt(siggma)
    else:
        return 0.0
