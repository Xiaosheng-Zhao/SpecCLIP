# Code for calculating the robust sigma in the tables.

def robust_sigma(in_y, zero=0):
    """Calculate a resistant estimate of the dispersion of a distribution."""
    y = in_y.reshape(in_y.size, )
    eps = 1.0E-20
    c1 = 0.6745
    c2 = 0.80
    c3 = 6.0
    c4 = 5.0
    c_err = -1.0
    min_points = 3

    if zero:
        y0 = 0.0
    else:
        y0 = np.median(y)
    
    dy = y - y0
    del_y = abs(dy)
    mad = np.median(del_y) / c1
    
    if mad < eps:
        mad = np.mean(del_y) / c2
    if mad < eps:
        return 0.0
        
    u = dy / (c3 * mad)
    uu = u*u
    q = np.where(uu <= 1.0)
    count = len(q[0])
    
    if count < min_points:
        print('ROBUST_SIGMA: This distribution is TOO WEIRD! Returning', c_err)
        return c_err
        
    numerator = np.sum((y[q]-y0)**2.0 * (1.0-uu[q])**4.0)
    n = y.size
    den1 = np.sum((1.0-uu[q]) * (1.0-c4*uu[q]))
    siggma = n * numerator / (den1 * (den1 - 1.0))
    
    if siggma > 0:
        out_val = np.sqrt(siggma)
    else:
        out_val = 0.0
    return out_val