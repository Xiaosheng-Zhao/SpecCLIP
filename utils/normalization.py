# Code for normalizing the LAMOST LRS spectra.

import numpy as np
from scipy.signal import medfilt, savgol_filter

def gaspp_fitcont2(ww, ff, cfsnr=60):
    """
    Fit continuum over 3850 - 9000 A region.

    Parameters:
    ww : array_like
        Wavelength array.
    ff : array_like
        Flux array.
    cfsnr : float
        Average signal-to-noise ratio in the spectrum.

    Returns:
    cc : ndarray
        Continuum flux array.
    orflx : ndarray
        Original flux array.
    """
    
    # Keep a copy of the original flux
    orflx = ff.copy()

    # **Convert ff to a supported data type with native byte order**
    ff = ff.astype(np.float64)

    # Smooth the flux with a median filter of size 7
    ff = medfilt(ff, kernel_size=7)

    # Copy ww and ff to ww100 and ff100
    ww100 = ww.copy()
    ff100 = ff.copy()

    # Define wavelength ranges and indices
    wran1ind = np.where((ww100 <= 5700.0) & (ww100 >= 3700.0))[0]
    nran1 = len(wran1ind)
    wran2ind = np.where(ww100 > 6100.0)[0]
    nran2 = len(wran2ind)

    # Prepare wavelength and flux arrays over certain ranges
    if nran1 > 100:
        wran1 = ww100[wran1ind]
        fran1 = ff100[wran1ind]
    else:
        wran1 = ww100
        fran1 = ff100

    if nran2 > 100:
        wran2 = ww100[wran2ind]
        fran2 = ff100[wran2ind]
    else:
        wran2 = ww100
        fran2 = ff100

    # Depending on cfsnr, set nl and apply Savitzky-Golay filter
    if cfsnr < 50:
        nl = int(-1.0 * cfsnr + 55)
        # Ensure nl is odd and at least 3
        if nl % 2 == 0:
            nl += 1
        if nl < 3:
            nl = 3
        ofl = savgol_filter(fran1, window_length=nl, polyorder=4, mode='nearest')
        ofl2 = savgol_filter(fran2, window_length=nl, polyorder=4, mode='nearest')
    else:
        ofl = savgol_filter(fran1, window_length=3, polyorder=2, mode='nearest')
        ofl2 = savgol_filter(fran2, window_length=3, polyorder=2, mode='nearest')

    # For wran1, perform initial polynomial fit and adjustments
    if nran1 > 100:
        # Initial polynomial fit to ofl
        coef0 = np.polyfit(wran1, ofl, 5)
        cfit0 = np.polyval(coef0, wran1)
        # Find the index where cfit0 is maximum
        flxind = np.argmax(cfit0)
        if wran1[flxind] < 4500:
            w11, w12 = 4030, 4160
            w21, w22 = 4270, 4410
            w31, w32 = 4800, 4940
            wbin = 10
            # Indices for specific wavelength ranges
            wind1ind = np.where((ww >= w11) & (ww <= w12))[0]
            wind2ind = np.where((ww >= w21) & (ww <= w22))[0]
            wind3ind = np.where((ww >= w31) & (ww <= w32))[0]
            # Indices for interpolation points
            indw11 = np.where((ww >= w11 - wbin) & (ww <= w11 + wbin))[0]
            indw12 = np.where((ww >= w12 - wbin) & (ww <= w12 + wbin))[0]
            indw21 = np.where((ww >= w21 - wbin) & (ww <= w21 + wbin))[0]
            indw22 = np.where((ww >= w22 - wbin) & (ww <= w22 + wbin))[0]
            indw31 = np.where((ww >= w31 - wbin) & (ww <= w31 + wbin))[0]
            indw32 = np.where((ww >= w32 - wbin) & (ww <= w32 + wbin))[0]
            # Get maximum fluxes at those points
            f11 = fran1[indw11].max() if len(indw11) > 0 else None
            f12 = fran1[indw12].max() if len(indw12) > 0 else None
            f21 = fran1[indw21].max() if len(indw21) > 0 else None
            f22 = fran1[indw22].max() if len(indw22) > 0 else None
            f31 = fran1[indw31].max() if len(indw31) > 0 else None
            f32 = fran1[indw32].max() if len(indw32) > 0 else None
            # Perform interpolation
            if f11 is not None and f12 is not None and len(wind1ind) > 0:
                fran1[wind1ind] = np.interp(ww[wind1ind], [w11, w12], [f11, f12])
            if f21 is not None and f22 is not None and len(wind2ind) > 0:
                fran1[wind2ind] = np.interp(ww[wind2ind], [w21, w22], [f21, f22])
            if f31 is not None and f32 is not None and len(wind3ind) > 0:
                fran1[wind3ind] = np.interp(ww[wind3ind], [w31, w32], [f31, f32])
        # Iteratively fit polynomial and adjust 'ofl'
        ofl = fran1.copy()
        for _ in range(10):
            coef = np.polyfit(wran1, ofl, 5)
            cfit = np.polyval(coef, wran1)
            ofl = np.maximum(cfit, ofl)
        # Final fit
        coef_tot = np.polyfit(wran1, ofl, 5)
        cfit1 = np.polyval(coef_tot, wran1)
    else:
        cfit1 = None

    # For wran2, perform iterative fitting with conditions
    n1000 = 0
    if nran2 > 100:
        while n1000 <= 8:
            coef2 = np.polyfit(wran2, ofl2, 4)
            cfit2 = np.polyval(coef2, wran2)
            ysig2 = np.std(ofl2 - cfit2)
            # Adjust ofl2 where condition is met
            mask = (ofl2 < cfit2) | (ofl2 > cfit2 + 3.0 * ysig2)
            ofl2[mask] = cfit2[mask]
            n1000 += 1
    else:
        cfit2 = None

    # Assemble total wavelength and continuum fit
    totwran = ww.copy()
    totcfit = ff.copy()
    if nran1 > 100:
        totcfit[wran1ind] = cfit1
    if nran2 > 100:
        totcfit[wran2ind] = cfit2
    # Replace non-positive values in totcfit with 1.0
    totcfit[totcfit <= 0.0] = 1.0

    cc = totcfit
    # Return the continuum and original flux
    return cc