"""Material-coordinate basin geometry for the approved experimental fold task.

Ported unchanged from the retained continuous-bending diagnostic.
"""
import numpy as np

def regions(grid, row, bandwidth):
    """Local basins separated by minima; centre/width weighted by turning.

    The half-height interval within each basin is used only for matching. Flat
    whole profiles have no localized bend. Numeric flat maxima use their
    midpoint and are explicitly flagged. No physical amplitude cutoff exists.
    """
    x, y = np.asarray(grid), np.asarray(row)
    floor = 128*np.finfo(float).eps*max(1., float(y.max()))
    if y.max() <= floor or np.ptp(y) <= floor:
        return []
    groups = []; begin = 0
    for end in range(1, len(y)+1):
        if end == len(y) or abs(y[end]-y[end-1]) > floor:
            groups.append((begin, end)); begin = end
    peaks = []
    for lo, hi in groups:
        level = float(np.min(y[lo:hi]))
        if (level > floor and (lo == 0 or level > y[lo-1]+floor)
                and (hi == len(y) or level > y[hi]+floor)):
            peaks.append(((lo+hi-1)//2, hi-lo > 1))
    if not peaks:
        return []
    cuts = [0]+[a+int(np.argmin(y[a:b+1])) for (a, _), (b, _) in zip(peaks, peaks[1:])]+[len(y)-1]
    result = []
    for k, (peak, plateau) in enumerate(peaks):
        lo, hi = cuts[k:k+2]
        xx, yy = x[lo:hi+1], y[lo:hi+1]
        mass = float(np.trapezoid(yy, xx))
        if mass <= floor*(x[-1]-x[0]):
            continue
        centre = float(np.trapezoid(xx*yy, xx)/mass)
        width = float(np.sqrt(max(0., np.trapezoid((xx-centre)**2*yy, xx)/mass)))
        left = peak
        while left > lo and y[left-1] >= .5*y[peak]:
            left -= 1
        right = peak
        while right < hi and y[right+1] >= .5*y[peak]:
            right += 1
        flags = []
        if plateau:
            flags.append('plateau')
        if x[peak]-x[0] < 2*bandwidth or x[-1]-x[peak] < 2*bandwidth:
            flags.append('boundary_affected')
        result.append(dict(centre_m=centre, width_m=width, turn_rad=mass,
                           peak_m=float(x[peak]), core_m=[float(x[left]), float(x[right])],
                           basin_m=[float(xx[0]), float(xx[-1])], flags=flags))
    return result
