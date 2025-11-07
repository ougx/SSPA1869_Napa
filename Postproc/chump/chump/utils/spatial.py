import numpy as np



def transform_grid(xx, yy, xoff=0., yoff=0., angrot=0., lfactor=1., inverse=False):
    """
    test:
    xx = np.array([0, 1])
    yy = np.array([0, 1])
    xoff=1.
    yoff=1.
    angrot=45.
    lfactor=2.
    inverse=False
    transform_grid(xx, yy, xoff, yoff, angrot, lfactor, inverse)
    """
    if inverse:
        angrot_radians = -np.pi * angrot / 180.
        x1 = (xx - xoff) / lfactor
        y1 = (yy - yoff) / lfactor
        return (
            x1 * np.cos(angrot_radians) - y1 * np.sin(angrot_radians),
            x1 * np.sin(angrot_radians) + y1 * np.cos(angrot_radians)
        )
    else:
        angrot_radians = np.pi * angrot / 180.
        return (
            (xx * np.cos(angrot_radians) - yy * np.sin(angrot_radians)) * lfactor + xoff,
            (xx * np.sin(angrot_radians) + yy * np.cos(angrot_radians)) * lfactor + yoff
        )
