def color_mask(mask, background=0):
    """# Generate a Colormap for the Mask

    If the `ncolor` library is available, that will be used
    Otherwise, color with a randomly shuffled colormap

    background: An integer or `None`
     `None` implies that the background should not be masked
     Otherwise, return the colormap masked to the background value"""
    try:
        import ncolor
        from numpy.ma import masked_equal

        nmask = ncolor.label(mask)
        if background is not None:
            return masked_equal(nmask, background)
        return nmask
    except:
        from .common import random_colormap

        cmap = random_colormap(mask)
        if background is not None:
            return cmap(masked_equal(mask, background))
        return cmap(mask)
