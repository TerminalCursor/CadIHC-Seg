from numpy.typing import NDArray
from numpy.ma import MaskedArray


def color_mask(mask: NDArray | MaskedArray, background: int | None = 0, random: bool = False, base_cmap: str | None = None):
    """# Generate a Colormap for the Mask

    If the `ncolor` library is available, that will be used
    Otherwise, color with a randomly shuffled colormap

    background: An integer or `None`
     `None` implies that the background should not be masked
     Otherwise, return the colormap masked to the background value
    random: Suppress `ncolor` usage
    base_cmap: Base colormap to use if `ncolor` isn't used"""
    from numpy.ma import masked_equal
    from numpy import unique
    from numpy.random import randint

    try:
        assert not random
        import ncolor

        nmask = ncolor.label(mask)
        if background is not None:
            return masked_equal(nmask, background)
        return nmask
    except:
        from .common import random_colormap

        cmap = random_colormap(mask, base_colormap=base_cmap)
        if background is not None:
            return cmap(masked_equal(mask, background))
        return cmap(mask)
