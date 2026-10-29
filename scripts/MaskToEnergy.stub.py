# Helper stub script used to convert a mask into the energetics
def process_mask(mask):
    """# Convert a mask into the energetics"""
    from numpy import zeros
    stat_dict = zeros((int(mask.max()), 4))
    def get_neighbors(mask, x, y):
        """# Neighbor Interaction Helper"""
        from numpy import ndarray
        assert isinstance(x, int) and isinstance(y, int)
        assert isinstance(mask, ndarray)
        assert x < mask.shape[0]
        assert y < mask.shape[1]
        value = mask[x,y]
        ct = False
        cc = False
        if x > 0:
            nvalue = mask[x-1,y]
            if nvalue == 0:
                ct = True
            elif nvalue != value:
                cc = True
        if x+1 < mask.shape[0]:
            nvalue = mask[x+1,y]
            if nvalue == 0:
                ct = True
            elif nvalue != value:
                cc = True
        if y > 0:
            nvalue = mask[x,y-1]
            if nvalue == 0:
                ct = True
            elif nvalue != value:
                cc = True
        if y+1 < mask.shape[1]:
            nvalue = mask[x,y+1]
            if nvalue == 0:
                ct = True
            elif nvalue != value:
                cc = True
        return ct, cc
    for y in range(mask.shape[1]):
        for x in range(mask.shape[0]):
            value = mask[x,y]
            if value == 0:
                continue
            ct, cc = get_neighbors(mask, x, y)
            # Area
            stat_dict[value-1,0] += 1
            # Perimeter
            if ct or cc:
                stat_dict[value-1,1] += 1
            # Cell-tissue contribution
            if ct:
                stat_dict[value-1,2] += 1
            # Cell-cell contribution
            if cc:
                stat_dict[value-1,3] += 1
    return stat_dict
