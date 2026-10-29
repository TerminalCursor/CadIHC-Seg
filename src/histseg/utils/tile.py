def get_shape(np_arr):
    from numpy import ndarray

    assert isinstance(np_arr, ndarray), "Not a numpy array"
    if len(np_arr.shape) == 2:
        return (*np_arr.shape,)
    w, h, c = np_arr.shape
    if w <= 4:
        c, w, h = np_arr.shape
    return w, h, c


class Tiler:
    def __init__(
        self,
        image_index,
        im_ma_t,
        tile_w=256,
        tile_h=256,
        stride_w=1,
        stride_h=1,
        threshold=6.0 / 10.0,
    ):
        self.image_index = image_index
        self.tile_w = tile_w
        self.tile_h = tile_h
        image = im_ma_t[0][image_index]
        mask = im_ma_t[1][image_index]
        w, h, c = get_shape(image)
        assert (w, h) == mask.shape, "Image-Mask Size Inconsistency"
        assert (w >= tile_w) and (h >= tile_h), "Tile too large for Image"
        assert (stride_w > 0) and (stride_h > 0), "Positive Strides Required"
        self.points = mask2tiles(
            mask, (tile_w, tile_h), (stride_w, stride_h), threshold=threshold
        )
        self.point_meta = dict([(i, []) for i in range(self.len())])

    def eval(self, idx, *results):
        assert isinstance(results, tuple)
        assert idx < self.len()
        self.point_meta[idx].append(results)

    def get(self, im_ma_t, point_index):
        objs = [im_ma_t[idx][self.image_index] for idx in range(len(im_ma_t))]
        point_index = (
            point_index % len(self.points)
            if point_index >= 0
            else -((-point_index - 1) % len(self.points))
        )
        point = self.points[point_index]
        x_slice = slice(point[0], point[0] + self.tile_w)
        y_slice = slice(point[1], point[1] + self.tile_h)
        rets = []
        for obj in objs:
            if len(obj.shape) == 3:
                if obj.shape[0] <= 4:
                    rets.append(obj[:, x_slice, y_slice].transpose(1, 2, 0))
                rets.append(obj[x_slice, y_slice, :])
            rets.append(obj[x_slice, y_slice, ...])
        return rets

    def len(self):
        return len(self.points)


SymTiler = lambda ii, imt, td, sd, th=6.0 / 10.0: Tiler(
    ii, imt, tile_w=td, tile_h=td, stride_w=sd, stride_h=sd, threshold=th
)
SymSweepTiler = lambda ii, imt, td, th=6.0 / 10.0: Tiler(
    ii, imt, tile_w=td, tile_h=td, stride_w=td / 2, stride_h=td / 4, threshold=th
)


def point2slice(tiler, point):
    """# Convert a point to a Slice

    Use tiler data to find width and height of slice
    """
    x_slice = slice(point[0], point[0] + tiler.tile_w)
    y_slice = slice(point[1], point[1] + tiler.tile_h)
    return x_slice, y_slice


def sliceNimNmask2tuple(slices, im, ma):
    """# Slice an Image and Mask Simultaneously"""
    if im.shape[0] <= 4:
        return im[:, slices[0], slices[1]], ma[slices[0], slices[1]]
    return im[slices[0], slices[1], :], ma[slices[0], slices[1]]


def mask2tiles(mask, tile_dim, strides, threshold=6.0 / 10.0):
    from scipy.signal import convolve2d
    from scipy.spatial import KDTree
    from numpy import ones, argwhere, arange, array, meshgrid, unique

    assert len(tile_dim) == 2, "2D Tiles Only"
    binary_mask = (mask > 0).astype(float)
    kernel = ones(tile_dim, float)
    threshold_v = threshold * kernel.sum()
    conv = convolve2d(binary_mask, kernel, mode="valid")
    valid = conv > threshold_v
    initial_points = argwhere(valid)
    sample_x = arange(0, conv.shape[0], strides[0])
    sample_y = arange(0, conv.shape[1], strides[1])
    xy = array(meshgrid(sample_x, sample_y))
    xy = xy.reshape((2, -1)).T
    model = KDTree(initial_points)
    _d, sample = model.query(xy, 1)
    sample = unique(sample)
    return initial_points[sample, ...]


def prepare_data(base_directory, half_res=True, tile=None):
    """# Prepare Data"""
    from .image import load_data

    train_data, test_data = load_data(base_directory)
    if half_res:
        train_data = (
            [train_data[0][idx][:, ::2, ::2] for idx in range(len(train_data[0]))],
            [train_data[1][idx][::2, ::2] for idx in range(len(train_data[1]))],
        )
        test_data = (
            [test_data[0][idx][:, ::2, ::2] for idx in range(len(test_data[0]))],
            [test_data[1][idx][::2, ::2] for idx in range(len(test_data[1]))],
        )
    train_images = train_data[0]
    train_labels = train_data[1]
    test_images = test_data[0]
    test_labels = test_data[1]
    if tile is not None:
        train_tilers = [
            SymSweepTiler(idx, train_data, tile) for idx in range(len(train_data[0]))
        ]
        test_tilers = [
            SymSweepTiler(idx, test_data, tile) for idx in range(len(test_data[0]))
        ]
        return (
            (train_images, train_labels, train_tilers),
            (test_images, test_labels, test_tilers),
        )
    return (
        (train_images, train_labels),
        (test_images, test_labels),
    )
