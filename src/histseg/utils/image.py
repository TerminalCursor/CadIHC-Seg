def load_image(ffile):
    image = plt.imread(ffile)
    assert len(image.shape) == 3
    if image.shape[0] > 4:
        image = image.transpose(2, 0, 1)
    if image.shape[0] == 4:
        image = image[:3, ...]
    image = transform_image(image)
    return image[:, ::2, ::2]

def add_peak(df, x, y, s, r, t):
    from numpy import sin, cos
    from itertools import product
    for _r in range(int(r),0,-1):
        xm = max(0,(int(x) - _r))
        xM = min(df.shape[0]-1,(int(x) + _r))+1
        ym = max(0,(int(y) - _r))
        yM = min(df.shape[1]-1,(int(y) + _r))+1
        xr = range(xm,xM)
        yr = range(ym,yM)
        for (_x, _y) in product(xr, yr):
            d = ((_x - x) ** 2. + (_y - y) ** 2.) - _r ** 2.
            if (d > 0):
                continue
            S = s * ((r-_r)/r)
            df[_x,_y,0] = S * sin(t)
            df[_x,_y,1] = S * cos(t)
    return df


def perterb_image_and_mask(image, mask, count=30, radius=60, pinch_strength=-12, _debug=True):
    from importlib.util import find_spec

    if find_spec("SimpleITK") is None:
        raise NotImplementedError
    if find_spec("numpy") is None:
        raise NotImplementedError
    import SimpleITK as sitk
    import numpy as np
    from matplotlib import pyplot as plt
    from copy import deepcopy as dc

    N_centers = count
    data_dict = {
        "masks": mask,
        "image": image,
    }
    s_Mask = data_dict["masks"]
    s_Image = sitk.GetImageFromArray(s_Mask)

    # 3. Generate a displacement field for pinching
    #    Initialize a displacement field with zeros (no displacement initially)
    image_size = s_Image.GetSize()
    displacement_field = np.zeros(
        (*image_size, s_Image.GetDimension()), dtype=np.float64
    )
    displacement_field = displacement_field.transpose(1,0,2)
    displacement_fields = np.zeros(
        (*image_size, s_Image.GetDimension(), N_centers), dtype=np.float64
    )
    displacement_fields = displacement_fields.transpose(1,0,2,3)

    centers = np.random.randint(
        (radius + 1, radius + 1),
        (image_size[1] - radius - 1, image_size[0] - radius - 1),
        size=(N_centers, 2),
    )

    # Create a light drag in a random direction
    for idx,center in enumerate(centers):
        theta = np.random.rand() * np.pi * 2
        displacement_field = add_peak(displacement_field, center[0], center[1], np.abs(pinch_strength), radius, theta)
        displacement_fields[...,idx] = add_peak(displacement_fields[...,idx], center[0], center[1], np.abs(pinch_strength), radius, theta)

    # Convert the numpy array displacement field to a SimpleITK image
    displacement_images = [sitk.GetImageFromArray(displacement_fields[...,idx], isVector=True) for idx in range(displacement_fields.shape[-1])]
    for idx in range(len(displacement_images)):
        displacement_images[idx].SetSpacing(s_Image.GetSpacing())
        displacement_images[idx].SetOrigin(s_Image.GetOrigin())
        displacement_images[idx].SetDirection(s_Image.GetDirection())
    displacement_transforms = [sitk.DisplacementFieldTransform(displacement_images[idx]) for idx in range(len(displacement_images))]

    interpolator = sitk.sitkNearestNeighbor
    n_Image = dc(s_Image)
    for displacement_transform in displacement_transforms:
        n_Image = sitk.Resample(
            n_Image,
            n_Image.GetSize(),
            displacement_transform,
            interpolator,
            n_Image.GetOrigin(),
            n_Image.GetSpacing(),
            n_Image.GetDirection(),
            0.0,  # Default pixel value for regions outside the input image
            n_Image.GetPixelID(),
        )
    transformed_mask = dc(n_Image)
    del n_Image

    final_mask = sitk.GetArrayFromImage(transformed_mask)

    interpolator = sitk.sitkLinear
    channels = []
    for idx in range(3):
        n_Image = sitk.GetImageFromArray(data_dict["image"][...,idx])
        for displacement_transform in displacement_transforms:
            n_Image = sitk.Resample(
                n_Image,
                n_Image.GetSize(),
                displacement_transform,
                interpolator,
                n_Image.GetOrigin(),
                n_Image.GetSpacing(),
                n_Image.GetDirection(),
                0.0,  # Default pixel value for regions outside the input image
                n_Image.GetPixelID(),
            )
        channel = sitk.GetArrayFromImage(n_Image)
        channels.append(channel[...,None])

    final_image = np.dstack(channels)

    if _debug:
        from histseg.utils.colors import color_mask
        displacement_field = displacement_fields.sum(-1) / displacement_fields.shape[-1]
        df_z = np.zeros((displacement_field.shape[0], displacement_field.shape[1],3))
        df_z[...,0] = displacement_field[...,0]
        df_z[...,1] = displacement_field[...,1]
        df_z[...,:2] = np.abs(df_z[...,:2])
        df_z[...,:2] = df_z[...,:2] - df_z[...,:2].min()
        df_z[...,:2] = df_z[...,:2] / df_z[...,:2].max()
        plt.imsave("oim.png", data_dict["image"])
        plt.imsave("im.png", final_image)
        plt.imsave("ma.png", color_mask(final_mask))
        plt.imsave("df.png", df_z)
    return final_mask, final_image


def grayscale_image_and_mask(_image, _mask):
    from numpy import vstack

    assert _image.shape[0] == 3
    grayed = _image.mean(axis=0)
    gray = vstack([grayed[None, ...], grayed[None, ...], grayed[None, ...]])
    return _mask, gray


def rechannel_image_and_mask(_image, _mask):
    from numpy.random import permutation

    assert _image.shape[0] == 3
    image = _image[permutation(3), ...]
    return _mask, image


def transform_image(
    im,
    baseline=[
        (0.66652524, 0.1445229),
        (0.62870604, 0.18943928),
        (0.59433895, 0.22827524),
    ],
    im_range=[0, 1],
):
    from numpy import clip, array

    chans = []
    transposed = False
    if im.shape[0] == 3:
        transposed = True
        im = im.T
    R, G, B = ((im - im_range[0]) / (im_range[1] - im_range[0])).reshape((-1, 3)).T
    for idx, channel in enumerate([R, G, B]):
        var_matched = channel * (baseline[idx][1] / channel.std())
        mean_matched = var_matched + (baseline[idx][0] - var_matched.mean())
        chans.append(clip(mean_matched, 0, 1))
    out_im = array(chans).T.reshape(im.shape)
    out_im = out_im * (im_range[1] - im_range[0]) + im_range[0]
    if transposed:
        out_im = out_im.T
    return out_im


def noise_image_and_mask(_image, _mask, noise_level=1e-2):
    from numpy.random import rand
    from numpy import clip

    assert _image.shape[0] == 3
    noise_filter = rand(*_image.shape) * 2 - 1
    noise_filter *= noise_level
    image = _image + noise_filter
    image = clip(image, 0, 1)
    return _mask, image


def renormalize_image_and_mask(_image, _mask, non_standard_renormalization=False):
    from numpy.random import rand

    assert _image.shape[0] == 3
    baseline = [
        (0.66652524, 0.1445229),
        (0.62870604, 0.18943928),
        (0.59433895, 0.22827524),
    ]
    eps = 0.07, 0.01
    new_baseline = []
    if non_standard_renormalization:
        for a, b in baseline:
            new_baseline.append(
                (
                    a + ((2 * rand() - 1) * eps[0]),
                    b + ((2 * rand() - 1) * eps[1]),
                )
            )
    else:
        new_baseline = baseline
    image = transform_image(_image, baseline=new_baseline)
    return _mask, image


def augment_image_and_mask(
    _image,
    _mask,
    count=30,
    radius=64,
    pinch_strength=-12,
    noise_level=1e-2,
    perterb_p=1.0,
    debug=False,
):
    from numpy.random import rand, randint

    channel_first = _image.shape[0] == 3
    perturb_kwargs = {
        "count": count,
        "radius": radius,
        "pinch_strength": pinch_strength,
        "_debug": debug,
    }
    renormalize_kwargs = {
        "non_standard_renormalization": rand() > 0.25,
    }
    noise_kwargs = {
        "noise_level": noise_level,
    }
    perterbed = False
    image, mask = _image, _mask
    if channel_first:
        image = _image.transpose(1, 2, 0)
    if rand() < perterb_p:
        perterbed = True
        mask, image = perterb_image_and_mask(image, mask, **perturb_kwargs)
    augment_functions = [
        grayscale_image_and_mask,
        rechannel_image_and_mask,
        lambda i, m: renormalize_image_and_mask(i, m, **renormalize_kwargs),
        lambda i, m: noise_image_and_mask(i, m, **noise_kwargs),
        lambda i, m: (m, i),
    ]
    augment_index = randint(len(augment_functions))
    image = image.transpose(2, 0, 1)
    if debug:
        import json

        print(perterb_p, perterbed)
        print(json.dumps(perturb_kwargs, indent=2))
        print(json.dumps(renormalize_kwargs, indent=2))
        print(augment_index)
    augment_function = augment_functions[augment_index]
    mask, image = augment_function(image, mask)
    if not channel_first:
        image = image.transpose(1, 2, 0)
    return mask, image


def load_data(base_directory, image_suffix="", mask_suffix="_seg.npy"):
    from cellpose import io

    train_dir = base_directory + "/train/"
    test_dir = base_directory + "/test/"
    train_images, train_masks, _, test_images, test_masks, _ = io.load_train_test_data(
        train_dir, test_dir, image_filter=image_suffix, mask_filter=mask_suffix
    )
    _preproc_fn = lambda x: x
    _preproc_fn = transform_image
    train_images = [_preproc_fn(train_images[idx]) for idx in range(len(train_images))]
    test_images = [_preproc_fn(test_images[idx]) for idx in range(len(test_images))]
    return (train_images, train_masks), (test_images, test_masks)


def generate_augmentations(data, config, progress=True):
    from sys import stderr as se
    from tqdm.auto import tqdm
    if progress:
        pbar = tqdm
    else:
        pbar = lambda x, *args, **kwargs: x

    w = se.write
    augmented_images, augmented_labels = [], []
    w(f"Generating {config.training.n_augments} augments")
    for image, label in pbar(zip(data[0], data[1])):
        for _ in range(config.training.n_augments):
            augmented_label, augmented_image = augment_image_and_mask(
                image, label, **config.augments.todict()
            )
            augmented_images.append(augmented_image)
            augmented_labels.append(augmented_label)
    return augmented_images, augmented_labels
