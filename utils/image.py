def perterb_image_and_mask(image, mask, count=30, radius=60, pinch_strength=-12):
    from importlib.util import find_spec

    if find_spec("SimpleITK") is None:
        raise NotImplementedError
    if find_spec("numpy") is None:
        raise NotImplementedError
    import SimpleITK as sitk
    import numpy as np

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
    d_size = displacement_field.shape

    radius, pinch_strength = 64, -12.0
    N_centers = count
    centers = np.random.randint(
        (radius + 1, radius + 1),
        (image_size[0] - radius - 1, image_size[1] - radius - 1),
        size=(N_centers, 2),
    )

    # Create a pinching effect:
    for center in centers:
        xmin = max(0, center[0] - radius)
        xmax = min(image_size[0], center[0] + radius)
        ymin = max(0, center[1] - radius)
        ymax = min(image_size[1], center[1] + radius)
        _dir = np.random.randint((-1, -1), (1 + 1, 1 + 1))
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                if distance < radius:
                    # Calculate the direction towards the center
                    direction_x = (center[0] - x) / distance if distance > 0 else 0
                    direction_y = (center[1] - y) / distance if distance > 0 else 0

                    # Scale the displacement based on proximity to the center and a pinching strength
                    displacement_x = (
                        _dir[0]
                        * np.abs(direction_x * (radius - distance) / radius)
                        * pinch_strength
                    )
                    displacement_y = (
                        _dir[1]
                        * np.abs(direction_y * (radius - distance) / radius)
                        * pinch_strength
                    )

                    displacement_field[x, y, 0] = displacement_x
                    displacement_field[x, y, 1] = displacement_y

    # Convert the numpy array displacement field to a SimpleITK image
    displacement_image = sitk.GetImageFromArray(displacement_field, isVector=True)
    displacement_image.SetSpacing(s_Image.GetSpacing())
    displacement_image.SetOrigin(s_Image.GetOrigin())
    displacement_image.SetDirection(s_Image.GetDirection())
    displacement_transform = sitk.DisplacementFieldTransform(displacement_image)

    interpolator = sitk.sitkNearestNeighbor
    transformed_mask = sitk.Resample(
        s_Image,
        s_Image.GetSize(),
        displacement_transform,
        interpolator,
        s_Image.GetOrigin(),
        s_Image.GetSpacing(),
        s_Image.GetDirection(),
        0.0,  # Default pixel value for regions outside the input image
        s_Image.GetPixelID(),
    )

    final_mask = sitk.GetArrayFromImage(transformed_mask)

    interpolator = sitk.sitkLinear

    image_r = sitk.GetImageFromArray(data_dict["image"][..., 0])
    transformed_image_r = sitk.Resample(
        image_r,
        image_r.GetSize(),
        displacement_transform,
        interpolator,
        image_r.GetOrigin(),
        image_r.GetSpacing(),
        image_r.GetDirection(),
        0.0,  # Default pixel value for regions outside the input image
        image_r.GetPixelID(),
    )
    red_channel = sitk.GetArrayFromImage(transformed_image_r)

    image_g = sitk.GetImageFromArray(data_dict["image"][..., 1])
    transformed_image_g = sitk.Resample(
        image_g,
        image_g.GetSize(),
        displacement_transform,
        interpolator,
        image_g.GetOrigin(),
        image_g.GetSpacing(),
        image_g.GetDirection(),
        0.0,  # Default pixel value for regions outside the input image
        image_g.GetPixelID(),
    )
    green_channel = sitk.GetArrayFromImage(transformed_image_g)

    image_b = sitk.GetImageFromArray(data_dict["image"][..., 2])
    transformed_image_b = sitk.Resample(
        image_b,
        image_b.GetSize(),
        displacement_transform,
        interpolator,
        image_b.GetOrigin(),
        image_b.GetSpacing(),
        image_b.GetDirection(),
        0.0,  # Default pixel value for regions outside the input image
        image_b.GetPixelID(),
    )
    blue_channel = sitk.GetArrayFromImage(transformed_image_b)

    final_image = np.dstack(
        [red_channel[..., None], green_channel[..., None], blue_channel[..., None]]
    )

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


def generate_augmentations(data, config):
    from sys import stderr as se

    w = se.write
    augmented_images, augmented_labels = [], []
    w(f"Generating {config.training.n_augments} augments")
    for image, label in zip(data[0], data[1]):
        for _ in range(config.training.n_augments):
            augmented_label, augmented_image = augment_image_and_mask(
                image, label, **config.augments.todict()
            )
            augmented_images.append(augmented_image)
            augmented_labels.append(augmented_label)
    return augmented_images, augmented_labels
