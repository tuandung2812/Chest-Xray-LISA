import torch


def normalize_image(
    image,
    pixel_mean,
    pixel_std,
    image_size_target):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    image = (image - pixel_mean) / pixel_std

    # Pad
    h, w = image.shape[-2:]
    padh = image_size_target - h
    padw = image_size_target - w
    image = torch.nn.functional.pad(image, (0, padw, 0, padh))
    return image