import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

class Transformations:

    def __init__(
        self, horizontal_flip_prob=0.0, vertical_flip_prob=0.0, gaussian_blur_prob=0.0,
        rotate_degree=0.0, cutout=0.0, cutout_height=0, cutout_width=0,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), train=True
    ):
        transforms_list = []

        if train:
            if horizontal_flip_prob > 0:  # Horizontal Flip
                transforms_list += [A.HorizontalFlip(p=horizontal_flip_prob)]
            if vertical_flip_prob > 0:  # Vertical Flip
                transforms_list += [A.VerticalFlip(p=vertical_flip_prob)]
            if gaussian_blur_prob > 0:  # Patch Gaussian Augmentation
                transforms_list += [A.GaussianBlur(p=gaussian_blur_prob)]
            if rotate_degree > 0:  # Rotate image
                transforms_list += [A.Rotate(limit=rotate_degree)]
            if cutout > 0:  # CutOut
                transforms_list += [A.CoarseDropout(
                    p=cutout, max_holes=1, fill_value=tuple([x * 255.0 for x in mean]),
                    max_height=cutout_height, max_width=cutout_width, min_height=1, min_width=1
                )]
        
        transforms_list += [
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensor()
        ]

        self.transform = A.Compose(transforms_list)
    
    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image