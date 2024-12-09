import imgaug


class Augmentation:
    @staticmethod
    def get_full():
        return imgaug.augmenters.Sometimes(0.01, imgaug.augmenters.Sequential([
            imgaug.augmenters.OneOf([
                imgaug.augmenters.Fliplr(0.5),  # horizontal flips
                imgaug.augmenters.Flipud(0.5),  # upside down flips
            ]),
            # imgaug.augmenters.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 5% of all images.
            # imgaug.augmenters.Sometimes(0.05,
            #                             imgaug.augmenters.GaussianBlur(sigma=(0, 0.5))
            #                             ),
            # Strengthen or weaken the contrast in each image.
            # imgaug.augmenters.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            # imgaug.augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            # imgaug.augmenters.Sometimes(0.2,
            #                             imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
            #                             ),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            imgaug.augmenters.Affine(shear=(-8, 8)),
            imgaug.augmenters.OneOf([
                imgaug.augmenters.Affine(rotate=(-45, 45)),
                imgaug.augmenters.Affine(rotate=(-90, 90)),
            ]),

            imgaug.augmenters.OneOf([
                imgaug.augmenters.Affine(scale=(0.5, 0.8)),
                imgaug.augmenters.Affine(scale=(1 / 1.2, 1 / 1.5)),
                imgaug.augmenters.Affine(scale=(1.8, 2.1)),
                imgaug.augmenters.Affine(scale=(2.4, 2.7)),
                imgaug.augmenters.Affine(scale=(1 / 3, 1 / 3.3)),
                imgaug.augmenters.Affine(scale=(1 / 3.6, 1 / 3.9)),

            ])

        ], random_order=True))
