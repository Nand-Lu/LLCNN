import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageFilter
import numpy as np
import torchvision as vision


toPIL = vision.transforms.ToPILImage()


def noisy(img, std=3.0):
    mean = 0.0
    gauss = np.random.normal(mean, std, (img.height, img.width, 3))
    # noisy = np.clip(np.uint8(img + gauss), 0, 255)
    noisy = np.uint8(img + gauss)
    return noisy


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, low_image_dir,high_image_dir,input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames_low = [join(low_image_dir, x)
                                for x in listdir(low_image_dir) if is_image_file(x)]
        self.image_filenames_high = [join(high_image_dir, x)
                                    for x in listdir(high_image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        input = load_img(self.image_filenames_low[index])
        # target = input.copy()
        # if self.input_transform:
        #     if self.add_noise:
        #         input = noisy(input, self.noise_std)
        #         input = toPIL(input)
        #     input = self.input_transform(input)
        # if self.target_transform:
        #     target = self.target_transform(target)
        target = load_img(self.image_filenames_high[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.input_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames_low)
