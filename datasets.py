import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def check_image_file(filename):
    r"""Determine whether the files in the directory are in image format.
    Args:
        filename (str): The current path of the image
    Returns:
        Returns True if it is an image and False if it is not.
    """
    return any(filename.endswith(extension) for extension in [".bmp", ".BMP",
                                                              ".jpg", ".JPG",
                                                              ".png", ".PNG",
                                                              ".jpeg", ".JPEG"])


class TrainingDataset(Dataset):
    def __init__(self, images_dir, image_size=256, scale_factor=4):
        """ Dataset loading base class.
       :parameter
            images_dir (str): The directory address where the image is stored.
            image_size (int): Original high resolution image size. Default: 256.
            scale_factor (int): Coefficient of image scale. Default: 4.
        """
        super(TrainingDataset, self).__init__()
        self.image_filenames = [os.path.join(images_dir, x) for x in
                                os.listdir(images_dir)
                                if check_image_file(x)]

        crop_size = image_size - (image_size % scale_factor)  # Valid crop size
        self.input_transform = transforms.Compose(
            [transforms.CenterCrop(crop_size),  # cropping the image
             transforms.Resize(crop_size // scale_factor),
             transforms.Resize(crop_size),
             transforms.ToTensor()])
        self.target_transform = transforms.Compose(
            [transforms.CenterCrop(crop_size),
             transforms.ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        image = image.convert("RGB")
        target = image.copy()

        inputs = self.input_transform(image)
        target = self.target_transform(target)

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)

class ValidationDataset(Dataset):
    def __init__(self, images_dir, scale_factor=4):
        """ Dataset loading base class.
       :parameter
            images_dir (str): The directory address where the image is stored.
            scale_factor (int): Coefficient of image scale. Default: 4.
        """
        super(ValidationDataset, self).__init__()
        self.image_filenames = [os.path.join(images_dir, x) for x in
                                os.listdir(images_dir)
                                if check_image_file(x)]

        self.input_transform = transforms.Compose(
            [transforms.ToTensor()])
        self.target_transform = transforms.Compose(
            [transforms.ToTensor()])
        self.scale = scale_factor

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        image = image.convert("RGB")
        target = image.copy()
        image = image.resize((image.width // self.scale, image.height // self.scale), Image.BILINEAR)
        image = image.resize((image.width * self.scale, image.height * self.scale), Image.BILINEAR)

        inputs = self.input_transform(image)
        target = self.target_transform(target)

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)
