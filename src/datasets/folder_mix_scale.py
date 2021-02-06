from .vision import VisionDataset

from PIL import Image
from .imresize import imresize

import imageio
import os
import os.path
import sys
import glob
import random
import transforms
import numpy as np
import torch

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 1024
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_hr(dir, extensions='.png', is_valid_file=None):
    '''
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                images.append(fname)
    '''
    images = glob.glob(os.path.join(dir, '*' + extensions))

    return images


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def imageio_loader(path):
    try:
        img = imageio.imread(path)
    except Exception as e:
        print('*' * 150)
        print(path)
        print('*' * 150)
        raise
    return img

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


def get_patch(lr, hr, patch_size, scale, is_same_size=False):
    ih, iw = lr.shape[:2]
    
    ip = patch_size
    if not is_same_size:
        ip = patch_size // scale

    iy = random.randrange(0, ih - ip + 1)
    ix = random.randrange(0, iw - ip + 1)

    ty, tx = iy, ix
    if not is_same_size:
        ty, tx = scale * iy, scale * ix

    lr = lr[iy:iy + ip, ix:ix + ip, :]
    hr = hr[ty:ty + patch_size, tx:tx + patch_size, :]

    return lr, hr 


class NoiseDataset(VisionDataset):
    def __init__(self, root, loader, size=32, image_extension='.png', transform=None, is_valid_file=None,):
        super(NoiseDataset, self).__init__(root=root)

        assert os.path.exists(root)

        self.root = root
        self.loader = loader
        #self.noise_imgs = sorted(glob.glob(base + '*.png'))
        self.size = size
        self.noise_imgs = make_dataset_hr(self.root, image_extension, is_valid_file)
        self.transform = transform

    def random_crop(self, img, size):
        h, w, c = img.shape
        rand_h = np.random.randint(0, h - size + 1)
        rand_w = np.random.randint(0, w - size + 1)
        return img[rand_h:rand_h+size, rand_w:rand_w+size, :]

    def __getitem__(self, index):
        #noise = self.pre_process(Image.open(self.noise_imgs[index]))
        noise = self.loader(self.noise_imgs[index])
        noise = self.random_crop(noise, self.size)
        noise = self.transform(noise)
        norm_noise = (noise - torch.mean(noise, dim=[1, 2], keepdim=True))
        return norm_noise

    def __len__(self):
        return len(self.noise_imgs)

class HRDatasetFolder(VisionDataset):
    def __init__(self, root, scale, patch_size=0, transform=None, 
        loader=imageio_loader, image_extension='.png', is_valid_file=None, is_same_size=False, lr_root=None, noise_root=None, noise_ratio=0.5):
        super(HRDatasetFolder, self).__init__(root, transform=transform)
        samples = make_dataset_hr(self.root, image_extension, is_valid_file)
        #if patch_size > 0:
        #    samples = 256 * samples
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root))

        self.root = root
        self.lr_root = lr_root

        self.scale= scale
        self.patch_size = patch_size

        self.loader = loader
        self.samples = samples
        self.transform = transform
        self.is_same_size = is_same_size
        self.noise_root = noise_root
        self.noise_ratio = noise_ratio
        if self.noise_root and self.noise_ratio > 0:
            noise_size = patch_size
            if not is_same_size:
                noise_size = patch_size // self.scale
            self.noise_dataset = NoiseDataset(noise_root, loader, size=noise_size, transform=self.transform)
            self.noise_dataset_len = len(self.noise_dataset)

        # add noise dataset

    def __getitem__(self, index):
        path = self.samples[index]
        fname = os.path.basename(path)
        hr_img = self.loader(path)
        # do inference if patch_size negaitve, only return hr and images file name
        if self.patch_size < 0:
            hr = hr_img
            if self.transform is not None:
                hr = self.transform(hr_img)
            return hr, os.path.splitext(fname)[0]

        scale = self.scale
        rnd = random.random()
        if rnd <= 0.7:
            scale = 2
        else:
            scale = 3

        h, w = hr_img.shape[:2]
        rescale_h = h // scale
        rescale_w = w // scale
        hr = hr_img[:rescale_h * scale, :rescale_w * scale, :]

        # Accept LR ground truth data, but need to keep exact same name as the HR file name
        if self.lr_root:
            lr_path = os.path.join(self.lr_root, fname) 
            lr = self.loader(lr_path)
        else:
            lr = imresize(hr, output_shape=(rescale_h, rescale_w)) 

        if self.is_same_size:
            lr = imresize(lr, output_shape=(rescale_h * scale, rescale_w * scale))

        if self.patch_size > 0:
            lr, hr = get_patch(lr, hr, self.patch_size, scale, self.is_same_size)

        if self.transform is not None:
            hr = self.transform(hr)
            lr = self.transform(lr)

        if self.noise_root and random.random() >= self.noise_ratio:
            noise = self.noise_dataset[np.random.randint(0, self.noise_dataset_len)]
            if self.is_same_size:
                noise = torch.round(noise).type(lr.type())
            lr = torch.clamp(lr + noise, 0, 255) 

        return hr, lr, fname

    def __len__(self):
        return len(self.samples)


