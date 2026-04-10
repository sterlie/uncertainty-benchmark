import gzip
import sys
import torch
from torch.utils.data import Dataset
import struct
import numpy as np
from pathlib import Path
from multipledispatch import dispatch
try:
    from morphomnist import morpho, perturb
except ImportError as exc:
    vendor_root = Path(__file__).resolve().parents[2] / "Morpho-MNIST"
    if vendor_root.exists():
        sys.path.insert(0, str(vendor_root))
        from morphomnist import morpho, perturb
    else:
        raise ImportError(
            "morphomnist is required for MorphoMNISTDataset. "
            "Install it before using this dataset."
        ) from exc
from torchvision import transforms


def pars_gzip_file(path):
    '''
    a function to parse a gzipped file and return its data in numpy array format.
    :param path: path to gzipped file
    :return: ndarray data
    '''
    with gzip.open(path, 'rb') as f:
        idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        buffer_length = int(np.prod(shape))
        data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def write_gzip_file(path, array):
    with gzip.open(path, 'wb') as f:
        data = np.asarray(array, dtype=np.uint8)
        f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
        f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
        f.write(data.tobytes())

class UseMorpho(object):
    def __init__(self, thinning, thickening, swelling, fractures):
        self.perturbations = (
            perturb.Thinning(amount=thinning),
            perturb.Thickening(amount=thickening),
            perturb.Swelling(strength=swelling[0], radius=swelling[1]),
            perturb.Fracture(num_frac=fractures),
        )

    def __call__(self, image):
        morphology = morpho.ImageMorphology(image, scale=4)
        perturbation = self.perturbations[np.random.randint(len(self.perturbations))]
        perturbed_image = perturbation(morphology)
        perturbed_image = morphology.downscale(perturbed_image)
        return perturbed_image

    def __repr__(self):
        return f"{self.__class__.__name__}"

class MorphoMNISTDataset(Dataset):
    '''
    Custom dataset class for Morpho-MNIST dataset.
    data can be read from gzip file (as provided by morpho-mnist repo to download) or ndarray data.
    **both images and labels should be in the same format.
    morpho-mnist perturbation can be applied to the data while fetching and returning them.
    '''

    @dispatch(str, str)
    def __init__(self, images, labels, perturbation=None, transform=None, portion=0):
        if (isinstance(images, str) and images.endswith(".gz")) and (isinstance(labels, str) and labels.endswith(".gz")):   ## if the path to the gzip file of data is provided
            self.images = pars_gzip_file(images)
            self.labels = pars_gzip_file(labels)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.perturbation = perturbation    ## should be one of morpho-mnist perturbation function
        if portion!=0: self.get_subdata(portion)

    @dispatch(np.ndarray, np.ndarray)
    def __init__(self, images, labels, perturbation=None, transform=None, portion=0):   ## if raw data is in numpy format
        self.images = images
        self.labels = labels

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.perturbation = perturbation    ## should be one of morpho-mnist perturbation function
        if portion!=0: self.get_subdata(portion)


    @dispatch(torch.utils.data.Dataset)
    def __init__(self, dataset, perturbation=None, transform=None, portion=0):
        self.images, self.labels = [], []
        for item in dataset:
            self.images.append(item[0])
            self.labels.append(item[1])
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.perturbation = perturbation
        if portion!=0: self.get_subdata(portion)

    def get_subdata(self, portion):
        if portion<1: portion = portion*100
        portion = int(len(self.images)*(portion/100))
        self.images = self.images[:portion]
        self.labels = self.labels[:portion]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.perturbation is not None:       ## perturb image based on the perturbation function, this part is used morpho-mnist structure
            morphology = morpho.ImageMorphology(image, scale=4)
            res = self.perturbation(morphology)
            image = morphology.downscale(res)
        if self.transform is not None:
            image = self.transform(image)
        return torch.tensor(image).float(), torch.tensor(label).long()     ## return image (in raw or perturbed version) and corresponding label, both in tensor format

    def save_dataset(self, images_output_file, labels_output_file):
        """
        Apply __getitem__ transforms/perturbations to each sample
        and save the processed dataset to gzip files.
        """
        new_images = []
        new_labels = []

        for idx in range(len(self)):
            img, lbl = self[idx]          # <-- calls __getitem__ → transformations applied
            new_images.append(img.numpy())
            new_labels.append(lbl.numpy())

        new_images = np.stack(new_images)
        new_labels = np.stack(new_labels)

        write_gzip_file(images_output_file, new_images)
        write_gzip_file(labels_output_file, new_labels)
