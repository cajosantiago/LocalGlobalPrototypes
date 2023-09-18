import os.path

from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import torch

from PIL import Image
import numpy as np
import copy
from tqdm import tqdm
import pickle

datasets_dir = {'2019':
                    {'train': '/home/csantiago/Data/dermo/ISIC19_F1/Training/',
                     'val': '/home/csantiago/Data/dermo/ISIC19_F1/Validation/',
                     'test': '/home/csantiago/Data/dermo/ISIC19_F1/Test/',},
                'CBIS_roi':
                    {'train': '/home/csantiago/Data/breast/CBIS_roi/train/',
                     'test': '/home/csantiago/Data/breast/CBIS_roi/test/'},
                'CBIS':
                    {'train': '/home/csantiago/Data/breast/DDSM+CBIS+MIAS_CLAHE/train/',
                     'test': '/home/csantiago/Data/breast/DDSM+CBIS+MIAS_CLAHE/test/'},
                }

def gray_pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        image = np.array(Image.open(f))

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), 256, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (cdf - cdf.min())*255/(cdf.max()-cdf.min())  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image = image_equalized.reshape(image.shape)

    return Image.fromarray(np.uint8(image), 'L')

def gray_pil_loader_wo_he(path: str) -> Image.Image:
    image = Image.open(path)
    return image.convert('L').resize((max(image.size),max(image.size)), resample=Image.BILINEAR)

def get_dataloaders(dataset, inputsize=224, batch_size=64, numworkers=0, preload_data=True):

    # Data sets
    trainfolder = datasets_dir[dataset]['train']
    valfolder = datasets_dir[dataset]['val'] if dataset=='2019' else trainfolder
    # testfolder = datasets_dir[dataset]['test']

    # data augmentation transformations
    if 'CBIS' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize([int(1.3*inputsize), int(1.3*inputsize)]),
            transforms.RandomCrop(inputsize, padding=0),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize([inputsize, inputsize]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        loader = gray_pil_loader_wo_he if dataset=='CBIS' else gray_pil_loader
    else:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize([int(1.3*inputsize), int(1.3*inputsize)]),
            transforms.RandomCrop(inputsize, padding=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize([inputsize, inputsize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        loader = default_loader
    if preload_data:
        train_set = ImageFolder_preload(trainfolder, transform=train_transforms, loader=loader, inputsize=inputsize)
        val_set = ImageFolder_preload(valfolder, transform=test_transforms, loader=loader, inputsize=inputsize)
    else:
        train_set = datasets.ImageFolder(trainfolder, transform=train_transforms, loader=loader)
        val_set = datasets.ImageFolder(valfolder, transform=test_transforms, loader=loader)

    if 'CBIS' in dataset:
        if dataset == 'CBIS':
            train_set.classes = ['benign', 'malignant']
            val_set.classes = ['benign', 'malignant']
            # train_set.samples = [s for s in train_set.samples if 'Mass-' in s[0]]
            # val_set.samples = [s for s in val_set.samples if 'Mass-' in s[0]]
            train_set.samples = [s for s in train_set.samples if s[1]<2]
            val_set.samples = [s for s in val_set.samples if s[1]<2]

        #train-val split
        torch.manual_seed(0)
        splt = torch.split(torch.randperm(len(train_set.samples)), int(.8 * len(train_set.samples)))
        train_set.samples = [s for [i, s] in enumerate(train_set.samples) if i in splt[0]]
        val_set.samples = [s for [i, s] in enumerate(val_set.samples) if i in splt[1]]

    # Pre-load datasets
    if hasattr(train_set, 'preloaded_data'):
        train_set.preload_data('training')
    if hasattr(val_set, 'preloaded_data'):
        val_set.preload_data('validation')

    train_set_notransform = copy.deepcopy(train_set)
    train_set_notransform.transform = test_transforms

    # Balanced batches
    class_counts = len(train_set.classes)*[0]
    labels = []
    for i in range(len(train_set)):
        c = train_set.samples[i][1]
        labels.append(c)
        class_counts[c] += 1
    num_samples = sum(class_counts)
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=numworkers)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler,
    #                                           pin_memory=(torch.cuda.is_available()), num_workers=numworkers)
    train_loader_noshuffle = torch.utils.data.DataLoader(train_set_notransform, batch_size=batch_size, shuffle=False,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=numworkers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=numworkers)

    return train_loader, train_loader_noshuffle, val_loader


def get_testloader(dataset, inputsize=224, batch_size=64, numworkers=0, preload_data=True):

    # Data sets
    testfolder = datasets_dir[dataset]['test']

    # transformations
    if 'CBIS' in dataset:
        test_transforms = transforms.Compose([
            transforms.Resize([inputsize, inputsize]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        loader = gray_pil_loader
    else:
        test_transforms = transforms.Compose([
            transforms.Resize([inputsize, inputsize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        loader = default_loader
    if preload_data:
        test_set = ImageFolder_preload(testfolder, transform=test_transforms, loader=loader, inputsize=inputsize)
    else:
        test_set = datasets.ImageFolder(testfolder, transform=test_transforms, loader=loader)

    if dataset == 'CBIS':
        test_set.classes = ['benign', 'malignant']
        test_set.samples = [s for s in test_set.samples if 'Mass-' in s[0]]

    # Pre-load datasets
    if hasattr(test_set, 'preloaded_data'):
        test_set.preload_data('test')

    # Data loaders
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=numworkers)

    return test_loader


class ImageFolder_preload(datasets.ImageFolder):
    def __init__(self, root, loader=default_loader, transform=None, target_transform=None, inputsize=224):
        super(ImageFolder_preload, self).__init__(root, loader=loader,transform=transform,target_transform=target_transform)
        self.root = root
        self.preloaded_data = []
        self.inputsize = (inputsize,inputsize)
    def preload_data(self, set_str=''):
        data = []
        print('Preloading %s dataset...' %set_str)
        datasetfile = os.path.join(self.root,'sz_' + str(self.inputsize[0]) + '.pickle')
        if os.path.exists(datasetfile):
            with open(datasetfile, 'rb') as handle:
                data = pickle.load(handle)
        else:
            for i in tqdm(range(len(self.samples))):
                path, target = self.samples[i]
                sample = self.loader(path).resize(self.inputsize)
                data.append((sample, target))
            with open(datasetfile, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.preloaded_data = data
    def __getitem__(self, item):
        if self.transform is not None:
            sample = self.transform(self.preloaded_data[item][0])
        return sample, self.preloaded_data[item][1]
    def __len__(self):
        return len(self.preloaded_data)

def get_singleimage(file, inputsize=224):

    # transformations
    inference_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return inference_transforms(default_loader(file).resize((inputsize,inputsize))).unsqueeze(0)