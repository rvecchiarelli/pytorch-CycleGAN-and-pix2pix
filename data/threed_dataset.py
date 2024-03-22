"""Code adapted from Aligned Dataset and https://github.com/neoamos/3d-pix2pix-CycleGAN

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import os
from data.image_folder import make_dataset
import h5py
import torch
import numpy as np



class ThreeDDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--cells_x', type=int, default=48, help='number of cells x direction')
        parser.add_argument('--cells_y', type=int, default=48, help='number of cells y direction')
        parser.add_argument('--cells_z', type=int, default=48, help='number of cells z direction')


        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.dir_paths = os.path.join(opt.dataroot, opt.phase) 
        self.image_paths = sorted(make_dataset(self.dir_paths, opt.max_dataset_size)) # get the image directory  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """


        path = self.image_paths[index]    # needs to be a string
        #f = h5py.File(path, 'r').get('dataset_1')
        data_A = torch.reshape(torch.tensor(np.array(h5py.File(path, 'r').get('data_A')))[0:32,0:32,0:32], (1,32,32,32))
        #data_A = torch.reshape(torch.tensor(h5py.File(path, 'r').get('data_A')), (1,48,48,48))
        #print(torch.Tensor.size(data_A))  # needs to be a tensor

        data_B = torch.reshape(torch.tensor(np.array(h5py.File(path, 'r').get('data_B')))[0:32,0:32,0:32], (1,32,32,32))
        #data_B = torch.reshape(torch.tensor(h5py.File(path, 'r').get('data_B')), (1,48,48,48))    # needs to be a tensor

        return {'A': data_A, 'B': data_B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
