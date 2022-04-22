import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader


class NumpyDataset(Dataset):
  def __init__(self,x,y,indexes=None,x_transforms_func = None):

    self.x = x[indexes]
    self.y = y[indexes]

    # Transforms that will be aplied to the every batch of lenses.
    # x_transforms_func must be callable.
    self.x_transforms = x_transforms_func


  def __len__(self):
    # Returns the length of the dataset
    return self.x.shape[0]

  def __getitem__(self, idx):
    # Returns an (image, label) tuple
    image, label = self.x[idx], self.y[idx]
    
    # Convert to Tensor and Float
    image = torch.tensor(image).float()
    label = torch.tensor(label).float()

    # Apply transforms
    if self.x_transforms!=None:
      image= self.x_transforms(image)

    return image, label


class ProcessDataset():
    def __init__(self,DATASET_PATH):
        self.path = DATASET_PATH
        self.data = None
        self.value = None
        # Normalized versions of the input
        self.data_norm = None
        self.value_norm = None

    def load(self):
        print('\tLoading data set: ',end='')
        images = []
        # axion mass 
        axion_mass = []
        for f_name in os.listdir(self.path):
          try:
            img, mass, _ = np.load(os.path.join(self.path,f_name),allow_pickle=True)
            # Add img and mass to separate lists
            # Add 1 as the first dimension for image
            images.append(img.reshape(1,img.shape[0],img.shape[1]))
            # Convert mass to a single element array with (1,1) dimensions
            axion_mass.append(np.array(np.log10(mass),ndmin=1))
          except:
            # Tossing out bad file
            pass

        # Images shape is (num_of_images,1,150,150)
        self.data = np.stack(images).astype('float32')
        # Axion mass shape is (num_of_images,1)
        self.value = np.stack(axion_mass).astype('float32')

        print('Complete')


    def normalize(self):
        print('\tNormalizing data set: ',end='')
        # Find stats of the dataset
        DATA_MEAN, DATA_STD = self.data.mean(), self.data.std()
        VALUE_MEAN, VALUE_STD = self.value.mean(), self.value.std()

        # Standardize the dataset
        self.data_norm = self.standardize(self.data,DATA_STD,DATA_MEAN)
        self.value_norm = self.standardize(self.value,VALUE_STD,VALUE_MEAN)
        print('Complete')
        return DATA_STD, DATA_MEAN, VALUE_STD, VALUE_MEAN

    """
        standardize - perform z-standardization of data set such that mean=0 and std=1
    """
    def standardize(self,element,STD,MEAN):
        return (element - MEAN) / STD

    """
        inv_standardize - reverse z-standardization
    """
    def inv_standardize(self,element,STD,MEAN):
        return element * STD + MEAN

def get_dataloaders(dat_dir,batch_size,train_test_split,return_stats=False):
    print('----------------------------------------------------')
    print('Constructing data loaders')
    print('----------------------------------------------------\n')

    # Make data set object
    data = ProcessDataset(dat_dir)
    # Load the data
    data.load()
    # Normalize the data
    stats = data.normalize()

    # Get the size of data set
    size = len(data.data_norm)

    # Grab ids of train test set
    train_indx = np.arange(0,int(train_test_split*size))
    test_indx  = np.arange(size-int(train_test_split*size),size)

    # Split data set into train/test data sets
    train_dataset = NumpyDataset(data.data_norm, data.value_norm, train_indx)
    test_dataset  = NumpyDataset(data.data_norm, data.value_norm, test_indx)

    # Create train/test dataloaders
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('\n\tDONE.\n')
    print('----------------------------------------------------')

    # Check if there is flag to return the mean, stdev from normlization
    if return_stats:
        return train_data_loader, test_data_loader, stats        
    else:
        return train_data_loader, test_data_loader       

if __name__ == "__main__":
    # Set of params
    dat_dir = '/users/mtoomey/scratch/shirley_project/data/vortex_axion_data'
    batch_size = 64
    train_test_split = 0.80

    train_ldr, test_ldr = get_dataloaders(dat_dir,batch_size,train_test_split)
