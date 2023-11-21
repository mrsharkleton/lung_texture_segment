import torch
import numpy as np
from skimage.transform import resize

class PatchDataset(torch.utils.data.Dataset):
    '''
        Specific class for our patches
    '''

    def __init__(self, data_list, transforms, train):
        self.data_list = data_list
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        resize_2d = True
        patch = self.data_list[index]['patch']
        
        # 2D case
        if len(patch.shape)==3 and patch.shape[2]==1:
            if resize_2d:
                patch = self.resize_arr(patch)
            patch = np.squeeze(patch)

        # 2.5D case, NB I messed up the dimension ordering when I saved these patches
        elif len(patch.shape)==3 and patch.shape[0]==3:
            ...# do nothing already z first
        
        # 3D case
        else:
            # we need to transform to z first
            patch = np.moveaxis(patch,2,0) 

        if self.train:
            classification = self.data_list[index]['classification'] -2
            return self.transforms(patch), classification
        
        else:
            return self.transforms(patch)
    
    def resize_arr(self, arr):
        shape = arr.shape
        if shape[0]<32 and shape[1]<32:
            # we need to resize the images to 32x32xz!
            new_shape = (32,32,shape[2])
            arr = resize(arr, new_shape)

        return arr