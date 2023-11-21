import argparse
import os
from random import Random
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import nibabel as nib
import numpy as np
import math
import multiprocessing as mp

import torch
from patch_dataset import PatchDataset
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)
from monai.utils import set_determinism
from monai.transforms.transform import Transform
from monai.config import DtypeLike, PathLike
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from monai.visualize import (
    GradCAM, 
    GradCAMpp,
    SmoothGrad,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad
)

def read_nii(args):
    '''
        Reads the image and label path and checks they are the same size etc.
        Returns the affine matrix, the ct image nd_array and the binary label mask nd_array (label >0)
    '''

    im_nii = nib.load(args.ct_path)
    im_affine = im_nii.affine
    lab_nii = nib.load(args.lung_label_path)
    lab_affine = lab_nii.affine
    if not np.equal(im_affine,lab_affine).min():
        raise RuntimeError("Image and label have different affine transforms.")
    
    im_arr = im_nii.get_fdata()
    lab_arr = lab_nii.get_fdata()
    lab_arr[lab_arr > 0] = 1    # now this is just 0 or 1

    return im_affine, im_arr, lab_arr

def calculate_slices_and_bounding_boxes(mask, label=1):
    '''
        Calculates which z-slices have lung present on them, and for each slice identified, returns bounding box coordinates for 
        the region of lung

        For 2D case this allows us to multi thread the extraction of patches from multiple slices at a time

        TODO write tests for this function
    '''
    result = []

    coords = np.where(mask==label)
    slices = np.unique(coords[2]) 

    for sli in slices:
        indexes = coords[2] == sli
        xs = coords[0][indexes]
        ys = coords[1][indexes]

        # our bounding boxes are [[x0,y0,z0],[x1,y1,z1]]

        result.append([[xs.min(), ys.min(), sli],[xs.max(), ys.max(), sli+1]])
    
    return result

def calculate_3d_bounding_box(mask, label=1):
    '''
        Calculates a 3D bounding box around all the voxels of a particular label
    '''

    coords = np.where(mask==label)

    return [[coords[0].min(), coords[1].min(), coords[2].min()],[coords[0].max(), coords[1].max(), coords[2].max()]]

def pad_array_using_stride_and_kernel(nd_array : np.ndarray, stride : list[int], kernel_size: list[int], padding_value=0):
    '''
        This function should pad a 3d [x_shape, y_shape, z_shape] array by the required size for the kernel [x_size, y_size, z_size] 
        and stride [x_stride, y_stride, z_stride]
        For a slice only case z_stride should be 1
    '''

    shape = np.array(nd_array.shape)
    stride = np.array(stride)
    kernel_size = np.array(kernel_size)

    '''
        Padding rules:
        1. if stride is > than kernel in any dimension then we do not need to pad that dimension
        2. if stride == 0 then no padding is required
        3. if stride is odd and kernel is even:
            padding = ()
        3b. if stride is odd and kernel is odd:

        4. if stride is even and kernel is even:
            padding = ((kernel - stride) / 2)
        5. if stride is even and kernel is odd:
            padding = ((kernel - stride +1)/ 2)

    '''
    padding = []
    for s, k in zip(stride,kernel_size):
        if s == 0 or s >=k:
            padding.append(int(0))
            continue
        if k%2==0: #k is even
            if s % 2 == 0: #s is even
                padding.append(int(k - s)/2)
            else: # s is odd
                padding.append(int(k - s + 1)/2)
        else:   # k is odd
            if s % 2 == 0: #s is even
                padding.append(int(k - s + 1)/2)
            else: # s is odd
                padding.append(int(k - s)/2)
        

    # pad_amount = ((kernel_size - stride) / 2).astype(int) # this is how much needs adding on all sides to  
    pad_amount = np.array(padding, int)
    if pad_amount.sum() == 0:
        # no padding is required
        return nd_array
    
    new_shape = shape +2*pad_amount
    if padding_value:
        new_slice = np.ones(new_shape)* padding_value
    else:
        new_slice = np.zeros(new_shape)
    
    '''
        I don't think there is a way around this, i think there are 7 cases we have to consider
        x=0
        x=0 & y=0
        x=0 & z=0
        x=0 & y=0 & z=0 -> covered above with early return
        y=0
        y=0 & z=0
        z=0    
    '''

    if pad_amount[0] == 0 and pad_amount[1] == 0:
        # x and y
        new_slice[:, :, pad_amount[2]:-pad_amount[2]] = nd_array

    elif pad_amount[0] == 0 and pad_amount[2] == 0:
        #x and z
        new_slice[:, pad_amount[1]:-pad_amount[1], :] = nd_array

    elif pad_amount[1] == 0 and pad_amount[2] == 0:
        #y and z
        new_slice[pad_amount[0]:-pad_amount[0], :, :] = nd_array

    elif pad_amount[0] == 0:
        # only x
        new_slice[:, pad_amount[1]:-pad_amount[1], pad_amount[2]:-pad_amount[2]] = nd_array

    elif pad_amount[1] == 0:
        # only y
        new_slice[pad_amount[0]:-pad_amount[0], :, pad_amount[2]:-pad_amount[2]] = nd_array

    elif pad_amount[2] == 0:
        #only z
        new_slice[pad_amount[0]:-pad_amount[0], pad_amount[1]:-pad_amount[1], :] = nd_array
    else:
        # none
        new_slice[pad_amount[0]:-pad_amount[0], pad_amount[1]:-pad_amount[1], pad_amount[2]:-pad_amount[2]] = nd_array

    return new_slice, padding

def unpad_slice_using_stride_and_kernel(nd_array : np.ndarray, stride : list[int], kernel_size: list[int]):
    '''
        This function should unpad a 3d [x_shape, y_shape, z_shape] array by the required size for the kernel [x_size, y_size, z_size] 
        and stride [x_stride, y_stride, z_stride]
        For a slice only case z_stride should be 1
    '''
    stride = np.array(stride)
    kernel_size = np.array(kernel_size)

    padding = []
    for s, k in zip(stride,kernel_size):
        if s == 0 or s >=k:
            padding.append(int(0))
            continue
        if k%2==0: #k is even
            if s % 2 == 0: #s is even
                padding.append(int(k - s)/2)
            else: # s is odd
                padding.append(int(k - s + 1)/2)
        else:   # k is odd
            if s % 2 == 0: #s is even
                padding.append(int(k - s + 1)/2)
            else: # s is odd
                padding.append(int(k - s)/2)

    pad_ammount = np.array(padding, int)
    if pad_ammount.sum() == 0:
        # no padding removal is required
        return nd_array
    '''
        I don't think there is a way around this, i think there are 7 cases we have to consider
        x=0
        x=0 & y=0
        x=0 & z=0
        x=0 & y=0 & z=0 -> covered above with early return
        y=0
        y=0 & z=0
        z=0    
    '''

    if pad_ammount[0] == 0 and pad_ammount[1] == 0:
        # x and y
        new_slice = nd_array[:, :, pad_ammount[2]:-pad_ammount[2]]

    elif pad_ammount[0] == 0 and pad_ammount[2] == 0:
        #x and z
        new_slice = nd_array[:, pad_ammount[1]:-pad_ammount[1], :]

    elif pad_ammount[1] == 0 and pad_ammount[2] == 0:
        #y and z
        new_slice = nd_array[pad_ammount[0]:-pad_ammount[0], :, :]

    elif pad_ammount[0] == 0:
        # only x
        new_slice = nd_array[:, pad_ammount[1]:-pad_ammount[1], pad_ammount[2]:-pad_ammount[2]]

    elif pad_ammount[1] == 0:
        # only y
        new_slice = nd_array[pad_ammount[0]:-pad_ammount[0], :, pad_ammount[2]:-pad_ammount[2]]

    elif pad_ammount[2] == 0:
        #only z
        new_slice = nd_array[pad_ammount[0]:-pad_ammount[0], pad_ammount[1]:-pad_ammount[1], :]
    else:
        # none
        new_slice = nd_array[pad_ammount[0]:-pad_ammount[0], pad_ammount[1]:-pad_ammount[1], pad_ammount[2]:-pad_ammount[2]]

    return new_slice

def check_if_patch_needed(  x:int, y:int, z:int, 
                            kernel_size :list[int], 
                            lung_limit, mask_slice_array:np.ndarray, patch_extractor):
    patch = patch_extractor(x, y, z, kernel_size, mask_slice_array)
    sum = patch['patch'].sum()
    if sum >= lung_limit:
        return True    
    return False

def extract_patch_centered(x : int, y: int, z:int, kernel_size: list[int], array_3d:np.ndarray, display_patch:bool = False):
    '''
    Extracts a 2D or 3D patch centered on x, y, z coordinate with size x,y,z_size
    '''
    x_max, y_max, z_max = array_3d.shape

    coronal = saggital = axial = None

    for i in range(3):
        if array_3d.shape[i] == 1 and kernel_size[i] == 1:
            #coronal 2d slice with multiprocessing
            if i == 0:
                coronal = True
            if i == 1:
                saggital = True
            if i == 2:
                axial = True


    if x >= x_max or x < 0:
        if x_max == 1:
            # this is a coronal 2d slice with multi processing???
            x_loc = 0
        else:
            raise ValueError("X must be between 0 and x_max inclusive")
    else:
        x_loc = x
    
    if y >= y_max or y < 0:
        if y_max ==1:
            # saggital 2d slice
            y_loc = 0
        else:
            raise ValueError("Y must be between 0 and y_max inclusive")
    else:
        y_loc = y

    if z >= z_max or z < 0:
        if z_max == 1:
            # axial slice
            z_loc = 0
        else:
            raise ValueError("Z must be between 0 and z_max inclusive")
    else:
        z_loc = z
    
    '''
    two options :
    1. even patch, e.g len 8 at coordinate 15
    image   11  12  13  14  15  16  17  18
                            |
    patch   0   1   2   3   4   5   6   7
        here the patch will be centred on the right of centre voxel

    2. odd patch, e.g len 7 at coordinate 15
    image   11  12  13  14  15  16  17  18
                            |
    patch       0   1   2   3   4   5   6 
        here the patch will be centred on the centre voxel

    '''
    def get_limits(x, x_size, x_max):

        if x_size % 2 == 0:
            #even x-voxel
            x_s = int(x_size/2) # x_s = what to subtract from x 
            x_a = x_s       # what to add to x (this value is exclusive!)
        else:
            x_a = int((x_size-1)/2)
            x_s = x_a + 1
    
        if x - x_s <0:
            x_l = 0
            x_u = x_size    # exclusive
        elif x + x_a > x_max:
            x_l = x_max - x_size
            x_u = x_max
        else:
            x_l = x - x_s
            x_u = x + x_a
        
        return x_l, x_u

    x_l, x_u = get_limits(x_loc, kernel_size[0], x_max)
    y_l, y_u = get_limits(y_loc, kernel_size[1], y_max)
    z_l, z_u = get_limits(z_loc, kernel_size[2], z_max)

    patch = array_3d[x_l:x_u,y_l:y_u,z_l:z_u]

    if display_patch:
        im = array_3d[:,:,z_l:z_u].squeeze()
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(im, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
        rect = patches.Rectangle((y_l,x_l), (x_u-x_l), (y_u-y_l), linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)

        axes[1].imshow(patch, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
        plt.show()

    if coronal:
        x_l = x
        x_u = x + 1
    if saggital:
        y_l = y
        y_u = y + 1
    if axial:
        z_l = z
        z_u = z + 1

    return {
        'x' : x,
        'y' : y,
        'z' : z,
        'kernel_size' : kernel_size,
        'bbox' : [[x_l, y_l, z_l],[x_u, y_u, z_u]],
        'patch' : patch,
        'dims' : "3"
    }

def extract_patch_centered_2point5d(x : int, y: int, z:int, kernel_size: list[int], array_3d:np.ndarray, display_patch:bool = False):
    '''
    Extracts a 2.5D patch centered on x, y, z coordinate with size x,y,z_size
    '''
    x_max, y_max, z_max = array_3d.shape

    coronal = saggital = axial = None

    for i in range(3):
        if array_3d.shape[i] == 1 and kernel_size[i] == 1:
            #coronal 2d slice with multiprocessing
            if i == 0:
                coronal = True
            if i == 1:
                saggital = True
            if i == 2:
                axial = True


    if x >= x_max or x < 0:
        if x_max == 1:
            # this is a coronal 2d slice with multi processing???
            x_loc = 0
        else:
            raise ValueError("X must be between 0 and x_max inclusive")
    else:
        x_loc = x
    
    if y >= y_max or y < 0:
        if y_max ==1:
            # saggital 2d slice
            y_loc = 0
        else:
            raise ValueError("Y must be between 0 and y_max inclusive")
    else:
        y_loc = y

    if z >= z_max or z < 0:
        if z_max == 1:
            # axial slice
            z_loc = 0
        else:
            raise ValueError("Z must be between 0 and z_max inclusive")
    else:
        z_loc = z
    
    '''
    two options :
    1. even patch, e.g len 8 at coordinate 15
    image   11  12  13  14  15  16  17  18
                            |
    patch   0   1   2   3   4   5   6   7
        here the patch will be centred on the right of centre voxel

    2. odd patch, e.g len 7 at coordinate 15
    image   11  12  13  14  15  16  17  18
                            |
    patch       0   1   2   3   4   5   6 
        here the patch will be centred on the centre voxel

    '''
    def get_limits(x, x_size, x_max):

        if x_size % 2 == 0:
            #even x-voxel
            x_s = int(x_size/2) # x_s = what to subtract from x 
            x_a = x_s       # what to add to x (this value is exclusive!)
        else:
            x_a = int((x_size-1)/2)
            x_s = x_a + 1
    
        if x - x_s <0:
            x_l = 0
            x_u = x_size    # exclusive
        elif x + x_a > x_max:
            x_l = x_max - x_size
            x_u = x_max
        else:
            x_l = x - x_s
            x_u = x + x_a
        
        return x_l, x_u

    x_l, x_u = get_limits(x_loc, kernel_size[0], x_max)
    y_l, y_u = get_limits(y_loc, kernel_size[1], y_max)
    z_l, z_u = get_limits(z_loc, kernel_size[2], z_max)
    
    ax = array_3d[x_l:x_u,y_l:y_u,z]
    sag = array_3d[x_l:x_u,y,z_l:z_u]
    cor = array_3d[x,y_l:y_u,z_l:z_u]

    patch = np.array((ax, cor, sag))

    if display_patch:
        im = array_3d[:,:,z_l:z_u].squeeze()
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(im, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
        rect = patches.Rectangle((y_l,x_l), (x_u-x_l), (y_u-y_l), linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)

        axes[1].imshow(patch, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
        plt.show()

    if coronal:
        x_l = x
        x_u = x + 1
    if saggital:
        y_l = y
        y_u = y + 1
    if axial:
        z_l = z
        z_u = z + 1

    return {
        'x' : x,
        'y' : y,
        'z' : z,
        'kernel_size' : kernel_size,
        'bbox' : [[x_l, y_l, z_l],[x_u, y_u, z_u]],
        'patch' : patch,
        'dims' : "2.5"
    }

def extract_patches_multiprocessing(ct_array, mask_array, bbox, args, shared_list):
    patches = extract_patches(ct_array, mask_array, bbox, args)
    if patches:
        shared_list.extend(patches)

def extract_patches(ct_array, mask_array, bbox, args):
    '''
    slice_bb should be [[x,y,z],[x1,y1,z1]]
    padding should be [x_pad, y_pad, z_pad]
    '''
    # look up our args
    kernel_size = np.array(args.kernel) # [x,y,z]
    stride = np.array(args.stride)    #[x,y,z]
    fraction_lung = args.fraction_lung 
    lung_limit = fraction_lung * kernel_size[0] * kernel_size[1] * kernel_size[2]

    # pad the image slice and the label slice to make things easier?
    ct_array, padding = pad_array_using_stride_and_kernel(ct_array, stride, kernel_size, padding_value=-1024)
    mask_array, _ = pad_array_using_stride_and_kernel(mask_array, stride, kernel_size, padding_value=0)

    # lookup our calculated z slice and bounding box. NB we add the padding to each location in the 
    # bounding box as we are padding the images
    bbox = np.array(bbox) +np.array(padding)
    limits = np.array(ct_array.shape)
    
    # calculate the start and end points that we need to extract patches over
    recon_limits = get_patch_start_and_end_coordinates(bbox, stride, kernel_size, limits)

    patches = []

    for z in range(recon_limits[0][2], recon_limits[1][2], stride[2] if stride[2]>0 else 1):
        for y in range(recon_limits[0][1], recon_limits[1][1], stride[1] if stride[1]>0 else 1):
            for x in range(recon_limits[0][0], recon_limits[1][0], stride[0] if stride[0]>0 else 1):
                if check_if_patch_needed(x, y,z, kernel_size, lung_limit, mask_array):
                    patches.append(extract_patch_centered(x, y, z, kernel_size, ct_array, display_patch=False))

    return patches

def get_patch_start_and_end_coordinates(bbox, stride, kernel_size, limits):
    '''
        Calculates the start and end positions for a strided extraction of patches from an image (2d or 3d)

    '''

    if stride[0] == 0:
        x_start = bbox[0][0]
        x_end = bbox[0][0] + kernel_size[0]
    else:
        x_start = (bbox[0][0]//stride[0]) * stride[0] if bbox[0][0] > math.floor(kernel_size[0]/2) else math.floor(kernel_size[0]/2)
        x_end = ((bbox[1][0]//stride[0]) +1) * stride[0] if bbox[1][0] < limits[0] - math.floor(kernel_size[0]/2) else limits[0] - math.floor(kernel_size[0]/2)

    if stride[1] == 0:
        y_start = bbox[0][1]
        y_end = bbox[0][1] + kernel_size[1]
    else:
        y_start = (bbox[0][1]//stride[1]) * stride[1] if bbox[0][1] > math.floor(kernel_size[1]/2) else math.floor(kernel_size[1]/2)
        y_end = ((bbox[1][1]//stride[1]) +1) * stride[1] if bbox[1][1] < limits[1] - math.floor(kernel_size[1]/2) else limits[1] - math.floor(kernel_size[1]/2)

    if stride[2] == 0:
        z_start = bbox[0][2]
        z_end = bbox[0][2] + kernel_size[2]
    else:
        z_start = (bbox[0][2]//stride[2]) * stride[2] if bbox[0][2] > math.floor(kernel_size[2]/2) else math.floor(kernel_size[2]/2)
        z_end = ((bbox[1][2]//stride[2]) +1) * stride[2] if bbox[1][2] < limits[2] - math.floor(kernel_size[2]/2) else limits[2] - math.floor(kernel_size[2]/2)


    return np.array([[x_start, y_start, z_start],[x_end, y_end, z_end]], dtype=int) 

def rebuild(patches, image_shape, affine, args, orig_image=None):

    new_image = np.zeros(image_shape, dtype=int)

    stride = args.stride
    new_image, padding = pad_array_using_stride_and_kernel(new_image, stride, args.kernel)
    for i in range(3):
        if stride[i] == 0:
            stride[i] = 1

    for patch in patches:
        insert_patch(patch, stride, new_image, display_patch=False, orig_image=orig_image)

    new_image = unpad_slice_using_stride_and_kernel(new_image, args.stride, args.kernel)
    nii_im = nib.Nifti1Image(new_image, affine)
    nib.save(nii_im, args.output_path)


def insert_patch(patch, stride, array_3d:np.ndarray, display_patch:bool=False, orig_image=None):
    # this is going to be the opposite of extract_patch_centered!

    x, y, z = patch['x'], patch['y'], patch['z']
    x_max, y_max, z_max = array_3d.shape

    def get_limits(x, x_size, x_max):
        if x_size % 2 == 0:
            #even x-voxel
            x_s = int(x_size/2) # x_s = what to subtract from x 
            x_a = x_s       # what to add to x (this value is exclusive!)
        else:
            x_a = int((x_size-1)/2)
            x_s = x_a + 1
    
        if x - x_s <0:
            x_l = 0
            x_u = x_size    # exclusive
        elif x + x_a > x_max:
            x_l = x_max - x_size
            x_u = x_max
        else:
            x_l = x - x_s
            x_u = x + x_a
        
        return x_l, x_u

    x_l, x_u = get_limits(x, stride[0], x_max)
    y_l, y_u = get_limits(y, stride[1], y_max)
    z_l, z_u = get_limits(z, stride[2], z_max)

    array_3d[x_l:x_u,y_l:y_u,z_l:z_u] = patch['inference'] +1 # because our classifier does not need to know "NOT LUNG CLASS"


    if display_patch:
        im = orig_image[:,:,z_l:z_u].squeeze()
        fig, axes = plt.subplots(1,3)
        axes[0].imshow(im, cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
        inference_rect = patches.Rectangle((y_l,x_l), (x_u-x_l), (y_u-y_l), linewidth=1, edgecolor='b', facecolor='none')
        axes[0].add_patch(inference_rect)
        bbox = patch['bbox']
        patch_rect = patches.Rectangle((bbox[0][1],bbox[0][0]),(bbox[1][0]-bbox[0][0]),(bbox[1][1]-bbox[0][1]), linewidth=1, edgecolor='r', facecolor='none')      
        axes[0].add_patch(patch_rect)
        axes[1].imshow(patch['patch'], cmap="gray", vmin=-1024, vmax=300, interpolation='bilinear')
        axes[2].imshow(patch['cam'].squeeze(), cmap="jet", interpolation='bilinear')
        plt.show()
        ...

def try_rebuild(args):
    with open("data//test_patches_inferenced.pickle", "rb") as f:
        patches = pickle.load(f)   
    image_shape = (512, 512, 493)
    affine = np.eye(4)
    image = rebuild(patches, image_shape, affine, args)

def try_inference(args, explainability=False):
    with open("data//test_patches.pickle", "rb") as f:
        patches = pickle.load(f)
        rand = Random()
        rand.shuffle(patches)
        patches = patches[:10]

    patches = inference(args, patches, explainability)

    return patches
    
def inference(args, patches, explainability=False):

    model_path = os.path.join(args.model_dir, "_".join(str(i) for i in args.kernel) + "_kernel.pth")
    test_batch_size=120000
    texture_names=["Normal", "Artefact", "Pure-GG", "GG Reticulation", "Honeycombing", "Emphysema"]
    expected_labels = [2,3,4,6,7,10]
    labels_to_change=[(6,5), (7,6), (10,7)]

    if labels_to_change:
        for (orig, new) in labels_to_change:
            for i, e in enumerate(expected_labels):
                if e == orig:
                    expected_labels[i]=new
    
    num_class = len(expected_labels)

    val_transforms = Compose(
        [ AddChannel(), ScaleIntensity(), EnsureType()])
    test_ds = PatchDataset(patches, val_transforms, train=False)
    test_loader = torch.utils.data.DataLoader( test_ds, batch_size=test_batch_size, shuffle=False, num_workers=args.processors)

    # define network and optimiser
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1,
                        out_channels=num_class, pretrained=False).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError("model path does not exist")
    
    cam = GradCAMpp(nn_module=model, target_layers="class_layers.relu")
    smooth_grad = SmoothGrad(model, verbose=False)
    guided_vanilla = GuidedBackpropGrad(model)
    guided_smooth = GuidedBackpropSmoothGrad(model, verbose=False)

    # now do the inferences
    model.eval()
    y_pred = []
    y_probabilities = []
    y_probability = []
    y_pred_logits = []
    y_gradcams = []
    y_smoothed_grad = []
    y_guided_vanilla = []
    y_guided_smooth = []
    y_class = []
    with torch.no_grad():   
        for data in test_loader:
            patch = data.to(device)
            pred_logits = model(patch)
            pred_label = pred_logits.argmax(dim=1)
            pred_prob = torch.nn.functional.softmax(pred_logits, dim=1)

            if explainability:
                grad_cam = cam(x=patch, class_idx=pred_label)
                smooth_cam = smooth_grad(x=patch)
                vanilla_cam = guided_vanilla(x=patch)
                guided_cam = guided_smooth(x=patch)

            for i in range(len(pred_label)):
                y_pred.append(pred_label[i].item())
                y_probabilities.append(pred_prob[i].detach().cpu().numpy())
                y_probability.append(pred_prob[i, pred_label[i]].item()*100)
                y_pred_logits.append(pred_logits[i].detach().cpu().numpy())
                y_class.append(texture_names[pred_label[i].item()]) 
                if explainability:
                    y_gradcams.append(grad_cam[i].detach().cpu().numpy())
                    y_smoothed_grad.append(smooth_cam[i].detach().cpu().numpy())
                    y_guided_vanilla.append(vanilla_cam[i].detach().cpu().numpy())
                    y_guided_smooth.append(guided_cam[i].detach().cpu().numpy())  
        
    
    for i in range(len(patches)):
        patches[i]['inference'] = y_pred[i]
        patches[i]['pred_logits_weight'] = y_pred_logits[i]
        patches[i]['pred_logits_prob'] = y_probabilities[i]
        patches[i]['probability'] = y_probability[i]
        patches[i]['inference_class'] = y_class[i]
        if explainability:        
            patches[i]['gradcampp'] = y_gradcams[i]
            patches[i]['smoothcam'] = y_smoothed_grad[i]
            patches[i]['guidedvacam'] = y_guided_vanilla[i]
            patches[i]['guidedsmcam'] = y_guided_smooth[i]
    
    return patches

def recon(args):
    # 1. Read the nii files 
    affine, im_arr, lab_arr = read_nii(args)

    # 2. Work out which slices actually require labelling this is optimised for 2d case currently
    bounding_boxes = calculate_slices_and_bounding_boxes(lab_arr)

    mask_slices = []
    image_slices = []

    for bb in bounding_boxes:
        mask_slices.append(lab_arr[:,:,bb[0][2]:bb[1][2]])
        image_slices.append(im_arr[:,:,bb[0][2]:bb[1][2]])

    # 3. Now we might be able to use some multi-processing to extract all of the patches we need to classify
    manager = mp.Manager()
    shared_list = manager.list()
    p = mp.Pool(args.processors)
    r = p.starmap_async(
        extract_patches_multiprocessing,
        zip(
            image_slices,
            mask_slices,
            bounding_boxes,
            [args] * len(bounding_boxes),
            [shared_list ] * len(bounding_boxes)
        ))
    r.get()
    p.close()
    p.join()

    patches = list(shared_list)

    # 4. Now we need to run an inference on each of the patches within our image
    patches = inference(args, patches)

    # 5. Now we should reconstruct the image from the patch classifications
    rebuild(patches, im_arr.shape, affine, args, orig_image=im_arr)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=    "recon_image.py: \n \
                        author: sharkeymj \n \
                        data: July 2022 \n \
                        This reconstructs a coarse semantic labelling of a thoracic lung scan showing areas of disease.")
    parser.add_argument(
        "--ct_path", "-c", help="The path to a thoracic CT scan to label", type=str, required=True
    )
    parser.add_argument(
        "--lung_label_path", "-l", help="The path to the lung label associated with the thoracic CT scan", type=str, required=True
    )
    parser.add_argument(
        "--kernel", "-k", help="The kernel size in pixels", nargs="+", type=int, default=[32,32,1]
    )
    parser.add_argument(
        "--model_dir", "-m", help="Directory containing models for the different kernels", type=str, default="models"
    )
    parser.add_argument(
        "--output_path", "-o", help="Path to save reconstruction to"
    )
    parser.add_argument(
        "--stride", help="The stride in pixels between kernels. Must be <=k. k=no overlap (fully strided)", nargs="+", type=int, default=[16,16,0]
    )
    parser.add_argument(
        "--fraction_lung", "-f", help="The fraction of a patch which must be within the lung before it is classified", default=0.5, type=float
    )
    parser.add_argument(
        "--processors", "-p", help="The number of processors to use for multiprocessing and extracting patches and reconstructing the image", default=8, type=int
    )
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    recon(args)
