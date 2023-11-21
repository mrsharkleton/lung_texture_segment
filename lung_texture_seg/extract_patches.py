'''
    Our data consists of images and labels
    The label is such that it is:
    0: not lung
    >1: Lung
    2: texture A
    3: texture B
    4: texture C
    etc

    For preliminary training we are only interested in patches in the axial plane.
    
    PATCHES:
    Patches will be 2D and of a variable (but fixed!!) size for each experiment e.g. 32x32 pixels. 
    Patches will have a minimum texture fill to be considered a particular class e.g. 0.75

    SLICE PREPROCESSING:
    Pre processing will involve going through the volume to work out which slices have a texture present and the total number of pixels.
    Following this slices with the minimum fill ratio for the patch size (e.g > 32x32*0.75 pixels of texture A present) can be selected using a random number generator.
    For each slice attempts will be made to find patches meeting the criteria above


    ASSUMPTIONS:
    1. The voxel dimensions are similar between images such that we don't have to resample for our patches


    EACH_PATCH_NEEDS:
    patch= {
        image : nd_array,
        class : int,
        image_name : original_file_path
        label_name : label_file_path
        location : {
            start : [x0, y0],
            end : [x1, x2]
        }
        size : patch_size
    }
'''

import argparse
from datetime import datetime
import os
import pickle
import numpy as np
import nibabel as nib
import json
import random
import socket
import math
from recon_image import extract_patch_centered, extract_patch_centered_2point5d


def preprocess_label(label: np.ndarray, label_name: str, image_name: str, labels: list) -> dict:
    '''
    This preprocesses the label to calculate which slices have a particular label and how much on each slice.
    It also calculates the total number of pixels present with each texture 
    returns = dict {
        'label_name' : 'label_name',
        'image_name' : 'image_name',
        'labels_present': [list of labels present],
        1 : {
            'slices' : [slice#, voxels],
            'total_vol' : vol,
            'num_slices' : #slices
            'coords' : [[xs],[ys],[zs]]
        },
        2 : {
            'slices' : [slice#, voxels],
            'total_vol' : vol,
            'num_slices' : #slices,
            'coords' : [[xs],[ys],[zs]]
        }, etc.
    }
    '''
    preprocess_dic = {}
    preprocess_dic['label_name'] = label_name
    preprocess_dic['image_name'] = image_name
    labels_present = []

    min_l = label.min()
    max_l = label.max()

    for l in labels:
        if l < min_l or l > max_l:
            continue
        coords = np.where(label==l) 
        if len(coords[0]) == 0:
            continue
        labels_present.append(l)

        slices, counts = np.unique(coords[2], return_counts=True)   ## need to check I'm getting the x values!
        total_vol = np.sum(counts)
        slice_vox_arr = np.concatenate((slices.reshape(-1,1), counts.reshape(-1,1)), axis=1)    # create a 2d array of [slice, voxels]
        preprocess_dic[l] = {
            'slices' : slice_vox_arr,
            'total_vol' : total_vol,
            'num_slices' : len(slices),
            'coords' : list(coords)
        }
    
    preprocess_dic['labels_present'] = labels_present

    return preprocess_dic


def process(image_list: list, label_list: list, output_dir, patch_size, fill_fraction, expected_labels=[2,3,4,5], preprocess_path=None, num_patches=10000, save_summary=False, dry_run=False, num_dims = "3", space_between_patches_mm=5):
    # lets say we have 10 images and 10 labels to begin with 
    os.makedirs(output_dir, exist_ok=True)

    # 1, Get the preprocessed label_info
    label_info = {}
    label_info['images'] = [[im, label] for im, label in zip(image_list, label_list)]

    if preprocess_path and os.path.exists(preprocess_path):
        with open(preprocess_path, 'rb') as f:
            label_info = pickle.load(f)
    else:
        for i in range(len(image_list)):
            im, label = image_list[i], label_list[i]
            nii_label = nib.load(label)
            nii_arr = nii_label.get_fdata()

            label_info[im] = preprocess_label(nii_arr, label, im, expected_labels)
        if preprocess_path:
            with open(preprocess_path, 'wb') as f:
                pickle.dump(label_info, f, protocol=pickle.HIGHEST_PROTOCOL)        
        else:
            with open(os.path.join(output_dir, 'preprocessed_label_info.pickle'), 'wb') as f:
                pickle.dump(label_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    threshold = fill_fraction*patch_size[0] * patch_size[1]
    #2. Now print summary of what we have
    if save_summary:
        now = datetime.now().strftime('%Y%m%d_%H%M')
        save_name = os.path.join(output_dir, f'label_summary_{now}.json')
    else:
        save_name = None
    summary = print_summary(label_info, threshold, save_name=save_name)
    
    #3. Now we need to try and select a balanced set of patches
    actual_labels = summary['labels_present']
    expected_labels_copy = expected_labels.copy()
    for label in expected_labels:
        if label not in actual_labels:
            print(f'{label} not present, removing from expected_labels')
            expected_labels_copy.remove(label)
    expected_labels = expected_labels_copy
    # if actual_labels != expected_labels:
    #     print(f'Not all expected labels present, preceeding with {actual_labels}')
    
    if dry_run:
        print("Dry run, exiting without generating patches...")
        return
    # for now store my patches in a dictionary, this will be a bad idea long term as probably inefficient if gets large
    patch_dict = {}

    for label in expected_labels:
        total_slices = summary[label]['slices_above_threshold']
        if total_slices ==0:
            print(f'{label} not present, removing from expected_labels')
            expected_labels.remove(label)
        else:
            patches_per_slice = num_patches/total_slices
        patch_dict[label] = {
            'total_slices_above_threshold' : total_slices,
            'patches_per_slice': patches_per_slice,
            'num_patches' : 0,
            'patch_list' : [],
            'images_to_choose' : summary[label]['images']
        }     


    # rethink this -> i want to load the images as few times as possible
    

    def check_enough_patches(patch_dict, num):
        for key in patch_dict:
            if patch_dict[key]['num_patches'] < num:
                return False
        return True

    total_patches_extracted = num_patches
    
    while not check_enough_patches(patch_dict, num_patches):
        if len(expected_labels) == 0:
            break
        if len(expected_labels) ==1:
            # we can ditch all the other images and just use the ones for this label
            image_list = [l[0] for l in patch_dict[expected_labels[0]]['images_to_choose']]
            if len(image_list) == 0:
                break
        # 1. select a random image
        r = random.randint(0, len(image_list)-1)
        im_name = image_list[r]
        lab_name = im_name.replace('image.nii.gz', 'label.nii.gz') #TODO this is risky as no check that the label matches the image

        im = None
        lab = None

        im_arr = None
        lab_arr = None
        affine = None
        voxel_size = None

        for label_idx, label in enumerate(expected_labels):
            # check there are still some of this lable left to extract
            if len(patch_dict[label]['images_to_choose']) == 0:
                # we've run out of steam here and can;t extract any more of this texture
                print(f"run out of patches for {label}")
                if total_patches_extracted > patch_dict[label]['num_patches']:
                    total_patches_extracted = patch_dict[label]['num_patches']
                del expected_labels[label_idx]
                continue
            # 1. check if I have sufficient labels
            if patch_dict[label]['num_patches'] > num_patches:
                # we can remove this from expected labels
                del expected_labels[label_idx]
                continue
            #2. check if this image has any of this label
            if im_name not in [n[0] for n in patch_dict[label]['images_to_choose']]:
                continue
            
            # Load the image and label if required
            if im == None:
                im = nib.load(im_name)
                lab = nib.load(lab_name)

                im_arr = im.get_fdata()
                lab_arr = lab.get_fdata()
                affine = im.affine
                voxel_size = np.array([abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2])])

            pps = patch_dict[label]['patches_per_slice']

            #3. Now we are ready to select some patches from this image
            patches = get_patch(lab_arr, im_arr, label, patch_size, threshold, im_name, lab_name, label_info, pps, voxel_size=voxel_size, space_between_patches_mm=space_between_patches_mm, attempts=1000, patch_dict=patch_dict, num_dims=num_dims)
            new_patches = len(patches)
            patch_dict[label]['patch_list'].extend(patches)
            patch_dict[label]['num_patches'] += new_patches
            print(f'Found {str(new_patches)} patches for label {label} for {im_name} !')
        
    # now save out our patches.
    for key in patch_dict:
        with open(os.path.join(output_dir, f'patches_{num_dims}D_{str(total_patches_extracted)}_{str(patch_size[0])}x{str(patch_size[1])}x{str(patch_size[2])}_{str(fill_fraction)}_{space_between_patches_mm}mm_{str(key)}.pickle'), 'wb') as f:
            pickle.dump(patch_dict[key], f, protocol=pickle.HIGHEST_PROTOCOL)

def get_patch(label:np.ndarray, image:np.ndarray, label_num:int, patch_size: list, threshold, im_name, lab_name, label_info, patches_per_slice, voxel_size=np.array([1,1,1]), space_between_patches_mm=10, attempts = 20, patch_dict=None, num_dims="3" ) -> None:

    # 1. lets get the info for this image
    info = label_info[im_name][label_num]
    patch_list = []
    def exclude_below_threshold(slice_list, threshold):
        new_list = []
        remove_list = []
        for l in slice_list:
            if l[1] > threshold:
                new_list.append(l)
            else:
                remove_list.append(l)
        return remove_list, new_list
    
    #2. lets get the slices that might be useful
    slices_to_remove, slice_list = exclude_below_threshold(info['slices'], threshold)

    def remove_indices_in_slice_list(coordinates, slice_list):
        # exes = coordinates[0]
        # whys = coordinates[1]
        # zeds = coordinates[2]
        # indices = np.where(np.isin(zeds, slice_list))[0]    # array of indices we will remove from all coordinates

        # exes = np.delete(exes, indices)
        # whys = np.delete(whys, indices)
        # zeds = np.delete(zeds, indices)

        # return [exes, whys, zeds]
        indices = np.where(np.isin(coordinates[2], slice_list))[0]    # array of indices we will remove from all coordinates        
        remove_indices(coordinates, indices)

    
    def remove_indices(coordinates, indices):
        
        for i in range(3):
            coordinates[i] = np.delete(coordinates[i], indices)   

    def remove_close_indices(coords, x,y,z,voxel_size_mm, limit_mm):

        # this needs to be vectorised!

        distance = (  ((coords[0] - x) * voxel_size_mm[0]) ** 2 
                    + ((coords[1] - y) * voxel_size_mm[1]) ** 2 
                    + ((coords[2] - z) * voxel_size_mm[2]) ** 2 ) ** 0.5  # should all be +ve numbers now

        indices = np.where(distance<limit_mm)[0]

        remove_indices(coords, indices)

    # print(f'Coordinate length initially = {len(info["coords"][0])}')
    remove_indices_in_slice_list(info['coords'], slices_to_remove) 
    # print(f'Coordinate length final = {len(info["coords"][0])}')
    #3. lets calculate how many patches we should aim for from this image
    num_expected_patches = int(math.ceil(patches_per_slice * len(slice_list)))
    num_patches = 0
    trys = num_expected_patches * attempts

    # sort out our patch extractor
    if num_dims=="3" or num_dims=="2":
        patch_extractor = extract_patch_centered
    elif num_dims =="2.5":
        patch_extractor = extract_patch_centered_2point5d
    else:
        raise ValueError("{num_dims} not a valid number, must be 2/2.5/3")

    for i in range(trys):
        if num_patches > num_expected_patches:
            break

        # random_slice = random.randint(0, len(slice_list)-1)
        try:
            random_coord = random.randint(0, len(info['coords'][0])-1)
        except ValueError as ve:
            # this means we have run out of coordinates to try!
            label_info[im_name]['labels_present'].remove(label_num)
            idx_to_remove = None
            for j,im_deets in enumerate(patch_dict[label_num]['images_to_choose']):
                if im_deets[0] == im_name:
                    idx_to_remove = j
                    break
            del patch_dict[label_num]['images_to_choose'][idx_to_remove]

            return patch_list
            
        x = info['coords'][0][random_coord]
        y = info['coords'][1][random_coord]
        z = info['coords'][2][random_coord]

        # z = slice_list[random_slice][0]
        # s_arr = label[:,:,z]
        # x_coords, y_coords = np.where(s_arr == label_num)

        # c = random.randint(0,len(x_coords)-1)
        # lab_bb, lab_patch = extract_patch(x_coords[c], y_coords[c], z, patch_size[0], patch_size[1], label )
        label_patch = patch_extractor(x,y,z,patch_size, label, display_patch=False)

        # lab_bb, lab_patch = extract_patch(x, y, z, patch_size[0], patch_size[1], label )
        pixel_count = len(np.where(label_patch['patch'] == label_num)[0])
        if pixel_count >= threshold:
            # success we have found a patch

            # im_patch = get_patch_from_bb(lab_bb, image)
            im_patch = patch_extractor(x,y,z,patch_size, image, display_patch=False)
            num_patches +=1
            im_patch['classification'] = label_num
            im_patch['image_name'] = im_name
            im_patch['label_name'] = lab_name


            # patch_dic = {
            #     'patch' : im_patch,
            #     'classification' : label_num,
            #     'image_name' : im_name,
            #     'label_name' : lab_name,
            #     'bounding_box' : lab_bb,
            #     'size' : patch_size                
            # }
            # patch_list.append(patch_dic)
            patch_list.append(im_patch)

            remove_close_indices(info['coords'], x,y,z, voxel_size, space_between_patches_mm)

        else:
            # unset the index from our list of coordinates so we don't choose it again
            remove_indices(info['coords'], random_coord)

    return patch_list

def get_patch_from_bb(bb, array_3d):
    return array_3d[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1],bb[0][2]]

def extract_patch(x,y,z,x_size : int, y_size: int, array_3d):
    x_max = array_3d[:,:,z].shape[0]
    y_max = array_3d[:,:,z].shape[1]

    if x_size % 2 == 0:
        x_s = int(x_size/2)
        x_a = x_s
    else:
        x_a = int((x_size-1)/2)
        x_s = x_a + 1

    
    if y_size % 2 == 0:
        y_s = int(y_size/2)
        y_a = y_s
    else:
        y_a = int((y_size-1)/2)
        y_s = y_a + 1

    if x - x_s <0:
        x_l = 0
        x_u = x_size
    elif x + x_a > x_max:
        x_l = x_max - x_size
        x_u = x_max

    else:
        x_l = x - x_s
        x_u = x + x_a

    if y - y_s <0:
        y_l = 0
        y_u = y_size
    elif y + y_a > y_max:
        y_l = y_max - y_size
        y_u = y_max

    else:
        y_l = y - y_s
        y_u = y + y_a        

    return [[x_l, y_l, z],[x_u, y_u, z]], array_3d[x_l:x_u,y_l:y_u,z]

def print_summary(label_info, threshold, save_name=None):
    '''
    Prints out a summary of the total # slices, total volume, for each label across all volumes in the training data
    Also prints a summary for each case of the cases with each label, and number of slices and volume
    '''

    all_labels_present = []

    for im_name in label_info:
        if im_name in ['images']:
            continue
        case = label_info[im_name]

        im_labels_present = case['labels_present']
        for l in im_labels_present:
            for sli in case[l]['slices']:
                if sli[1] > threshold:
                    all_labels_present.append(l)
                    break
            # if l not in all_labels_present:
            #     all_labels_present.append(l)
    all_labels_present = sorted(list(set(all_labels_present)))
    summary_dic = {}
    summary_dic['images'] = label_info['images']
    summary_dic['labels_present'] = sorted(all_labels_present)
    for l in all_labels_present:
        summary_dic[l] = { 
            'images': [],
            'total_slices' : 0,
            'total_vol' : 0,
            'count_images' :0,
            'slices_above_threshold' : 0
        }
        
    for im_name in label_info:
        if im_name in ['images']:
            continue
        case = label_info[im_name]  
        im_labels_present = case['labels_present']

        for l in im_labels_present:
            if l in case:
                current_label = case[l]
                add_to_summary = False
                num_slices_above_threshold = 0
                for sli in case[l]['slices']:
                    if sli[1] > threshold:
                        summary_dic[l]['slices_above_threshold'] +=1 
                        num_slices_above_threshold +=1
                        add_to_summary = True
                if add_to_summary:
                    summary_dic[l]['images'].append([im_name, case['label_name'], int(num_slices_above_threshold), int(current_label['total_vol'])])
                    summary_dic[l]['total_slices'] += int(current_label['num_slices'])
                    summary_dic[l]['total_vol'] += int(current_label['total_vol'])
                    summary_dic[l]['count_images'] +=1    


    print(json.dumps(summary_dic, indent=4, sort_keys=False))

    if save_name:
        with open(save_name, 'w') as writer:
            writer.writelines(json.dumps(summary_dic, indent=4, sort_keys=False))

    return summary_dic

def parse_args():
    parser = argparse.ArgumentParser(description='Description of your script.')

    parser.add_argument('--base_data_dir', type=str, required=True,
                        help='Base directory for data.')
    
    parser.add_argument('--test_patches', type=int, default=50,
                        help='Number of patches for the test set.')
    
    parser.add_argument('--val_patches', type=int, default=50,
                        help='Number of patches for the validation set.')
    
    parser.add_argument('--train_patches', type=int, default=250,
                        help='Number of patches for the training set.')
    
    parser.add_argument('--patch_size', nargs='+', type=int, default=[32, 32, 1],
                        help='Patch size as a list of three integers [depth, height, width].')
    
    parser.add_argument('--fill_fraction', type=float, default=0.5,
                        help='Fill fraction for data augmentation.')
    
    parser.add_argument('--expected_labels', nargs='+', type=int, default=[2, 3, 4, 6, 7, 10],
                        help='List of expected label values.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    for folder, num_patches in zip(["test", "val", "train"], [args.test_patches, args.val_patches, args.train_patches]):
        data_dir = os.path.join(args.base_data_dir, folder)
        images = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('image.nii.gz')])
        labels = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('label.nii.gz')])
        preprocess_path = os.path.join(args.base_data_dir, "preprocessed", "preprocessed_label_info_" + folder + ".pickle")
        output_path = os.path.join(args.base_data_dir, "preprocessed", folder)
        process(
            images,
            labels,
            output_path,
            args.patch_size,
            args.fill_fraction,
            args.expected_labels,
            preprocess_path=preprocess_path,
            save_summary=True,
            dry_run=False,
            num_patches=num_patches
        )


        