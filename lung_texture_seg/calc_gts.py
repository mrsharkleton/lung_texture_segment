import json
import pickle
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import metrics
from training import calculate_roc_metrics


def calc(gt_dir, inference_basedir, output_dir, pickle_name):

    # get the ground truths
    ct_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) 
                        if f.endswith("_image.nii.gz") and os.path.exists(os.path.join(gt_dir,f.replace("_image.nii.gz", "_label.nii.gz")))])
    
    label_files = [f.replace("_image.nii.gz", "_label.nii.gz") for f in ct_files]

    labels_to_change = [(4,3), (6,4), (7,5), (10,6)]

    kernels = [
        #3d
        [64,64,64],
        [48,48,48],
        [32,32,32],
        [24,24,24],
        #3d
        [64,64,8],
        [48,48,8],
        [32,32,8],
        [24,24,8],
        #2d
        [64,64,1],
        [48,48,1],
        [32,32,1],
        [24,24,1],
        # 2.5D
        [64,64,64],
        [48,48,48],
        [32,32,32],
        [24,24,24],
    ]
    dims = [
        "3",
        "3",        
        "3",
        "3",

        "3",
        "3",
        "3",
        "3",

        "3",
        "3",
        "3",
        "3",

        "2.5",
        "2.5",
        "2.5",
        "2.5",]

    texture_names=["Failed to Classify", "Normal", "Pure-GG", "GG Reticulation", "Honeycombing", "Emphysema"]

    results_dict = {}
    for kernel, dim in zip(kernels, dims):
        name = "_".join([str(k) for k in kernel])+ "_" + dim +"_kernel"    
        results_dict[name] = {
            "gt_arr" : np.empty(0),
            "pred_arr" : np.empty(0),
            "dice" : {},
            "accuracy" : {},
            "jaccard" : {}

        }

    for ct, label in zip(ct_files, label_files):

        ct_nii = nib.load(ct)
        ct_arr = ct_nii.get_fdata()
        ct_name = os.path.basename(ct)

        lab_nii = nib.load(label)
        lab_arr = lab_nii.get_fdata()

        # lets get our lab_arr into the same 0-5 as our inferences: NB this still isn't quite right yet as we have "Lung", "normal" as label 1, whereas others start "normal" ...
        for (orig, new) in labels_to_change:
            lab_arr[lab_arr==orig] = new
        
        slices_to_check = np.unique(np.where(lab_arr>1)[2])

        # now subtract 1 from the lab_arr to get rid of the lung class
        lab_arr = lab_arr-1
        #lazy fix
        lab_arr[lab_arr<0] =0 # get rid of pesky -1s!

        for kernel, dim in zip(kernels, dims):
            name = "_".join([str(k) for k in kernel])+ "_" + dim +"_kernel"

            dir=os.path.join(inference_basedir, name)
            inference_name = ct_name.replace("image.nii.gz", "inference.nii.gz")
            inference_path = os.path.join(dir, inference_name)
            inf_nii = nib.load(inference_path)
            inf_arr = inf_nii.get_fdata()
            im_name = os.path.join(output_dir, os.path.basename(ct).replace("image.nii.gz", name + '.png'))
            plt_title = f'{os.path.basename(ct).replace("image.nii.gz", "")} {name}'
            gt_arr, pred_arr = create_composite(ct_arr, lab_arr, inf_arr, slices_to_check, im_name, plt_title)
            results_dict[name]['gt_arr'] = np.concatenate((gt_arr, results_dict[name]['gt_arr'] ))
            results_dict[name]['pred_arr'] = np.concatenate((pred_arr, results_dict[name]['pred_arr'] ))
            dice = []
            accuracy = []
            jaccard = []

            eye = np.eye(len(texture_names))  # one extra column now because we have some unlabelled pixels in our prediction towards the edge of the lungs
            gt_arr_onehot = eye[gt_arr.astype(int)]
            pred_arr_onehot = eye[pred_arr.astype(int)]
            for i in range(len(texture_names)):
                # fpr[i], tpr[i], thresholds = roc_curve(y_true[:, i], y_pred[:, i])
                # roc_auc[i] = auc(fpr[i], tpr[i])
                # thresh[i] = thresholds[np.argmax(tpr[i] - fpr[i])]
                cm = metrics.ConfusionMatrix(gt_arr_onehot[:, i], pred_arr_onehot[:, i])
                dice.append(metrics.dice(confusion_matrix=cm))
                accuracy.append(metrics.accuracy(confusion_matrix=cm))
                jaccard.append(metrics.jaccard(confusion_matrix=cm))



            results_dict[name]['dice'][os.path.basename(ct).replace("image.nii.gz", "")] = dice
            results_dict[name]['accuracy'][os.path.basename(ct).replace("image.nii.gz", "")] = accuracy
            results_dict[name]['jaccard'][os.path.basename(ct).replace("image.nii.gz", "")] = jaccard
        
    r = calc_aucs(results_dict, kernels, dims, texture_names, output_dir)

    for key in r:
        results_dict[key]['summary'] = r[key]
        print (f'{key} (Dice): {results_dict[key]["summary"]["dice"]} ')
        print (f'{key} (Accuracy): {results_dict[key]["summary"]["accuracy"]} ')

    # now save the result?? hopefully it won;t be too big
    with open(pickle_name, 'wb') as fp:
        pickle.dump(results_dict, fp)


def calc_aucs(big_dict, kernels, dims, texture_names, output_path):
    ### to do this I need 2 massive arrays in one_hot format. With one row per pixel in the lung region for each slice
    results = {}

    for kernel, dim in zip(kernels, dims):

        name = "_".join([str(k) for k in kernel])+ "_" + dim +"_kernel"    
        image_save_name = os.path.join(output_path, f'ROCAUC_{name}.jpg')

        eye = np.eye(len(texture_names[1:])) ## nb modified to allow me to ignore the "Not classified pixels"

        gt_arr = big_dict[name]['gt_arr']
        gt_arr = gt_arr -1 # only for this case when we have the "Unclassified" pixels we want to ignore
        gt_arr_onehot = eye[gt_arr.astype(int)]


        pred_arr = big_dict[name]['pred_arr']
        pred_arr = pred_arr -1  # only for this case when we have the "Unclassified" pixels we want to ignore
        pred_arr_onehot = eye[pred_arr.astype(int)]

        r = calculate_roc_metrics(
            gt_arr_onehot, 
            pred_arr_onehot, 
            len(texture_names[1:]), 
            texture_names[1:],
            image_save_name,
            image_save_name.replace("ROCAUC_", "CM_")) 
        
        results[name] = r

    return results    

def create_composite(image, gt, inference, slice_nums, save_name, plt_title):
    '''
        Create a composite image of our inference compared to the gt and the original image
    '''
    fig, axes = plt.subplots(len(slice_nums),6) # create a n x 5 figure
    fig.set_figheight(2*len(slice_nums))
    fig.set_figwidth(2*5)

    cmap_im = "gray"
    # cmap_lab = "viridis"
    colors =[(0,0,0,0),(1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1),(0,1,1,1)] 
    cmap_lab = ListedColormap(colors, name="label_display", N=6)
    fsize='6'

    pred_array = np.empty(0)
    gt_array = np.empty(0)

    for i, slice_num in enumerate(sorted(slice_nums, reverse=True)):
        # i = i*len(slice_nums)
        # just image
        axes[i][0].imshow(rot_im(image[:,:,slice_num]), cmap=cmap_im, vmin=-1042, vmax=300, interpolation="bilinear")
        #image + gt
        axes[i][1].imshow(rot_im(image[:,:,slice_num]), cmap=cmap_im, vmin=-1042, vmax=300, interpolation="bilinear")
        axes[i][1].imshow(rot_im(gt[:,:,slice_num]), cmap=cmap_lab, vmin=0, vmax=5, interpolation="nearest", alpha=0.5)
        #just gt
        axes[i][2].imshow(rot_im(gt[:,:,slice_num]), cmap=cmap_lab, vmin=0, vmax=5, interpolation="nearest")
        # image + inf
        axes[i][3].imshow(rot_im(image[:,:,slice_num]), cmap=cmap_im, vmin=-1042, vmax=300, interpolation="bilinear")
        axes[i][3].imshow(rot_im(inference[:,:,slice_num]), cmap=cmap_lab, vmin=0, vmax=5, interpolation="nearest", alpha=0.5)
        #just inf    
        axes[i][4].imshow(rot_im(inference[:,:,slice_num]), cmap=cmap_lab, vmin=0, vmax=5, interpolation="nearest")
        emph_mask = np.copy(image[:,:,slice_num])

        # Post-process our emphysema mask, purely based on the HU values within the original image
        emph_mask[emph_mask>-950] = 0
        emph_mask[emph_mask<=-950] = 1 

        emph_mask *= gt[:,:,slice_num]
        emph_mask[emph_mask>0] = 5  # texture_names=["Normal", "Pure-GG", "GG Reticulation", "Honeycombing", "Emphysema"] 
        axes[i][5].imshow(rot_im(emph_mask), cmap=cmap_lab, vmin=0, vmax=5, interpolation="nearest")
        if i==0:
            axes[i][0].set_title("CT", fontsize=fsize)
            axes[i][1].set_title("CT + GT\nFusion", fontsize=fsize)
            axes[i][2].set_title("GT", fontsize=fsize)
            axes[i][3].set_title("CT + Prediciton\nFusion", fontsize=fsize)
            axes[i][4].set_title("Prediction", fontsize=fsize)
            axes[i][5].set_title("Low Attenuation < 950HU", fontsize=fsize)
        
        # now let's get our gt and pred as a list
        indices = np.where(gt[:,:,slice_num] > 0)
        gt_array = np.concatenate((gt[:,:,slice_num][indices], gt_array))
        pred_array = np.concatenate((inference[:,:,slice_num][indices], pred_array))

    for axi in axes.ravel():
        # axi.set_axis_off()
        axi.set_xticklabels([])
        axi.set_yticklabels([])
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_aspect('equal')

    fig.set_tight_layout('tight')
    fig.suptitle(plt_title)
    fig.subplots_adjust(wspace=0, hspace=0.05)
    # plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.cla()
    plt.close()

    # our prediction array will contain some 0's where there is "step" artefact at the edges of the lung
    return gt_array, pred_array

def rot_im(arr):
    return np.rot90(np.rot90(np.rot90(np.flipud(arr))))
