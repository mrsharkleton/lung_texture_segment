import nibabel as nib
import numpy as np
import os
import multiprocessing as mp

def worker(label_path, result):
    print(f"Processing {label_path}")
    tmp = []
    tmp.append(label_path)

    label = nib.load(label_path)
    aff = label.affine

    x_dim = np.abs(aff[0][0])
    y_dim = np.abs(aff[1][1])
    z_dim = np.abs(aff[2][2])
    tmp.append(x_dim)
    tmp.append(y_dim)
    tmp.append(z_dim)

    pix_size = x_dim*y_dim*z_dim #(mm^3)
    tmp.append(pix_size)

    label_arr = label.get_fdata()
    count_lung = np.count_nonzero(label_arr)
    tmp.append(count_lung)
    for i in range(1,11):
        tmp.append(len(np.where(label_arr.flatten()==i)[0]))
    
    result.append(tmp)

def main(input_dir, csv_name):

    label_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                        if f.endswith("_label.nii.gz")])
    result = []    

    manager = mp.Manager()
    result_list = manager.list()

    p = mp.Pool(32)
    r = p.starmap(worker, zip(label_files, [result_list]*len(label_files)))
    p.close()
    p.join()


    result = list(result_list)
    with open(csv_name, 'w') as f:
        for line in result:
            r_new = [str(l) for l in line]
            f.write(','.join(r_new)+'\n')


if __name__ == "__main__":

    main("/media/experiments/lazy_seg/data/train/", "training_cases1.csv")
