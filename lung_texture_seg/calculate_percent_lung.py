import nibabel as nib
import numpy as np
import os

def main(input_dir, reference_dir, csv_name):

    inference_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                        if f.endswith("_inference.nii.gz")])

    reference_files = sorted([os.path.join(reference_dir, f) for f in os.listdir(reference_dir) 
                        if f.endswith("_label.nii.gz")])        

    result = []                

    for inf_path, ref_path in zip(inference_files, reference_files):
        tmp = []
        tmp.append(inf_path)
        tmp.append(ref_path)
        try:
            inf = nib.load(inf_path)
            ref = nib.load(ref_path)
            inf_arr = inf.get_fdata()
            ref_arr = ref.get_fdata()
        except Exception as ex:
            print(f"Problem with {inf_path}" )
            continue
        aff = inf.affine

        x_dim = np.abs(aff[0][0])
        y_dim = np.abs(aff[1][1])
        z_dim = np.abs(aff[2][2])

        tmp.append(x_dim)
        tmp.append(y_dim)
        tmp.append(z_dim)

        pix_size = x_dim*y_dim*z_dim #(mm^3)
        tmp.append(pix_size)


        count_lung = np.count_nonzero(ref_arr)
        tmp.append(count_lung)


        for i in range(1,6):
            tmp.append(len(np.where(inf_arr.flatten()==i)[0]))
        
        result.append(tmp)
    
    with open(csv_name, 'w') as f:
        for line in result:
            r_new = [str(l) for l in line]
            f.write(','.join(r_new)+'\n')


if __name__ == "__main__":

    main(f'/media/experiments/lazy_seg/data/562pts/final/64_64_64_2.5_kernel/', f'/media/experiments/lazy_seg/data/562pts/images/', '64x64x64x2.5_final.csv')
    