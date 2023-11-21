import unittest
from argparse import Namespace
from lung_texture_seg.recon_image import *


class TestReadNii(unittest.TestCase):

    # create our command line args

    args = Namespace(
        ct_path = "Z:\\git\\lazy_texture_seg\\data\\test_images\\image.nii.gz",
        lung_label_path = "Z:\\git\\lazy_texture_seg\\data\\test_images\\label.nii.gz",
        kernel = [32, 32, 1],
        stride = [16,16,0],
        fraction_lung = 0.5,
        model_dir = "models",
        output_path = "output/test.nii.gz",
        processors = 8,
    )
    affine_matrix = np.array(
        [[  -0.65429699,   -0.        ,   -0.        ,  170.1000061 ],
        [  -0.        ,   -0.65429699,   -0.        ,  137.5       ],
        [   0.        ,    0.        ,    0.625     , -262.25      ],
        [   0.        ,    0.        ,    0.        ,    1.        ]]
    )
    image_shape = (512, 512, 493)

    def test_arguments(self):
        self.assertEqual(self.args.ct_path, "Z:\\git\\lazy_texture_seg\\data\\test_images\\image.nii.gz")

    def test_read_nii(self):
        affine, im_arr, lab_arr = read_nii(self.args)

        comparison = np.isclose(affine, self.affine_matrix)
        self.assertTrue(comparison.min())
        self.assertEqual(im_arr.shape, self.image_shape)
        self.assertEqual(lab_arr.shape, self.image_shape)
        self.assertEqual(lab_arr.max(), 1)
        self.assertEqual(lab_arr.sum(), 15645418.0)
        self.assertEqual(im_arr.sum(), -74407265867.0)

    def test_extract_patch_centered_first_patch(self):
        '''
            Tests extract_patch_centred returns the correct first patch
        '''
        k = [j+512*i for i in range(32) for j in range(32)]

        test_kernel = np.expand_dims(np.array(k).reshape(32,32), axis=2)
        test_image = np.expand_dims(np.arange(0, 512*512).reshape([512,512]), axis=2)
        test_dict = {
            'x': 16,
            'y': 16,
            'z': 0,
            'kernel_size' : [32,32,1],
            'bbox' : [[0,0,0],[32,32,1]],
            'patch' : test_kernel
        }

        patch_dict = extract_patch_centered(16,16,0,[32,32,1],test_image)
        self.assertEqual(patch_dict['kernel_size'], test_dict['kernel_size'])
        self.assertEqual(patch_dict['bbox'], test_dict['bbox'])
        np_array_match = np.equal(patch_dict['patch'], test_dict['patch']).min()
        self.assertTrue(np_array_match)


    def test_extract_patch_centered_before_first_patch(self):
        '''
            Tests extract_patch_centred returns the correct patch even if the 
            patch location is too close to the edge of the image
        '''
        k = [j+512*i for i in range(32) for j in range(32)]

        test_kernel = np.expand_dims(np.array(k).reshape(32,32), axis=2)
        test_image = np.expand_dims(np.arange(0, 512*512).reshape([512,512]), axis=2)
        test_dict = {
            'x': 16,
            'y': 16,
            'z': 0,
            'kernel_size' : [32,32,1],
            'bbox' : [[0,0,0],[32,32,1]],
            'patch' : test_kernel
        }

        patch_dict = extract_patch_centered(10,10,0,[32,32,1],test_image)
        self.assertEqual(patch_dict['kernel_size'], test_dict['kernel_size'])
        self.assertEqual(patch_dict['bbox'], test_dict['bbox'])
        np_array_match = np.equal(patch_dict['patch'], test_dict['patch']).min()
        self.assertTrue(np_array_match)

    def test_extract_patch_centered_last_patch(self):
        '''
            Tests extract_patch_centred returns the correct patch even if the 
            patch location is too close to the edge of the image
        '''
        k = [j+544*i for i in range(512, 512+32) for j in range(512, 512+32)]

        test_kernel = np.expand_dims(np.array(k).reshape(32,32), axis=2)
        test_image = np.expand_dims(np.arange(0, 544*544).reshape([544,544]), axis=2)
        test_dict = {
            'x': 528,
            'y': 528,
            'z': 0,
            'kernel_size' : [32,32,1],
            'bbox' : [[512,512,0],[544,544,1]],
            'patch' : test_kernel
        }

        patch_dict = extract_patch_centered(528,528,0,[32,32,1],test_image)
        self.assertEqual(patch_dict['kernel_size'], test_dict['kernel_size'])
        self.assertEqual(patch_dict['bbox'], test_dict['bbox'])
        np_array_match = np.equal(patch_dict['patch'], test_dict['patch']).min()
        self.assertTrue(np_array_match)

    def test_extract_patch_centered_after_last_patch(self):
        '''
            Tests extract_patch_centred returns the correct patch even if the 
            patch location is too close to the edge of the image
        '''
        k = [j+544*i for i in range(512, 512+32) for j in range(512, 512+32)]

        test_kernel = np.expand_dims(np.array(k).reshape(32,32), axis=2)
        test_image = np.expand_dims(np.arange(0, 544*544).reshape([544,544]), axis=2)
        test_dict = {
            'x': 528,
            'y': 528,
            'z': 0,
            'kernel_size' : [32,32,1],
            'bbox' : [[512,512,0],[544,544,1]],
            'patch' : test_kernel
        }

        patch_dict = extract_patch_centered(534,534,0,[32,32,1],test_image)
        self.assertEqual(patch_dict['kernel_size'], test_dict['kernel_size'])
        self.assertEqual(patch_dict['bbox'], test_dict['bbox'])
        np_array_match = np.equal(patch_dict['patch'], test_dict['patch']).min()
        self.assertTrue(np_array_match)

    def test_extract_patch_centered_raises_value_error(self):
        test_image = np.expand_dims(np.arange(0, 512*512).reshape([512,512]), axis=2)
        self.assertRaises(ValueError, extract_patch_centered, -1,0,0,[32,32,1],test_image) # x <0
        self.assertRaises(ValueError, extract_patch_centered, 512,0,0,[32,32,1],test_image) # x = x_max
        self.assertRaises(ValueError, extract_patch_centered, 0,-1,0,[32,32,1],test_image) # y <0
        self.assertRaises(ValueError, extract_patch_centered, 0,512,0,[32,32,1],test_image) # y = y_max
        self.assertRaises(ValueError, extract_patch_centered, 0,0,-1,[32,32,1],test_image) # z <0
        self.assertRaises(ValueError, extract_patch_centered, 0,0,1,[32,32,1],test_image) # z =z_max

    def test_extract_patch_centered_odd_patch(self):
        '''
            Test an odd patch
        '''
        k = [j+512*i for i in range(33) for j in range(33)]

        test_kernel = np.expand_dims(np.array(k).reshape(33,33), axis=2)
        test_image = np.expand_dims(np.arange(0, 512*512).reshape([512,512]), axis=2)
        test_dict = {
            'x': 17,
            'y': 17,
            'z': 0,
            'kernel_size' : [33,33,1],
            'bbox' : [[0,0,0],[33,33,1]],
            'patch' : test_kernel
        }

        patch_dict = extract_patch_centered(17,17,0,[33,33,1],test_image)
        self.assertEqual(patch_dict['kernel_size'], test_dict['kernel_size'])
        self.assertEqual(patch_dict['bbox'], test_dict['bbox'])
        np_array_match = np.equal(patch_dict['patch'], test_dict['patch']).min()
        self.assertTrue(np_array_match)

    def test_pad_slice_using_stride_and_kernel(self):
        '''
            Test that the function pads correctly a margin of 14 pixels

        '''
        test_slice = np.ones([100,100, 1])
        stride = [4,4,0]
        kernel_size = [32,32,1]
        result_slice, result_padding = pad_array_using_stride_and_kernel(test_slice, stride, kernel_size, padding_value=10)

        expected_shape = (128,128,1)
        expected_sum = 10000 + 63840
        expected_padding =[14,14,0]
        self.assertEqual(result_slice.shape, expected_shape)
        self.assertEqual(result_slice.sum(), expected_sum)
        self.assertEqual(result_padding, expected_padding)

    def test_unpad_using_stride_and_kernel(self):
        '''
            Test that the function correctly removes a 14 pixel margin.
        '''
        test_slice = np.ones([100,100,1])
        stride = [4,4,0]
        kernel_size = [32,32,1]
        result_slice = unpad_slice_using_stride_and_kernel(test_slice, stride, kernel_size)
        expected_shape = (72,72,1)
        expected_sum = 10000 - 4816
        self.assertEqual(result_slice.shape, expected_shape)
        self.assertEqual(result_slice.sum(), expected_sum)

    #TODO more tests for unpad_using_stride_and_kernel
    
    def test_check_if_patch_needed_true(self):
        '''
            This should have >=limit pixels within the desired mask and therefore
            return True
        '''
        mask = np.ones([100,100,1])
        mask[:,:50] =0

        val = check_if_patch_needed(50,50,0,[32,32,1],512, mask)
        self.assertTrue(val)

    def test_check_if_patch_needed_false(self):
        '''
            This should have <limit pixels within the desired mask and return false
        '''
        mask = np.ones([100,100,1])
        mask[:,:50] =0        
        val = check_if_patch_needed(49,49,0,[32,32,1],512, mask)
        self.assertFalse(val)

    def test_get_patch_start_and_end_coordinates_2d_normal(self):
        '''
            Check the start and end coordinates are correct for an image (after padding)
            such that the entire bbox is extracted if necessary

        '''
        
        # test normal usage
        limits = [100,100,1]
        bbox = [[10,10,0],[90,90,1]]
        kernel_size = [8,8,1]
        stride = [4,4,0]
        expected_coords = np.array([[8,8,0],[92,92,1]])
        coords = get_patch_start_and_end_coordinates(bbox, stride, kernel_size, limits)

        self.assertTrue(np.equal(expected_coords, coords).min())

    def test_get_patch_start_and_end_coordinates_2d_inbetween(self):
        '''
            Check the start and end coordinates are correct for an image (after padding)
            such that the entire bbox is extracted if necessary

        '''        
        # test between the kernel size and the limits
        limits = [100,100,1]
        bbox = [[2,2,0],[98,98,1]]
        kernel_size = [8,8,1]
        stride = [4,4,0]
        expected_coords = np.array([[4,4,0],[96,96,1]])
        coords = get_patch_start_and_end_coordinates(bbox, stride, kernel_size, limits)

        self.assertTrue(np.equal(expected_coords, coords).min())

    def test_get_patch_start_and_end_coordinates_2d_limits(self):
        '''
            Check the start and end coordinates are correct for an image (after padding)
            such that the entire bbox is extracted if necessary

        '''
        # test between the kernel size at the limits
        limits = [100,100,1]
        bbox = [[0,0,0],[100,100,1]]
        kernel_size = [8,8,1]
        stride = [4,4,0]
        expected_coords = np.array([[4,4,0],[96,96,1]])
        coords = get_patch_start_and_end_coordinates(bbox, stride, kernel_size, limits)

        self.assertTrue(np.equal(expected_coords, coords).min())

    def test_get_patch_start_and_end_coordinates_2d_asymetric(self):
        '''
            Check the start and end coordinates are correct for an image (after padding)
            such that the entire bbox is extracted if necessary

        '''
        # test between the kernel size at the limits
        limits = [100,100,1]
        bbox = [[13,27,0],[56,72,1]]
        kernel_size = [8,8,1]
        stride = [4,4,0]
        expected_coords = np.array([[12,24,0],[60,76,1]])
        coords = get_patch_start_and_end_coordinates(bbox, stride, kernel_size, limits)

        self.assertTrue(np.equal(expected_coords, coords).min())

    def test_extract_patches(self):
        '''
        Simple test to check we get the correct number of patches for a boring image
        '''
        # our default args have kernel [32,32,1], with stride [16,16,0]
        ct_array = np.ones((100,100,1))
        mask_array = np.zeros_like(ct_array)
        mask_array[4:96, 4:96, 0:1] = 1
        # padding = [14,14,0]
        bbox = [[4,4,0],[96,96,1]]
        args = self.args
        patches = extract_patches(ct_array, mask_array, bbox, args)

        self.assertTrue(len(patches)==33)

    def test_calculate_slices_and_bounding_boxes(self):
        mask_array = np.zeros((100,100,100))
        mask_array[10:90, 10:90, 10:90] =1
        bboxs = calculate_slices_and_bounding_boxes(mask_array, label=1)
        expected_bbox = [ [[10,10,z],[89,89,z+1]] for z in range(10,90) ]

        self.assertEqual(bboxs, expected_bbox)

if __name__ == "__main__":
    unittest.main()