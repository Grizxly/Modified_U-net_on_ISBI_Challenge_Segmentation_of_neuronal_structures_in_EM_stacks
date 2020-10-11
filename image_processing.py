import numpy as np
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import os
from skimage.segmentation import find_boundaries
from skimage.morphology import label



# hyper-parameters
LABEL_RECOVER_SIGMA = 1 # Gaussian filter for recover the property of deformed label
LABEL_RECOVER_THRESH = 155
WEIGHT_MAP_W0 = 10
WEIGHT_MAP_SIGMA = 5
WEIGHT_MAP_NEIGHB = 4 # int, 4 or 8




def object_separation(mask, neighb=4):
    """
    separate objects in mask and cast each object into a new layer
    :param mask: dnarray
                input binary mask, shape(height, width)
    :param neighb: int, optional
                could be 4 or 8, indicates number of neighbors for sliding window
    :return: ndarray
                output separated object layers, shape(num_obj, height, width)
    """
    height, width = mask.shape
    labeled, n = label(mask, neighbors=neighb, background=0, return_num=True)
    sep_masks = np.zeros((n, height, width))
    for i in range(n):
        sep_masks[i] = np.where(labeled==i+1, 1, 0)
    return sep_masks, n


def get_weight_map(mask, w0, sigma, neighb=4):
    """
    get weight map using method in
    <<U-Net: Convolutional Networks for Biomedical Image Segmentation>>
    :param mask:ndarray
                input binary mask, shape(height, width)
    :param w0:int
                reference to the paper above
    :param sigma:int
                referece to the paper above
    :param neighb:int, optional
                could be 4 or 8, indicates number of neighbors for sliding window
    :return: ndarray
                weight map for input mask, shape(height, width)
    """
    height, width = mask.shape
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    mask = (mask > 127).astype(int) # trans to boolean map
    # separate cells
    sep_masks, obnum = object_separation(mask, neighb=neighb) #shape(num_obj, weight, width)
    if obnum < 2:
        print('Err: get_weight_map(): object number < 2. exit')
        exit()
    mask_x, mask_y = np.meshgrid(np.arange(height), np.arange(width)) # coordinates of mask
    mask_x, mask_y = np.c_[mask_x.reshape(-1), mask_y.reshape(-1)].astype(np.int32).T # shape(height*width,)
    dist_mat = np.zeros((obnum, height*width)) # from each mask pix, shortest dist to each obj's bond.
    # calculate dist_mat
    for i, obj in enumerate(sep_masks, 0):
        obj_boundaries = find_boundaries(obj, mode='inner')
        bond_x, bond_y = np.nonzero(obj_boundaries) # coordinates of obj boundaries pixel, shape(num_bound_pixel,)
        bond_x = bond_x.astype(np.int32) # int64-->int32, reduce memory cost
        bond_y = bond_y.astype(np.int32)
        ob_dist = np.zeros((height * width))
        # compute ob_list(height*width,) i.e. for each object mask pixel, compute its shortest dist to object bound pixel
        for j in range(height * width):
            x_sqsum = (bond_x.ravel() - mask_x[j]) ** 2 # shape(num_obj_bound_pixel,)
            y_sqsum = (bond_y.ravel() - mask_y[j]) ** 2 # shape(num_obj_bound_pixel,)
            ob_dist[j] = np.sqrt(x_sqsum + y_sqsum).min()
        dist_mat[i] = ob_dist
        print('\rtotal_num_obj:', obnum, '  |   obj_dist done:', i, end='')
    d1 = np.zeros((height*width))
    d2 = np.zeros((height*width))
    for k in range(d1.shape[0]):
        d1[k] = np.sort(dist_mat[:,k])[0]
        d2[k] = np.sort(dist_mat[:,k])[1]
    bond_loss_arr = w0 * np.exp((-1 * (d1 + d2) ** 2 / (2 * sigma ** 2))) # shape(height*width,)

    bond_loss_map = np.zeros((height, width))
    bond_loss_map[mask_x, mask_y] = bond_loss_arr
    # compute probability map
    clas_1 = 1 - (mask.sum() / mask.size)
    clas_2 = 1 - clas_1
    prob_map = np.where(mask>0, clas_1, clas_2)
    weight_map = bond_loss_map + prob_map
    print('\nweight_map made.')
    return weight_map


def feature_map_crop(feature_map, out_shape):
    """
    crop the feature map w.r.t output Height and Width, remain the other dimensions unchanged
    :param feature_map: 4-d nparray
                        shape of (batch, height, width, channel)
    :param out_shape: tuple/list
                      desired output shape, format of (height, width)
    :return: 4-d nparray
             cropped feature map
    """
    _, height, width, _ = feature_map.shape # expect (batch, height, width, channel)
    if height < out_shape[0] or width < out_shape[1]:
        print('err: image_crop expect out_size NOT smaller than image shape:')
        print('ori_height:', height)
        print('ori_width:', width)
        print('out_height:', out_shape[0])
        print('out_width:', out_shape[1])
        print('exit')
        exit()
    center = (height // 2, width // 2)
    up_bond = center[0] - (out_shape[0] // 2)
    left_bond = center[1] - (out_shape[1] // 2)
    #print('result_shape:', feature_map[:, up_bond:up_bond + out_shape[0], left_bond:left_bond + out_shape[1], :])
    return feature_map[:, up_bond:up_bond + out_shape[0], left_bond:left_bond + out_shape[1], :]


def image_elastic_deformation(img, lbl, sigma, alpha, seed=None):
    """
    Do elastic deformation to an image and its corresponding label
    using bi-cubic interpolation on Gaussian filtered random displacement field.
    reference to:
    [2003] Patrice Y. Simard, Dave Steinkraus, John C. Platt
    "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"

    :param img: ndarray
                input images
    :param lbl: ndarray
                label of the image
    :param sigma: integer
                standard deviation, as elasticity coefficient
    :param alpha: number
                scaling factor, as the intensity of the deformation
    :param seed: integer
                a seed to generate random displacement field
    :return: list, list
                elastic deformation result of input image, label
    """
    height, width = img.shape
    rand = np.random.RandomState(seed)

    # generate random displacement fields, normalize to norm of 1
    dx = (rand.rand(height, width) * 2 - 1)
    dy = (rand.rand(height, width) * 2 - 1)

    # displacement fields convolved with Gaussian filter with standard deviation sigma,
    # and multiply by scaling factor alpha
    dx = gaussian_filter(dx, sigma, mode='mirror') * alpha
    dy = gaussian_filter(dy, sigma, mode='mirror') * alpha

    # create coordinate arrays
    rowval, colval = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # create indices(1d arrays) to do interpolation
    inds = np.reshape(rowval + dx, (-1, 1)), np.reshape(colval + dy, (-1, 1))

    # do interpolation and reshape
    imgresult = np.reshape(map_coordinates(img, inds, order=3, mode='mirror'), img.shape)
    labelresult = np.reshape(map_coordinates(lbl, inds, order=2, mode='mirror'), lbl.shape)

    # post-process to recover the property of the deformed lbl
    labelresult =  gaussian_filter(labelresult, LABEL_RECOVER_SIGMA, mode='mirror')
    _, labelresult = cv.threshold(labelresult, LABEL_RECOVER_THRESH, 255, cv.THRESH_BINARY)

    return imgresult, labelresult


def image_rot_flip(img, lb):
    """
    perform rotation and flip on input image and its corresponding label
    :param img: ndarray
                input image
    :param lb: ndarray
                label of image
    :return: list, list
                image, label
    """
    height, width = img.shape
    img_list = [img]
    label_list = [lb]
    # check the image shape and perform rotations(square = 3 rotations or rectangle = 1 rotations)
    if height == width:
        img_list.append(np.rot90(img, k=1))
        img_list.append(np.rot90(img, k=3))
        label_list.append(np.rot90(lb, k=1))
        label_list.append(np.rot90(lb, k=3))
    img_list.append(np.rot90(img, k=2))
    label_list.append(np.rot90(lb, k=2))

    # perform flip
    img_list.append(np.fliplr(img))
    img_list.append(np.flipud(img))
    label_list.append(np.fliplr(lb))
    label_list.append(np.flipud(lb))

    return img_list, label_list


def data_augmentation(img_list, label_list, quantity, sigma=10, alpha=120, seed = None, SAVE_DIR=None):
    """
    input a pair of lists of image/label, output lists of augmented image/label lists
    :param img_list: list
                    list contains input images
    :param label_list: list
                    list contains corresponding labels
    :param quantity: positive integer
                    indicate how many groups of deformed image/label and weight_map will be generate for each seed image
                    total number groups to be generate(for square img) = (quantity + 1) * #img * 6
    :param sigma: integer
                    standard deviation, as elasticity coefficient
                    default = 10
    :param alpha: number
                    scaling factor, as the intensity of the deformation
                    default = 180
    :param seed: integer
                    a random seed to generate random displacement field
                    default = None
    :param SAVE_DIR: str
                    directory to save the result
                    default = None, which is not to save
    :return: list, list, list
                    augmented images list, label list, weight map list
    """

    # make sure #img == #label
    if len(img_list) != len(label_list):
        print('err: expect same size of image_list and label_list, got ' + str(len(img_list)) + ' and ' + str(len(label_list)))
        exit()
    img_result_list = []
    label_result_list = []
    weight_map_list = []
    # generate new images/labels
    print('total num groups to be generated:', len(img_list) * 6 * (quantity+1))
    for i in range(len(img_list)):
        # get rotated and flipped image/label
        seed_img_list, seed_label_list = image_rot_flip(img_list[i], label_list[i])

        for sample in seed_img_list:
            img_result_list.append(sample)
        for gt in seed_label_list:
            label_result_list.append(gt)

        for j in range(quantity):
            for k in range(len(seed_img_list)):
                imgresult, labelresult = image_elastic_deformation(seed_img_list[k], seed_label_list[k],
                                                                   sigma=sigma, alpha=alpha, seed=seed + j)
                img_result_list.append(imgresult)
                label_result_list.append(labelresult)
                print('\rimg/label pair generated: ' + str(len(img_result_list)), end='')

    print('\nstart making weight map...')
    for l in range(len(label_result_list)):
        print('number to be make:', len(label_result_list),'  |   now making:', l+1)
        wm = get_weight_map(label_result_list[l], w0=WEIGHT_MAP_W0, sigma=WEIGHT_MAP_SIGMA, neighb=WEIGHT_MAP_NEIGHB)
        weight_map_list.append(wm)
    # save the samples & labels & weight_map if need
    if SAVE_DIR:
        print('\ndata generate complete! start saving...')
        if not os.path.exists(SAVE_DIR + '/dev_samples'):
            os.mkdir(SAVE_DIR + '/dev_samples')
        if not os.path.exists(SAVE_DIR + '/dev_labels'):
            os.mkdir(SAVE_DIR + '/dev_labels')
        if not os.path.exists(SAVE_DIR + '/weight_maps'):
            os.mkdir(SAVE_DIR + '/weight_maps')
        i = 0
        for image in img_result_list:
            cv.imwrite(SAVE_DIR + '/dev_samples/train' + str(i) + '.png', image, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
            i += 1
        i = 0
        for lab in label_result_list:
            cv.imwrite(SAVE_DIR + '/dev_labels/label' + str(i) + '.png', lab, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
            i += 1
        i = 0
        for wm in weight_map_list:
            wmimg = (wm*25).astype(np.uint8)
            cv.imwrite(SAVE_DIR + '/weight_maps/wm' + str(i) + '.png', wmimg, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
            i += 1
        print('img,label,weight_map groups saved:', i)
        print('save complete!')
    return img_result_list, label_result_list, weight_map_list


def image_mirror_padding(img, shape):
    """
    do mirror padding on input image and return result image
    :param img: ndarray
                input image
    :param shape: tuple
                (height, width) shape of output
                shape must be larger in terms of height and width than input image
    :return: ndarray
                output image
    """
    img_height, img_width = img.shape
    if shape[0] < img_height or shape[1] < img_width:
        print('err: output shape must be larger than input image.')
        exit()
    top = (shape[0] - img_height) // 2
    bottom = (shape[0] - img_height) - top
    left = (shape[1] - img_width) // 2
    right = (shape[1] - img_width) - left

    return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_REFLECT_101)
