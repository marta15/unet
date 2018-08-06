import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from itertools import izip

#from fcn_model import fcn_model
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES

from PIL import Image

from time import *
import scipy

seed = 5321
np.random.seed(seed)

SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = './'

RESCALE = True

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'SCD_ManualContours')
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'SCD_Images')


def shrink_case(case):
    toks = case.split('-')
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return '-'.join([shrink_if_number(t) for t in toks])


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-.*', ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
        print("Created contour for case "+self.case)
    
    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)
    
    __repr__ = __str__


def read_contour(contour, data_path):
    filename = 'DICOM/IM-0001-%04d.dcm' % (contour.img_no)
    full_path = os.path.join(data_path, contour.case, filename)
    target_dir = "./tmp/"
    f = dicom.read_file(full_path)
    img = f.pixel_array

    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)

    img_path = target_dir + str(f.PatientsName).rjust(4, '0') + "_" + str(time()) + ".jpg"
    scipy.misc.imsave(img_path, img)
    img = cv2.imread(img_path, 0)

    mask_path = target_dir + str(f.PatientsName).rjust(4, '0') + "_" + str(time()) + "_l.jpg"
    scipy.misc.imsave(mask_path, mask)
    #mask = cv2.imread(mask_path, 0)

    '''
    if f[0x18, 0x1312].value == "COL":
            # rotate counter clockwise when image is column oriented.
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)
            mask = cv2.transpose(mask)
            mask = cv2.flip(mask, 0)
           # print "col"
    '''

    # rescaling needed due to different pixel spacing
    if RESCALE:
        scale = f.PixelSpacing[0]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)

    img = cv2.fastNlMeansDenoising(img,None,5,7,21)

    # contrast stretching
    clahe = cv2.createCLAHE(tileGridSize=(1, 1))
    cl_img = clahe.apply(img)
  
    if cl_img.ndim < 3:
        cl_img = cl_img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return cl_img, mask


def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours


def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask
        
    return images, masks


if __name__== '__main__':

    contour_type = "i"
    crop_size = 128

    print('Mapping ground truth '+contour_type+' contours to images in train...')
    train_ctrs = map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True)
    print('Done mapping training set')
    
    split = int(0.1*len(train_ctrs))
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                crop_size=crop_size)
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(dev_ctrs,
                                            TRAIN_IMG_PATH,
                                            crop_size=crop_size)
    
    X = np.array(img_train, dtype=np.float32)
    y = np.array(mask_train)
    np.save('data/X_train.npy', X)
    np.save('data/y_train.npy', y)
    X_v = np.array(img_dev, dtype=np.float32)
    y_v = np.array(mask_dev)
    np.save('data/X_validate.npy', X_v)
    np.save('data/y_validate.npy', y_v)

    for i in range(y_v.shape[0]):
        img = y_v[i]
        img = array_to_img(img)
        img.save("./dev/%d_l.jpg"%(i))
    for i in range(X_v.shape[0]):
        img = X_v[i]
        img = array_to_img(img)
        img.save("./dev/%d.jpg"%(i))     

    print X.shape,y.shape

    
    imgs = np.load('./data/X_train.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("./train/%d.jpg"%(i))
        if i%100 == 0:
            print(i)

    imgs = np.load('./data/X_validate.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("./test/%d.jpg"%(i))
        if i%100 == 0:
            print(i)

    imgs = np.load('./data/y_train.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("./label/%d.jpg"%(i))
        if i%100 == 0:
            print(i)

    imgs = np.load('./data/y_validate.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("./test_label/%d.jpg"%(i))
        if i%100 == 0:
            print(i)
    
    print('Done.')