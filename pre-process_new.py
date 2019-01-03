# -*- coding: utf-8 -*-

import tarfile
import scipy.io
import numpy as np
import os
import cv2 as cv
import shutil
import random
from console_progressbar import ProgressBar
import time

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def find_paths(path):
    paths = []
    level_a = os.listdir(path)
    #print level_a
    for level_name in level_a:
    #   print level_name
        for image_name in os.listdir(os.path.join(path, level_name)):
            paths += [os.path.join(path, level_name, image_name)]

    return paths

def save_train_and_test_data(fnames, fnamesPaste, dict_labels):

    src_folder = 'data/DeepLearningFilesPosAug'
    num_samples = len(fnames)
    print 'num_samples: ', num_samples #1903

    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    print 'num_train: ', num_train #1522

    train_indexes = random.sample(range(num_samples), num_train)
    print 'train_indexes: ', train_indexes

    pb = ProgressBar(total=100, prefix='Save train and test data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        fnamePaste = fnamesPaste[i]
        # print 'fname: ', fname
        # print 'fnamePaste', fnamePaste
        #label = labels[i]
        #(x1, y1, x2, y2) = bboxes[i]

        src_path = os.path.join(src_folder, fnamePaste)
        src_image = cv.imread(src_path)
        # print 'src_path: ', src_path
        # print 'src_image: ', src_image
        
        #height, width = src_image.shape[:2]

        # margins of 16 pixels
        # margin = 16
        # x1 = max(0, x1 - margin)
        # y1 = max(0, y1 - margin)
        # x2 = min(x2 + margin, width)
        # y2 = min(y2 + margin, height)
        # print("{} -> {}".format(fname, label))
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        if i in train_indexes:
            dst_folder = 'data/train'
        else:
            dst_folder = 'data/test'


        label = dict_labels[fnamePaste[:4]]
        print 'label: ', label

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)

        # print 'src_image: ', src_image
        # crop_image = src_image[y1:y2, x1:x2]
        # dst_img = cv.resize(src=crop_image, dsize=(500, 500))
        # print 'dst_img: ', dst_img
        cv.imwrite(dst_path, src_image)

# def save_test_data(fnames, bboxes):
#     src_folder = 'cars_test'
#     dst_folder = 'data/test'
#     num_samples = len(fnames)

#     pb = ProgressBar(total=100, prefix='Save test data', suffix='', decimals=3, length=50, fill='=')

#     for i in range(num_samples):
#         fname = fnames[i]
#         (x1, y1, x2, y2) = bboxes[i]
#         src_path = os.path.join(src_folder, fname)
#         src_image = cv.imread(src_path)
#         height, width = src_image.shape[:2]
#         # margins of 16 pixels
#         margin = 16
#         x1 = max(0, x1 - margin)
#         y1 = max(0, y1 - margin)
#         x2 = min(x2 + margin, width)
#         y2 = min(y2 + margin, height)
#         # print(fname)
#         pb.print_progress_bar((i + 1) * 100 / num_samples)

#         dst_path = os.path.join(dst_folder, fname)
#         crop_image = src_image[y1:y2, x1:x2]
#         dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
#         cv.imwrite(dst_path, dst_img)


# def process_train_data():
#     print("Processing train data...")
#     cars_annos = scipy.io.loadmat('devkit/cars_train_annos')
#     annotations = cars_annos['annotations']
#     annotations = np.transpose(annotations)

#     fnames = []
#     class_ids = []
#     bboxes = []
#     labels = []

#     for annotation in annotations:
#         bbox_x1 = annotation[0][0][0][0]
#         bbox_y1 = annotation[0][1][0][0]
#         bbox_x2 = annotation[0][2][0][0]
#         bbox_y2 = annotation[0][3][0][0]
#         class_id = annotation[0][4][0][0]
#         labels.append('%04d' % (class_id,))
#         fname = annotation[0][5][0]
#         bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
#         class_ids.append(class_id)
#         fnames.append(fname)

#     labels_count = np.unique(class_ids).shape[0]
#     print(np.unique(class_ids))
#     print('The number of different cars is %d' % labels_count)

#     save_train_data(fnames, labels, bboxes)


# def process_test_data():
#     print("Processing test data...")
#     cars_annos = scipy.io.loadmat('devkit/cars_test_annos')
#     annotations = cars_annos['annotations']
#     annotations = np.transpose(annotations)

#     fnames = []
#     bboxes = []

#     for annotation in annotations:
#         bbox_x1 = annotation[0][0][0][0]
#         bbox_y1 = annotation[0][1][0][0]
#         bbox_x2 = annotation[0][2][0][0]
#         bbox_y2 = annotation[0][3][0][0]
#         fname = annotation[0][4][0]
#         bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
#         fnames.append(fname)

#     save_test_data(fnames, bboxes)


if __name__ == '__main__':
    # parameters
    img_width, img_height = 500, 500

    # print('Extracting cars_train.tgz...')
    # if not os.path.exists('cars_train'):
    #     with tarfile.open('cars_train.tgz', "r:gz") as tar:
    #         tar.extractall()
    # print('Extracting cars_test.tgz...')
    # if not os.path.exists('cars_test'):
    #     with tarfile.open('cars_test.tgz', "r:gz") as tar:
    #         tar.extractall()
    # print('Extracting car_devkit.tgz...')
    # if not os.path.exists('devkit'):
    #     with tarfile.open('car_devkit.tgz', "r:gz") as tar:
    #         tar.extractall()

    #cars_meta = scipy.io.loadmat('devkit/cars_meta')

    #fnames = np.array(['Onix', 'HB20', 'Sand', 'VGol', 'Ford'])  # shape=(1, 5)
    #class_names = np.transpose(class_names)
    #fnames.shape = (5, 1)
    #print('class_names.shape: ' + str(fnames.shape))
    #print('Sample class_name: [{}]'.format(fnames[3][0]))

                    #chave : valor
    dict_labels = { 
                    'Onix' : '001', 
                    'HB20' : '002',
                    'Sand' : '003',
                    'VGol' : '004',
                    'Ford' : '005'
                }

    # #Apenas para testar se os valores e chaves do cicionário estão de acordo
    # print 'dict_labels: ', dict_labels
    # print 'keys: ', dict_labels.keys()
    # for typeCar in dict_labels:
    #     values = dict_labels[typeCar]
    #     print 'values:', values

    ensure_folder('data/train')
    #ensure_folder('data/valid')
    ensure_folder('data/test')

    base_path = 'data/DeepLearningFilesPosAug'
    paths = find_paths(base_path)

    #Carrega os nomes das imagens em 'fnames'
    fnames = []
    fnamesPaste = []
    for image_path in paths:
        #print image_path
        #im_data = load_image_data(image_path)
        im_data = cv.imread(image_path)
        fname = image_path[34:]
        fnamePaste = image_path[29:]
        #print 'fnamesPaste: ', fnamePaste
        fnames.append(fname)
        fnamesPaste.append(fnamePaste)

    #print 'fnamesPaste: ', fnamesPaste

    save_train_and_test_data(fnames, fnamesPaste, dict_labels)

    #process_train_data()
    #process_test_data()

    # clean up
    #shutil.rmtree('cars_train')
    #shutil.rmtree('cars_test')
    # shutil.rmtree('devkit')
