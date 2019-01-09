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
    for level_name in level_a:
        for image_name in os.listdir(os.path.join(path, level_name)):
            paths += [os.path.join(path, level_name, image_name)]

    return paths

#Deixa as imagens no mesmo tamanho
def load_image_data(path):

    image = cv.imread(path)    
    if image.shape[0] < image.shape[1]:
        image = cv.flip(image, flipCode=1)
    im_data = cv.resize(image, (224,224))
    return im_data

def save_train_and_test_data(fnames, fnamesPaste, imagesPaths, dict_labels):

    # src_folder = 'data/DeepLearningFilesPosAug'
    # src_folder = '/home/ml/datasets/DeepLearningFiles'

    num_samples = len(fnames)
    # print 'num_samples: ', num_samples #1903

    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    # print 'num_train: ', num_train #1522

    train_indexes = random.sample(range(num_samples), num_train)
    # print 'train_indexes: ', train_indexes

    pb = ProgressBar(total=100, prefix='Save train and test data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        fnamePaste = fnamesPaste[i]
        image_path = imagesPaths[i]
        # print 'fname: ', fname
        # print 'fnamePaste', fnamePaste

        # src_path = os.path.join(src_folder, fnamePaste)

        #Deixa as imagens originais no mesmo tamanho
        im_data = load_image_data(image_path)

        # src_image = cv.imread(im_data)
        # print 'src_path: ', src_path
        # print 'src_image: ', src_image
        
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        if i in train_indexes:
            dst_folder = 'data/pre_train'
        else:
            dst_folder = 'data/test'

        label = dict_labels[fnamePaste[:4]]
        # print 'label: ', label

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)

        
        cv.imwrite(dst_path, im_data)

    # save_valid_data(num_train)


# def save_valid_data(num_pretrain):

#     src_folder = 'data/pre_train'
#     # print 'num_pretrain: ', num_pretrain #1522 e 1370 de treino, tirando os 10% de validação

#     valid_split = 0.1
#     num_valid = int(round(num_pretrain * valid_split))
#     # print 'num_valid: ', num_valid #152

#     valid_indexes = random.sample(range(num_pretrain), num_valid)
#     # print 'valid_indexes: ', valid_indexes

#     pb = ProgressBar(total=100, prefix='Save valid data', suffix='', decimals=3, length=50, fill='=')

#     paths = find_paths(src_folder)

#     #Carrega os nomes das imagens em 'fnames'
#     fnames = []
#     fnamesPaste = []
#     for image_path in paths:
#         # print image_path
#         fname = image_path[20:]
#         # print 'fname: ', fname
#         fnamePaste = image_path[15:]
#         # print 'fnamesPaste: ', fnamePaste
#         fnames.append(fname)
#         fnamesPaste.append(fnamePaste)

#     # print 'fnames: ', fnames
#     # print 'fnamesPaste: ', fnamesPaste

#     for i in range(num_pretrain):
#         fname = fnames[i]
#         fnamePaste = fnamesPaste[i]
#         # print 'fname: ', fname
#         # print 'fnamePaste', fnamePaste

#         src_path = os.path.join(src_folder, fnamePaste)
#         src_image = cv.imread(src_path)
#         # print 'src_path: ', src_path
#         # print 'src_image: ', src_image
        
#         pb.print_progress_bar((i + 1) * 100 / num_pretrain)

#         if i in valid_indexes:
#             dst_folder = 'data/valid'
#         else:
#             dst_folder = 'data/train'


#         label = fnamePaste[:4]
#         # print 'label valid: ', label

#         dst_path = os.path.join(dst_folder, label)
#         if not os.path.exists(dst_path):
#             os.makedirs(dst_path)
#         dst_path = os.path.join(dst_path, fname)

#         cv.imwrite(dst_path, src_image)


if __name__ == '__main__':
    # parameters
    img_width, img_height = 224, 224

                    #chave : valor
    dict_labels = { 
                    'onix' : '0001', 
                    'hb20' : '0002',
                    'sand' : '0003',
                    'vgol' : '0004',
                    'ford' : '0005'
                }

    # ensure_folder('data/pre_train')
    ensure_folder('data/pre_train')
    # ensure_folder('data/valid')
    ensure_folder('data/test')

    base_path = '/home/ml/datasets/DeepLearningFiles'
    paths = find_paths(base_path)

    #Carrega os nomes das imagens em 'fnames'
    fnames = []
    fnamesPaste = []
    imagesPaths = []
    for image_path in paths:
        # print image_path
        fname = image_path[41:]
        # print 'fname: ', fname
        fnamePaste = image_path[36:]
        # print 'fnamesPaste: ', fnamePaste
        fnames.append(fname)
        fnamesPaste.append(fnamePaste)
        imagesPaths.append(image_path)

    #print 'fnamesPaste: ', fnamesPaste

    save_train_and_test_data(fnames, fnamesPaste, imagesPaths, dict_labels)