#!/usr/bin/env python3

""" Low Cost Transfert Learning on CIBR with Inceptionv3 ConvNet

Description:
============

see this script as a disappointment to me.

Was hoping to correctly use ~~~InceptionV3~~~ VGG16 model by freezing the layers and fitting data generator to train this ConvNet.

The current script collect extracted features from ~~~InceptionV3~~~ VGG16 and names to write Hierarchical Data Format file.

Required setup:
===============

$ git clone https://github.com/aryapei/In-shop-Clothes-From-Deepfashion.git
$ rsync -a ./In-shop-Clothes-From-Deepfashion/Img/MEN/ ./In-shop-Clothes-From-Deepfashion/Img/WOMEN/

Thoses commands clone and merge current Fashion dataset hosted at https://github.com/aryapei/In-shop-Clothes-From-Deepfashion in the same folder ./In-shop-Clothes-From-Deepfashion/Img/WOMEN/
"""

import numpy as np
from numpy import linalg as LA
import os
import h5py
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing  import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob

class ConvNet:
    def __init__(self):
        self.model = VGG16(input_shape=(244, 244, 3), weights="imagenet", include_top=False, pooling="max")
        self.model.predict(np.zeros((1, 244, 244, 3)))

    '''
    Use inceptionv3 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(244,244))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

if __name__ == "__main__":

    db = "./dataset-retr/train"
    img_list = glob(f"{db}/*.jpg")
    #print(img_list)
    
    print(f"{' feature extraction starts ':=^120}")
    
    feats = []
    names = []

    model = ConvNet()
    for img_path in img_list:
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print(f"feat extraction of {img_name}.")

    feats = np.array(feats)
    names = np.string_(names)
    
    print(f"{' writing feature extraction results ':=^120}")
    h5f = h5py.File("featureCNN.h5", 'w')
    h5f.create_dataset('dataset_feat', data=feats)
    h5f.create_dataset('dataset_name', data=names)
    h5f.close()
    
    # Read the produced files :
    
    h5f = h5py.File('./featureCNN.h5', 'r')
    feats = h5f['dataset_feat'][:]
    imgNames = h5f['dataset_name'][:]
    h5f.close()
    
    print(f"{' searching starts ':=^120}")
    queryDir = './dataset-retr/train/ukbench00033.jpg'
    queryImg = mpimg.imread(queryDir)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(queryImg)
    plt.title("Query Image")
    plt.axis('off')

    model = ConvNet()
    queryVec = model.extract_feat(queryDir)
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    
    # number of top retrieved images to show
    maxres = 3
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)


    # show top #maxres retrieved result one by one
    db_path = "./dataset-retr/train"
    for i, im in enumerate(imlist):

        image = mpimg.imread(f"{db_path}/{im.decode('utf-8')}")
        plt.subplot(2, 3, i+4)
        plt.imshow(image)
        plt.title("search output %d" % (i + 1))
        plt.axis('off')
    plt.show()
