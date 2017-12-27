# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:50:53 2017

@author: aumale
"""
import zipfile as zipf
import scipy
import numpy as np



class LoadCelebA:
    def __init__(self,zipLocation= './Data/', one_hot=True, flatten=True,start_index = 1, scale =100):
        self.zipLocation = zipLocation
        self.zip = zipf.ZipFile(self.zipLocation+'img_align_celeba.zip','r')
        self.filelist = self.zip.namelist()
        self.filelist.sort()
        self.validation_index = int(0.8*np.array(self.filelist).size)
        self.test_index = int(0.9*np.array(self.filelist).size)
        self.label = np.loadtxt(self.zipLocation+"list_attr_celeba.txt", skiprows =2,converters = {0: id}, usecols = 16)
        self.next_batch_index = start_index   
        self.one_hot = one_hot
        self.flatten = flatten
        self.scale = scale
        
    def load(self,count = 10, start_index=1, mode = 'L'):
        
        celeb_img = []
        celeb_label = []
        end_index = start_index+count                
        
        for index, file in enumerate(self.filelist[start_index : end_index], start= start_index):
            
            with self.zip.open(file, 'r') as img:
                img_arr = scipy.misc.imread(img,mode='L')
                img_arr= scipy.misc.imresize(img_arr, (180, 220))
                img_dimy = img_arr[0].size
                img_dimx = img_arr.T[0].size
                
                #resize image
                img_dimx = abs(img_dimx *(self.scale/100))
                img_dimy = abs(img_dimy *(self.scale/100))

                #make 0 = black and 1 = white and scale rest in between
                img_arr = abs((img_arr-255.0)*1/255).round(3)
                
                #resize the image to 28*28
                #img_arr= scipy.misc.imresize(img_arr, (img_dimx, img_dimy))
                img_arr= scipy.misc.imresize(img_arr, (28*4, 28*4))
                
                #flatten image to give flat vector instead of 28*28 matrix
                if self.flatten == True:
                    img_arr = img_arr.flatten()
                    
                celeb_img.append(img_arr)
                
        for index in range(start_index-1, end_index-1):
            new_lbl = [0]*2
            if self.one_hot == True:
                if self.label[index] == 1:
                    new_lbl[0] = 1
                else:
                    new_lbl[1] = 1
                celeb_label.append(new_lbl)
            else:
                celeb_label = self.label[start_index-1:end_index]
                
                        
        celeb_img = np.array(celeb_img)
        celeb_label = np.array(celeb_label, dtype='uint8')
            
        return celeb_img, celeb_label

    
    def validationSet(self):
        return self.load(count = 100, start_index = self.validation_index)
    
    
    def testSet(self):
        return self.load(count = 100, start_index = self.test_index)
        
    
    def next_batch(self,batch_size = 100):
        img, label = self.load(count=batch_size,start_index = self.next_batch_index)
        self.next_batch_index += batch_size
        return img, label
        