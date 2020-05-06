# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:52:07 2020

@author: marce
"""

from PIL import Image
import glob

im1 = Image.open('ResultFrame/frame24-0049.png')
im2 = Image.open('imagem/imagem.png')

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (2720,1530))
    dst.paste(im1, (0,0))
    dst.paste(im2, (im1.width, 0))
    return dst

get_concat_h(im1, im2).save('imagem/imagem.png')

"""
#im1 = Image.open('ResultFrame/frame24-0000.png')
#im2 = Image.open('ResultFrame/frame24-0001.png')

list1 = glob.glob('ResultFrame/*.png')
print(list1)


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (2720,1530))
    dst.paste(im1, (0,0))
    dst.paste(im2, (0, im1.height))
    return dst

#get_concat_h(im1, im2).save('imagem/imagem.png')


def get_concat_h_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h(_im, im)
    return _im

get_concat_h_multi_blank(list1).save('imagem/imagem.png')
"""

"""
import cv2
import numpy
import glob
import os

dir = "ResultFrame/" # current directory
ext = ".png" # whatever extension you want

pathname = os.path.join(dir, "*" + ext)
images = [cv2.imread(img) for img in glob.glob(pathname)]

height = sum(image.shape[0] for image in images)
width = max(image.shape[1] for image in images)
output = numpy.zeros((height,width,3))

y = 0
for image in images:
    h,w,d = image.shape
    output[y:y+h,0:w] = image
    y += h
    
cv2.imwrite("imagem/test.png", output)
"""

"""
from PIL import Image
import glob

def main():
    layers_l = glob.glob("ResultFrame/*.png")
    print(layers_l)
    layers_l = map(lambda x :Image.open(x) ,layers_l)
    
    image  = Image.new( "RGB", (2720, 1530))
    image.save("imagem/new_image.png")
    base = Image.open("imagem/new_image.png")
    
    for layer in layers_l[1:]:
        base.paste(layer, (0,0), layer.convert("RGB") )
        image.save("imagem/new_image.png")
        print (layer)
    
    print (image.info)
    
if ( __name__ == "__main__" ): main()
"""