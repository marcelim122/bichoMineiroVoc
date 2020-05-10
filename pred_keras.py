
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:14:44 2018

@author: v3n0w
"""

import tensorflow as tf
import imageio
import numpy as np
from sklearn.metrics import recall_score
import glob 
import matplotlib.pyplot as plt

ImageTest = glob.glob('Teste/Images/*.png')
ImagesLabel = glob.glob('Teste/Labels/*.png')

y_all_train = np.zeros((len(ImageTest), 256, 256, 1))

for i, imageFile in enumerate(ImageTest):
    label = imageio.imread(ImagesLabel[i])
    label = np.reshape(label[:,:,0], (256, 256, 1))
    label = label >= 1
    y_all_train[i, :,:,:] = label

model = tf.keras.models.load_model('model.h5')

TP = 0
TN = 0
FP = 0
FN = 0

val_y = y_all_train

save_dir = 'Result/'

for i, imageFile in enumerate(ImageTest):
    
    image = imageio.imread(imageFile)
    #folder_0,imageName = imageFile.split("\\")
    folde_0, folder_1, imageName = imageFile.split("/")
    image = np.reshape(image, (1, 256, 256, 3))
    image = image / 255.
    print(imageName)
    #result = model.predict([image, np.ones((1, 256, 256, 1))])
    result = model.predict([image])
    result = np.round(result)
    
    current_TP = np.count_nonzero(result * val_y[i, :, :, :])
    current_TN = np.count_nonzero((result - 1) * (val_y[i, :, :, :] - 1))
    current_FP = np.count_nonzero(result * (val_y[i, :, :, :] - 1))
    current_FN = np.count_nonzero((result - 1) * val_y[i, :, :, :])
    print("TP: " + str(current_TP))
    print("FP: " + str(current_FP))
    print("TN: " + str(current_TN))
    print("FN: " + str(current_FN))
    TP = TP + current_TP
    TN = TN + current_TN
    FP = FP + current_FP
    FN = FN + current_FN
    
    total = current_TP + current_TN + current_FP + current_FN
    
    print('Total number: %d'  %(total))
    print('Acc: %f'  %((current_TP + current_TN) / total))
    
    result_image = save_dir + imageName
    
    imageio.imsave(result_image, result[0, :, :, 0])


print('TP: %f' %(TP))
print('FP: %f' %(FP))
print('TN: %f' %(TN))
print('FN: %f' %(FN))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall / (precision + recall))
acc = (TP + TN) / (TP + TN + FP + FN)

print('Precision >')
print(precision)
print('Recall >')
print(recall)
print('F-Measure >')
print(f1)
print('Test Accuracy: %f' %(acc))


print (result.shape)

#Plot!!!
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(image[0, :, :, :])
plt.figure()
plt.imshow(result[0, :, :, 0])
plt.figure()
image[0, :, :, 1] = result[0, :, :, 0] > 127
plt.imshow(image[0, :, :, :])

