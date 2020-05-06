import cv2
import numpy as np
from PIL import Image

"""from PIL import Image
im = Image.open('frame22.png')
pixelMap = im.load()

img = Image.new(im.mode, im.size)
pixelsNew = img.load()
for i in range(img.size[0]):
    for j in range(img.size[1]):
        if 255 in pixelMap[i,j]:
            pixelsNew[i,j] = (-1,-1,-1,255)
        else:
            pixelsNew[i,j] = pixelMap[i,j]
im.close()
img.show()
img.save("out.png")
img.close()"""

img = cv2.imread('frame22.png',0)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilationcomerosion = cv2.dilate(erosion,kernel,iterations = 1)

#liberdade de resize da janela
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
cv2.namedWindow("dilationcomerosion", cv2.WINDOW_NORMAL)

cv2.imshow('Input', img)
cv2.imshow('dilationcomerosion', dilationcomerosion)



#rows, cols = img.shape
#with open('mycsv.csv', 'w') as f:
#    thewriter = csv.writer(f)
#    for i in range(rows):
#        for j in range(cols):
#            k = img[i,j]
#            thewriter.writerows(k)


cv2.waitKey(0)
