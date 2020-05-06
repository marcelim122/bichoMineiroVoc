from PIL import Image
import os


images = os.listdir(r"D:\UFU\IC\bichoMineiroVoc\TrainSet\Images")
path = r"D:\UFU\IC\bichoMineiroVoc\TrainSet\Images"
for image in images:
	print(image)
	Image.open(path+"\\"+image).convert('RGB').save(path+"\\"+image)

images = os.listdir(r"D:\UFU\IC\bichoMineiroVoc\TrainSet\Labels")
path = r"D:\UFU\IC\bichoMineiroVoc\TrainSet\Labels"
for image in images:
	print(image)
	Image.open(path+"\\"+image).convert('RGB').save(path+"\\"+image)

images = os.listdir(r"D:\UFU\IC\bichoMineiroVoc\Teste\Images")
path = r"D:\UFU\IC\bichoMineiroVoc\Teste\Images"
for image in images:
	print(image)
	Image.open(path+"\\"+image).convert('RGB').save(path+"\\"+image)

images = os.listdir(r"D:\UFU\IC\bichoMineiroVoc\Teste\Labels")
path = r"D:\UFU\IC\bichoMineiroVoc\Teste\Labels"
for image in images:
	print(image)
	Image.open(path+"\\"+image).convert('RGB').save(path+"\\"+image)
	
images = os.listdir(r"D:\UFU\IC\bichoMineiroVoc\Validation\Images")
path = r"D:\UFU\IC\bichoMineiroVoc\Validation\Images"
for image in images:
	print(image)
	Image.open(path+"\\"+image).convert('RGB').save(path+"\\"+image)

images = os.listdir(r"D:\UFU\IC\bichoMineiroVoc\Validation\Labels")
path = r"D:\UFU\IC\bichoMineiroVoc\Validation\Labels"
for image in images:
	print(image)
	Image.open(path+"\\"+image).convert('RGB').save(path+"\\"+image)
		

#for i in range(1207):
#	Image.open("img"+str(i)+".png").convert('RGB').save("img"+str(i)+".png")
	