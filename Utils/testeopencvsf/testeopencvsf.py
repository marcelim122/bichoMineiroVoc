import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import glob


def mostrar_imagem(img, titulo):
    plt.imshow(img)
    plt.title(titulo)
    plt.colorbar()
    plt.show()

#Inicialização
# Carregar imagem da internet
#resp = urllib.request.urlopen("https://i.stack.imgur.com/gVDrL.png")
#img = cv2.imread('framepng/frame24.png', 1)
image = glob.glob('imagemcomparacao/*.png')
kernel = np.ones((5,5),np.uint8)
#mostrar_imagem(img, "Imagem lida")

save = 'imagemcomparacao/Teste/Labels/'
quantidade = 0

#convertendo em bytearray o que foi lido no request do url
#img = np.asarray(bytearray(resp.read()), dtype="uint8")
#img = cv2.imdecode(img, cv2.IMREAD_COLOR)

for i, imageFile in enumerate(image):
    img = cv2.imread(imageFile, 1)
    folder_0,imageName = imageFile.split("\\")
    #aplicando erode e dilate para melhorar o contorno, por ser 1px de espessura
    erosion = cv2.erode(img, kernel, iterations = 1)
    img = cv2.dilate(erosion, kernel, iterations = 1)
    #mostrar_imagem(img, "Imagem após erode e dilate")
    
    #remover o cinza, pois ele ja esta em escala de cinza lido pelo .imread()
    # Converte para escala de cinza
    cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #mostrar_imagem(cinza, "Variavel cinza, GRAY")
    
    # Obtém o binário
    #_, binario = cv2.threshold(cinza, 127, 255, cv2.THRESH_BINARY_INV)
    _, binario = cv2.threshold(cinza, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Cria uma cópia para manter a original
    copia = img.copy()
    
    #procurar os contornos, procura contornos dentro dos contornos, aproxima os contornos, fundo deve ser preto e objeto deve ser branco (gray)
    tmp = cv2.findContours(binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # compatibilidade entre versões diferentes do OpenCV
    contornos = tmp[0] if len(tmp) == 2 else tmp[1]
    #retorna 3 parametros, 1)imagens, 2)contornos, 3)hierarquia dos contornos, (python list)
    
    #Conversão de escala de cinza para BGR
    img_contornos = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    #mostrar_imagem(img_contornos, "Imagem contornos, BGR")
    
    #Desenha os contornos
    #imagem, contornos, (-1) são todos, a cor, espessura
    teste1 = cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 1)
    mostrar_imagem(teste1, "Desenho do contorno")
    
    # Extrai somente o contorno de espessura 1 px
    mask = cv2.inRange(img_contornos, (0, 254, 0), (0, 255, 0))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #mostrar_imagem(mask, "Mask, gray")
    
    # Substitui os valores da máscara por -1
    binario = cv2.cvtColor(binario, cv2.COLOR_GRAY2RGB)
    binario = np.int16(binario)
    #mostrar_imagem(binario, "binario GRAY")
    #binario = cv2.cvtColor(binario, cv2.COLOR_GRAY2RGB)
    binario[mask == 255] = 1
    #mostrar_imagem(binario, "binario RGB -1")

    """# Mostrar os valores -1 como cor branca
    binario = np.ma.masked_where(binario == 170, binario)
    mostrar_imagem(binario, "Imagem binario, apos trocar mascara para -1")
    cmap = matplotlib.cm.Greys  # Can be any colormap that you want after the cm
    cmap.set_bad(color='red')
    plt.title("Imagem com mapa do contorno em vermelho")
    plt.imshow(binario, cmap=cmap)
    plt.show()"""
    
    
    #salvar imagem do matplotlib que estava em binario
    #teste = binario
    #teste = teste.astype('uint8')
    #mostrar_imagem(binario, "Imagem binario, GRAY")
    #binario = binario.astype('uint8')
    #mostrar_imagem(binario, "teste")
    binario = binario.astype('uint8')
    #binario = cv2.cvtColor(binario, cv2.COLOR_BGR2RGB) #############
    #teste=binario
    print(imageName)
    print(np.unique(binario))
    
    result_save = save + imageName
    
    matplotlib.image.imsave(result_save, binario)
    
    quantidade=quantidade+1
    #imagem_rgb = binario[:,:,::-1]

print("Quantidade convertida: %d" %(quantidade))

"""
teste = cv2.cvtColor(binario, cv2.COLOR_RGB2GRAY)
rett, thresh2 = cv2.threshold(binario, 90, 255, cv2.THRESH_BINARY)
binario[thresh2==255]=170 #praga para 170
binario[thresh2==0]=255 #resto para 255
binario[thresh2==170]=0 #praga 170 para 0
binario = cv2.cvtColor(binario, cv2.COLOR_GRAY2RGB)
mostrar_imagem(binario, "teste branco")
"""