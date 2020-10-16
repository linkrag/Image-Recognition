from google_images_download import google_images_download   #importing the library

import numpy as np
from PIL import Image
import cv2
import os

busca1 = "Tijolo"
busca2 = "telha"
busca3 = "concreto"
busca4 = "cimento"
busca5 = "brita de construcao"
busca6 = "prego"
busca7 = "areia construcao"
busca8 = "porta"
busca9 = "tabua"
busca10 = "vergalhao"
busca11 = "viga de aco"
busca12 = "tubos de aco"
busca13 = "trelica"


response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"viga construcao", "limit":100, "path":busca1}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

IMG_DIR = 'C:/Users/Felipe Pinto/Roaming/Documents/Python/APS_6/downloads/' + busca2

for img in os.listdir(IMG_DIR):
    img_array = cv2.imread(os.path.join(IMG_DIR, img), cv2.IMREAD_GRAYSCALE)

    img_pil = Image.fromarray(img_array)
    img_32x32 = np.array(img_pil.resize((32, 32), Image.ANTIALIAS))

    img_array = (img_32x32.flatten())

    img_array = img_array.reshape(-1, 1).T

    print(img_array)

    with open(busca2 + '.csv', 'ab') as f:

        np.savetxt(f, img_array, delimiter=",")