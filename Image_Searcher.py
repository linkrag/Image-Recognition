from google_images_download import google_images_download   #importing the library

import numpy as np
from PIL import Image
import cv2
import os

busca = ""



response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":busca, "limit":100, "path":busca, "format":"jpg", "format":"png"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

IMG_DIR = 'C:/Users/Felipe Pinto/Roaming/Documents/Python/APS_6/downloads/' + busca

# Create a csv doc with the images searched 
for img in os.listdir(IMG_DIR):
    img_array = cv2.imread(os.path.join(IMG_DIR, img), cv2.IMREAD_GRAYSCALE)

    img_pil = Image.fromarray(img_array)
    img_32x32 = np.array(img_pil.resize((32, 32), Image.ANTIALIAS))

    img_array = (img_32x32.flatten())

    img_array = img_array.reshape(-1, 1).T

    print(img_array)

    with open(busca2 + '.csv', 'ab') as f:

        np.savetxt(f, img_array, delimiter=",")
