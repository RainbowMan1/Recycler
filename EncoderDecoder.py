import base64
from PIL import Image
from io import BytesIO
import numpy as np
import keras

class Decoder:
    @classmethod
    def decode(self,base64image):
        im = Image.open(BytesIO(base64.b64decode(base64image)))
        im.save('C:/Users/Nikesh/PycharmProjects/Recycler/Images/image.png', 'PNG')
        im = keras.preprocessing.image.load_img('C:/Users/Nikesh/PycharmProjects/Recycler/Images/image.png', target_size=(299, 299))
        img = (keras.preprocessing.image.img_to_array(im))
        img = img[np.newaxis, :]
        print(img[0].shape)
        return img
