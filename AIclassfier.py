import tensorflow as tf
import numpy as np
class classifier:
    def __init__(self):
        self.labels = {0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash'}
        self.img_shape = 299
        net = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling = 'avg')
        num_class = 6
        self.model = tf.keras.Sequential([net,
                    tf.keras.layers.Dense(num_class, activation='softmax')])
        self.model.layers[0].trainable = False
        self.model.load_weights('C:/Users/Nikesh/PycharmProjects/Recycler/recycler2.h5')
        self.data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
        print(self.model.summary())
    def classify(self, image_array):
        image_array = self.data_generator.standardize(image_array)
        result = self.model.predict(image_array)
        result = np.argmax(result, axis=1)
        print(self.labels[result[0]])
        return self.labels[result[0]]
