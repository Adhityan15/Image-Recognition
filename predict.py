import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model/cnn_model.h5")
classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def predict_image(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    return classes[np.argmax(prediction)]
