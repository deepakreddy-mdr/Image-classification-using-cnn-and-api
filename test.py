import tensorflow as tf
import numpy as np
import cv2
import glob

class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

model = tf.keras.models.load_model("model.h5")

# find any image automatically
image_files = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.jpeg")

if len(image_files) == 0:
    print("No image found in folder")
    exit()

image_path = image_files[0]
print("Using image:", image_path)

img = cv2.imread(image_path)

img = cv2.resize(img, (32,32))
img = img / 255.0
img = np.reshape(img, (1,32,32,3))

prediction = model.predict(img)
confidence = np.max(prediction) * 100
predicted_class = class_names[np.argmax(prediction)]

print("Predicted Class:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))
