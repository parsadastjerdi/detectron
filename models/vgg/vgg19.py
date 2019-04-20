'''
MIT License

Copyright (c) 2018 Parsa Dastjerdi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import keras
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

model = VGG19()
model.summary()

image = load_img('images/coffee_mug.jpg', target_size=(224, 224))
image = img_to_array(image)

# input array will need to be 4-dimensional: samples, rows, columns, and channels.
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# preprocess input for VGG19
image = preprocess_input(image)

# make prediction
y_hat = model.predict(image)

# decode probabilities into class labels
label = decode_predictions(y_hat)

print(label)

# retreive the most likely result (highest probability)
label = label[0][0]

prediction = '%s (%.2f%%)' % (label[1], label[2]*100)
print(prediction)

