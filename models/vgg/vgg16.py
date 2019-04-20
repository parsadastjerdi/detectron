from __future__ import absolute_import

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


'''
References:
    Very-Deep Convolutional Networks
'''

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array

class VGG16:
    '''
    Builds the VGG16 CNN architecture and trains as well
    '''
    def __init__(self, nb_channels, num_input_channels):
        '''
            Create model
        '''
        model = Sequential()
        model.add(ZeroPadding2D((1, 1),input_shape=(3, 224, 224)))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(MaxPool2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, 3, 3, activation='relu'))
        model.add(MaxPool2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(MaxPool2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(MaxPool2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(MaxPool2D((2, 2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        return model  

    def train(self):
        pass      

    def predict(self, image):
        '''
            Returns a set of labels
        '''
        image = load_img('images/coffee_mug.jpg', target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # image = preprocess_input(image)
        y_hat = model.predict(image)
        # labels = decode_predictions(y_hat)
        # return labels


if __name__ == '__main__':
    model = VGG16()
    model.predict()
