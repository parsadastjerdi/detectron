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


# load VGG model without top classifier
# freeze all the layers (i.e. `trainable = False`)
# add some layers to the top
# compile and train the model on some data
# un-freeze some of the layers of VGG by setting `trainable = True`
# compile the model again  <-- DON'T FORGET THIS STEP!
# train the model on some data

from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

from keras.layers import Flatten, Dropout, Dense, Conv2D
from keras.models import Sequential, Model


class SSD300:
    def __init__(self):
        vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        vgg19.layers.pop()
        vgg19.layers.pop()
        vgg19 = Model(vgg19.input, vgg19.layers[-1].output)

        output = vgg19.output

        output = Conv2D(19, 19, )

        output = Flatten()(output)
        output = Dense(1024, activation='relu')(output)
        output = Dropout(0.5)(output)
        output = Dense(1024, activation='relu')(output)
        output = Dense(16, activation='softmax')(output)

        self.model = Model(input=vgg19.input, output=output)
        self.model.summary()
    
    def predict(self):
        pass

    
    def _load_weights(self):
        pass


if __name__ == '__main__':
    ssd = SSD300()