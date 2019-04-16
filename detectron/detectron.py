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

from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading, datetime, os

import keras
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array


class Detectron(Frame):
    def __init__(self, model, stream=None, master=None, **kwargs):
        super().__init__(master)
        self.master = master
        self.stream = stream
        self.model = model

        # variable to stop polling once exit is pressed
        self.exit = False

        # panel is the window where each frame will be displayed
        self.panel = None
        self.frame = None

        # setup widgets 
        self.setup()
        print('Camera setup')

        # run the video on a separate thread
        self.thread = threading.Thread(target=self.poll, args=())
        self.thread.start()
        print('Thread started.')

        # should this be above the previous statement?
        self.master.wm_title('Detectron')
        self.master.wm_protocol("WM_DELETE_WINDOW", self.quit)

        # not sure if this necessary
        self.pack()

    
    def setup(self):
        '''
            Setup the GUI
        '''

        # create panel for webcam
        self.panel = Label()
        self.panel.pack(side='top', fill='both', expand='yes')

        self.screenshot_button = Button(self.master, text='Predict',
                                fg='green',
                                command=self.predict)
        self.screenshot_button.pack(side='left', fill='both', expand='yes', padx=10, pady=10)

        self.quit_button = Button(self.master, text='Quit', 
                                fg='red',
                                command=self.quit)
        self.quit_button.pack(side='right', fill='both', expand='yes', padx=10, pady=10)

    
    def poll(self):
        '''
            Contiuously poll the webcam for new frames
        '''

        try:
            while not self.exit:
                _, self.frame = self.stream.read()

                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, 1)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)

                if self.panel is None:
                    self.panel = Label(image=img)
                    self.panel.image = img
                    self.panel.pack(side='top', fill='both', expand='yes')
                else:
                    self.panel.configure(image=img)
                    self.panel.image = img

        except RuntimeError:
            print('Caught a RuntimeError')



    def predict(self):
        '''
            Make a prediction based on the polled frame
        '''

        # copy current frame and transform into tensor of shape (224, 224, 3)
        image = cv2.resize(self.frame.copy(), (224, 224), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)

        # input array will need to be 4-dimensional: samples, rows, columns, and channels.
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # make prediction and decode into class labels
        y_hat = self.model.predict(image)
        label = decode_predictions(y_hat)

        # retreive the most likely result (highest probability)
        label = label[0][0]

        prediction = '%s (%.2f%%)' % (label[1], label[2]*100)
        print(prediction)

        cv2.imwrite('images/' + label[1] + '.jpg', image)


    def quit(self):
        print('Exiting application.')
        self.exit = True
        self.stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model = VGG19(weights='imagenet')
    root = Tk()

    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    app = Detectron(model=model, master=root, stream=webcam)
    app.master.mainloop()