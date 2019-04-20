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
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# from keras.applications.vgg19 import preprocess_input

from keras.preprocessing.image import img_to_array
from classifier import Classifier


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
            # while not self.exit:
            while True:
                _, self.frame = self.stream.read()
                # self.predict()

                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, 1)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)

                # self.frame = img
                # self.predict()

                if self.panel is None:
                    self.panel = Label(image=img)
                    self.panel.image = img
                    self.panel.pack(side='top', fill='both', expand='yes')
                else:
                    self.panel.configure(image=img)
                    self.panel.image = img
                    # self.predict()

        except RuntimeError:
            print('Caught a RuntimeError')


    def predict(self):
        '''
            Make a prediction based on the polled frame
        '''
        orig_images = [] 
        input_images = []
        img = self.frame.copy()

        orig_images.append(img)

        # copy current frame and transform into tensor of shape (300, 300, 3)
        image = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

        image = img_to_array(image)
        input_images.append(image)
        input_images = np.array(input_images)

        y_pred_thresh = self.model.predict(input_images)
        self.draw_rectangles(orig_images, y_pred_thresh)
    

    def draw_rectangles(self, orig_images, y_pred_thresh):
        '''
        Display the image and draw the predicted boxes onto it
        '''

        for label, box, color in self.model.get_boxes(orig_images, y_pred_thresh):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])

            cv2.rectangle(img=orig_images[0], pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img=orig_images[0], text=label, org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), lineType=cv2.LINE_AA) 
        
        # cv2.imwrite('images/' + label.split(':')[0] + '.jpg', orig_images[0])
        # self.panel = orig_images[0]
        cv2.imshow("Prediction", orig_images[0])
    

    def screenshot(self):
        '''
        Take a screenshot of the image and save it under the screenshots folder
        '''
        pass
            

    def quit(self):
        print('Exiting application.')
        self.master.destroy()

        '''
        self.exit = True
        self.stream.release()
        cv2.destroyAllWindows()
        '''


if __name__ == '__main__':
    model = Classifier(weights_path='../../weights/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5')

    root = Tk()
    stream=cv2.VideoCapture(0)

    app = Detectron(model=model, master=root, stream=stream)
    app.master.mainloop()