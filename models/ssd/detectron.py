'''
Copyright (C) 2019 Parsa Dastjerdi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''
from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading, datetime, os
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from classifier import PascalVOCClassifier


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

        self.screenshot_button = Button(self.master, text='Screenshot',
                                fg='green',
                                command=self.screenshot)
        self.screenshot_button.pack(side='left', fill='both', expand='yes', padx=10, pady=10)

        self.quit_button = Button(self.master, text='Quit', 
                                fg='red',
                                command=self.quit)
        self.quit_button.pack(side='right', fill='both', expand='yes', padx=10, pady=10)

    
    def poll(self):
        '''
            Contiuously poll the webcam for new frames.
            This is handled on a separate thread from Tkinter's main loop. 
        '''

        try:
            # while not self.exit:
            while True:
                _, self.frame = self.stream.read()
                self.predict()

                # Tranform image so that it can be displayed within the GUI
                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                # img = cv2.flip(img, 1)
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
        Args:
        Returns: 
            Returns the new frame with the predicted bounding boxes on the image itself.
        '''

        input_images = []

        img = self.frame.copy()

        # Copy current frame and transform into tensor of shape (300, 300, 3)
        image = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        input_images.append(image)
        input_images = np.array(input_images)

        # Get predicted class and bounding boxes for all objects within frame
        y_pred_thresh = self.model.predict(input_images)

        # Return image with boxes/labels drawn on
        self.draw_boxes(img, y_pred_thresh)
    

    def draw_boxes(self, img, y_pred_thresh):
        '''
        Take as input the original image and the predictions
        Args:
            orig_images:
            y_pred_thresh:
        Returns:
            A single image that contains the bounding boxes/classes for each object
        '''

        for box in y_pred_thresh[0]:
            # Transform bounding boxes from 300x300 to original dimensions
            xmin = int(box[2] * img.shape[1] / self.model.img_width)
            ymin = int(box[3] * img.shape[0] / self.model.img_height)
            xmax = int(box[4] * img.shape[1] / self.model.img_width)
            ymax = int(box[5] * img.shape[0] / self.model.img_height)

            color = self.model.colors[int(box[0])]
            label = '{}: {:.2f}'.format(self.model.classes[int(box[0])], box[1])

            cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img=img, text=label, org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), lineType=cv2.LINE_AA) 

        self.frame = img
    

    def screenshot(self):
        '''
        Take a screenshot of the image and save in the images folder
        '''      
        ts = datetime.now()
        filename = "images/{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        cv2.imwrite(filename, self.frame.copy())
            

    def quit(self):
        print('Exiting application.')
        self.master.destroy()

        '''
        self.exit = True
        self.stream.release()
        cv2.destroyAllWindows()
        '''


if __name__ == '__main__':
    model = PascalVOCClassifier(weights_path='weights/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5')
    # model = CocoClassifier(weights_path='weights/VGG_coco_SSD_300x300_iter_400000.h5')

    root = Tk()
    stream=cv2.VideoCapture(0)

    app = Detectron(model=model, master=root, stream=stream)
    app.master.mainloop()