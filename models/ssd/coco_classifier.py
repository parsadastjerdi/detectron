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

from __future__ import absolute_import

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam

from imageio import imread
import numpy as np
import matplotlib

# Need this in order to matplotlib and tkinter at the same time
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss


'''
Description:
    This is a simple wrapper class for the ssd_keras ssd implementation using the Microsoft COCO dataset. 
    It is used to clean up some of
    the code inside the detectron.py file.
'''
class CocoClassifier:
    img_height = 300
    img_width = 300
    classes = ['background',
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, weights_path):
        K.clear_session() # Clear previous models from memory.

        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                        n_classes=20,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        # Load the trained weights into the model.
        self.model.load_weights(weights_path, by_name=True)

        # Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


    def predict(self, img):
        '''
        Predict the bounding boxes and class for a given image
        '''

        y_pred = self.model.predict(img)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

        return y_pred_thresh
    
    def get_boxes(self, orig_images, y_pred_thresh):
        '''
        Return a dictionary of the boxes
        '''

        boxes = []
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        for box in y_pred_thresh[0]:
            # Transform bounding boxes from 300x300 to original dimensions
            xmin = box[2] * orig_images[0].shape[1] / self.img_width
            ymin = box[3] * orig_images[0].shape[0] / self.img_height
            xmax = box[4] * orig_images[0].shape[1] / self.img_width
            ymax = box[5] * orig_images[0].shape[0] / self.img_height  

            # Set the color and label for the box
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])

            rectangle = [xmin, ymin, xmax, ymax]

            boxes.append((label, rectangle, color))
        
        return boxes


    def display(self, orig_images, y_pred_thresh):
        '''
        Display the image and draw the predicted boxes onto it
        '''

        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.figure(figsize=(20,12))
        plt.imshow(orig_images[0])

        current_axis = plt.gca()

        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * orig_images[0].shape[1] / self.img_width
            ymin = box[3] * orig_images[0].shape[0] / self.img_height
            xmax = box[4] * orig_images[0].shape[1] / self.img_width
            ymax = box[5] * orig_images[0].shape[0] / self.img_height

            # Set the color and label for the box
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])

            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

        plt.show()
    

    def load_image(self, img_path):
        '''
        Loads the original image and the transformed image
        '''
        orig_images = [] # Store the images here.
        input_images = [] # Store resized versions of the images here.

        # We'll only load one image in this example.
        img_path = 'examples/fish_bike.jpg'

        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img = image.img_to_array(img) 
        input_images.append(img)
        input_images = np.array(input_images)

        return (orig_images, input_images)



if __name__ == '__main__':
    classifier = PascalVOCClassifier(weights_path='../../weights/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5')
    orig_images, input_images = classifier.load_image('examples/fish_bike.jpg')
    y_pred_thresh = classifier.predict(input_images)
    classifier.display(orig_images, y_pred_thresh)