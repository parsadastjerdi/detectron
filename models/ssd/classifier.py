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

import tensorflow as tf 

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam

import numpy as np

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss


class PascalVOCClassifier:
    '''
    Description:
        This is a simple wrapper class for the ssd_keras ssd implementation using the Pascal VOC dataset. 
        It is used to clean up some of
        the code inside the detectron.py file.
    '''

    img_height = 300
    img_width = 300
    classes = ['background',
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

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

        # This is needed to deal with an error
        self.graph = tf.get_default_graph()
        self.model._make_predict_function()


    def predict(self, img):
        '''
        Predict the bounding boxes and class for a given image
        '''
        with self.graph.as_default():
            y_pred = self.model.predict(img)

        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        return y_pred_thresh


class CocoClassifier:
    pass


if __name__ == '__main__':
    classifier = PascalVOCClassifier(weights_path='weights/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5')
    orig_images, input_images = classifier.load_image('examples/fish_bike.jpg')
    y_pred_thresh = classifier.predict(input_images)
    classifier.display(orig_images, y_pred_thresh)