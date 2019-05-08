# Detectron
This project contains a simple GUI built with Tkinter and OpenCV to test the objection detection algorithms, specifically SSD. The model and training weights used are from ssd_keras, which is also on GitHub. 

## Dependencies
- Python 3.x
- Numpy
- TensorFlow 1.x
- Keras 2.x
- OpenCV
- Beautiful Soup 4.x
- Tkinter
- Matplotlib


## Usage
First, clone the repository
```
git clone https://github.com/parsadastjerdi/detectron
```

The PASCAL VOC 2007 weights will need to be downloaded from [here](https://drive.google.com/open?id=1vtNI6kSnv7fkozl7WxyhGyReB6JvDM41) and stored in ``` detectron/weights ```.
Make sure you're using ```VGG_VOC0712_SSD_300x300_ft_iter_120000.h5```. 
If the weights do not work, try some of the other PASCAL VOC weights listed in ```docs/README.md```.


Then run the application from the detectron root directory using:
```
python detectron.py
```
