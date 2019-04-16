from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K 
from keras.models import load_model
from math import ceil
import numpy as np 
from matplotlib import pyplot as plt 


