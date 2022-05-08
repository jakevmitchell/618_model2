import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras.activations import relu
from traceloader import TraceConfig
import numpy
import logging
import pynvml
import os

os.chdir(r"C:\Users\Charg\OneDrive\Documents\618_proj\DL-LA-main\DL-LA-main")
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# Specify the raw data set to be analyzed (e.g.)
traceset = 'FPGA_PRESENT_RANDOMIZED_CLOCK'
traceconfig = TraceConfig()
tracelength = traceconfig.getnrpoints(traceset)
peakdist = traceconfig.getpeakdistance(traceset)


# Define the training and validation parameters
nrtrain = 5000
nrval = 10000
nrepochs = 50
batchsize = 2000
filter = 12
kernel_mult = 2
strides = 2
pool = 2
if nrtrain > 100000:
    nrsensi = 100000
else:
    nrsensi = nrtrain
balance = 1

train_x, train_y, val_x, val_y = traceconfig.prep_traces(traceset, nrtrain, nrval, balance)

model = tf.keras.models.load_model(r"CNNModel")
model.summary()
model.compile(optimizer="Adam",loss="mse")
print(model.evaluate(val_x,val_y))