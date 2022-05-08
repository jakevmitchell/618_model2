import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras.activations import relu
from traceloader import TraceConfig
import numpy
import logging
import pynvml
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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


# Call the trace preparation function to create the required training and validation set
train_x, train_y, val_x, val_y = traceconfig.prep_traces(traceset, nrtrain, nrval, balance)

# Define the model, train the model and validate the model using the created data sets
model = Sequential(
    [Reshape((tracelength,1), input_shape = (tracelength,)),
    Conv1D(filters=filter, kernel_size=kernel_mult*peakdist, strides=peakdist//strides, input_shape=(tracelength,1), activation='relu'),
    MaxPooling1D(pool_size=pool),
    Flatten(),
    Dense(2, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
out=model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=nrepochs, batch_size=batchsize, verbose=1)


# Dump the validation accuracy into an ASCII file
logging.basicConfig(filename='val_acc.log', format='%(message)s', level=logging.INFO)
logging.info("val_accuracy: {}".format(out.history['val_accuracy']))


# Perform sensitivity analysis based on the training set
inp = tf.Variable(train_x[:nrsensi], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(inp)
    preds = model(inp)
    trues = tf.Variable(train_y[:nrsensi], dtype=tf.float32)
    loss = tf.keras.losses.binary_crossentropy(trues, preds)
    grads = tape.gradient(loss, inp)
grads_sum = numpy.sum(numpy.abs(grads), axis=0)


# Dump the sensitivity analysis results into a binary file
f = open("sensi.dat","wb")
f.write(grads_sum.astype("double"))
f.close()

measure = pynvml.nvmlDeviceGetPowerUsage(handle)
print(str(measure/1e3) + "W")

model.save("CNNModel",include_optimizer=False)
converter = tf.lite.TFLiteConverter.from_saved_model("CNNModel")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)