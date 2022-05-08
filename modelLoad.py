import tensorflow as tf

model_path = "CNNModel/saved_model.pb"
model = tf.saved_model.load(model_path)