from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model


from matplotlib.mlab import PCA

from data_creator import *
import sys
import numpy as np

np.random.seed(123)


if len(sys.argv)==1:
    (x, y, samples) = create_block_input(50000, 1, True, True, True, True, True)
elif len(sys.argv)==3:
    x_file = str(sys.argv[1])
    y_file = str(sys.argv[2])
    (samples, seq_length, inp) = read_from_file(x_file)
    (samples, t, output) = read_from_file(y_file)

    x = np.array(inp)
    y = np.array(output)
    
#find data scale and normalize data
scale,xmin = pre_normalization(x)
x = normalize(x, scale, xmin)
    
input_num = len(x[0])
X = np.reshape(x,(samples, 1, input_num))
Y = to_categorical(y, num_classes=5)

#data = x[:, np.apply_along_axis(np.count_nonzero, 0, x) > 0]
#results = PCA(data)
#x = results.Y



model = Sequential()
model.add(LSTM(10, input_shape=X.shape[1:], batch_input_shape=(1,1,input_num), stateful=True, return_sequences = True))
model.add(LSTM(10))
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()
plot_model(model, to_file="LSTM_50.png", show_shapes=True)

sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


model.fit(X, Y, validation_split=0.2, batch_size=1, epochs=10, verbose=1)
#print(X[1:6])
#print(model.predict(X[1:6], batch_size=1))



model.save("./models/model.h5")
filename = "./models/settings"
np.savez(filename, data_scale=scale, data_min=xmin)
#out_file = open(filename, 'w')
#out_file.write(str(scale) + "," + str(xmin)) #write scale,xmin

