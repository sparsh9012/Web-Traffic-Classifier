from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.callbacks import History

from matplotlib.mlab import PCA

from data_creator import *
import sys
import numpy as np

np.random.seed(123)


def write_results(filename, cells_per_layer, acc):
    out_file = open(filename, 'a')

    out_file.write(cells_per_layer+"|"+acc)
    out_file.write("\n")
    out_file.close()

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
X = np.reshape(x,(samples, input_num))
Y = to_categorical(y, num_classes=5)

# code used for PCA analysis
#data = x[:, np.apply_along_axis(np.count_nonzero, 0, x) > 0]
#results = PCA(data)
#x = results.Y

# create MLP model
cells_per_layer = 30

model = Sequential()
model.add(Dense(cells_per_layer, input_shape=X.shape[1:], activation='relu'))
#model.add(Dropout)
model.add(Dense(5, activation='softmax'))

# set model name and plot model
name = "MLP"
model.summary()
plot_model(model, to_file=name + ".png", show_shapes=True)

# set training method
sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# create callback that saves metrics during training
history = History()

# train MlP model
hist = model.fit(X, Y, validation_split=0.2, batch_size=1, epochs=10, callbacks=[history])

# print results
#print(hist.history['acc'])
#print(X[1:6])
#print(model.predict(X[1:6], batch_size=1))

# save model and settings
model.save("./models/" + name + ".h5")
filename = "./models/" + name + "_settings"
np.savez(filename, data_scale=scale, data_min=xmin)

# save results
write_results("./results/MLP_acc.txt", str(cells_per_layer), str(hist.history['val_acc'][-1]))
