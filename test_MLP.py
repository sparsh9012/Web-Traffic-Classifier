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
    (x, y, samples) = create_block_input(50000, 100, False, True, False, True, True)
elif len(sys.argv)==3:
    x_file = str(sys.argv[1])
    y_file = str(sys.argv[2])
    (samples, seq_length, inp) = read_from_file(x_file)
    (samples, t, output) = read_from_file(y_file)

    x = np.array(inp)
    y = np.array(output)

#load data scale and normalize data
filename = "./models/MLP/all_attacks/all_attacks_settings.npz"
#in_file = open(filename, 'r')
#col = in_file.readline().split(",")
#scale = float(col[0]))
#xmin = float(col[1]))
settings = np.load(filename)
x = normalize(x, settings["data_scale"], settings["data_min"])
    
input_num = len(x[0])
X = np.reshape(x,(samples, input_num))
Y = to_categorical(y, num_classes=5)

#load model
model = load_model("./models/MLP/all_attacks/MLP_hidLayers=2_cellsPerLayer=20_Dataset=all_attacks_SGD=0,01.h5")

score = model.evaluate(X, Y, batch_size=1, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


d = model.predict_classes(X[0:15], batch_size = 1)


print(d)

