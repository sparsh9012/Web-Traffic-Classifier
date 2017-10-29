from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, RMSprop
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.callbacks import History

from matplotlib.mlab import PCA

from data_creator import *
import sys
import numpy as np

np.random.seed(123)


def write_results(filename, hidden_layers, cells_per_layer, val_acc, val_loss, acc, loss):
    out_file = open(filename, 'a')

    out_file.write(hidden_layers+"|"+cells_per_layer+"|"+val_acc+"|"+acc+"|"+val_loss+"|"+loss)
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
    
# find and save data scale and normalize data
scale,xmin = pre_normalization(x)
x = normalize(x, scale, xmin)

dataset_name = "all_attacks(aggregation)"
#train_method = "SGD=0,01"
train_method = "AdaGrad=0,01"
#train_method = "RMSProp=0,001"
dropout = "Dropout=0"
models_path = "./models/MLP/"+ dataset_name + "/" + train_method + "/"
filename = models_path + dataset_name +"_"+train_method+"_"+ dropout +"_settings"
#np.savez(filename, data_scale=scale, data_min=xmin)

input_num = len(x[0])
X = np.reshape(x,(samples, input_num))
Y = to_categorical(y, num_classes=5)

# code used for PCA analysis
#data = x[:, np.apply_along_axis(np.count_nonzero, 0, x) > 0]
#results = PCA(data)
#x = results.Y

# create MLP model
cells_per_layer_list = [5, 10, 20, 30]
hidden_layers_list = [1, 2, 3, 4]
batch_len = 32
drop_rate = 0

for hidden_layers in hidden_layers_list:
    for cells_per_layer in cells_per_layer_list:
        # create sequential model and add input and first hidden layer
        model = Sequential()
        model.add(Dense(cells_per_layer, input_shape=X.shape[1:], activation='relu'))
        model.add(Dropout(drop_rate))

        # add extra hidden layers
        for i in range(hidden_layers-1):
            model.add(Dense(cells_per_layer, activation='relu'))
            model.add(Dropout(drop_rate))

        # add output layer
        model.add(Dense(5, activation='softmax'))

        # set model name and plot model
        name = "MLP" + "_hidLayers=" + str(hidden_layers) + "_cellsPerLayer=" + str(cells_per_layer) + "_Dataset=" + dataset_name
        model.summary()
        ####plot_model(model, to_file=name + ".png", show_shapes=True)
    
        # set training method
        sgd = SGD(lr=0.001)
        adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])

        # train MlP model
        hist = model.fit(X, Y, validation_split=0.2, batch_size=batch_len, epochs=10, callbacks=[History()], shuffle=False)

        # print results
        #print(hist.history['acc'])
        #print(X[1:6])
        #print(model.predict(X[1:6], batch_size=1))

        # save model
        model.save(models_path + name + "_" + train_method + "_" + dropout + ".h5")
    
        # save results
        results_file = "./results/MLP, " + dataset_name + ", " + train_method + ", " + dropout + ".txt"
        write_results(results_file, str(hidden_layers), str(cells_per_layer), str(hist.history['val_acc'][-1]), str(hist.history['acc'][-1]), str(hist.history['val_loss'][-1]), str(hist.history['loss'][-1]))
