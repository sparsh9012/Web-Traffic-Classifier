from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras import initializers

from keras.optimizers import SGD, RMSprop
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
    (x, y, samples) = create_input_rand_pattern(10000, True, True, True, True, True)
elif len(sys.argv)==3:
    x_file = str(sys.argv[1])
    y_file = str(sys.argv[2])
    (samples, seq_length, inp) = read_from_file(x_file)
    (samples, t, output) = read_from_file(y_file)

    x = np.array(inp)
    y = np.array(output)
    
#find and save data scale and normalize data
scale,xmin = pre_normalization(x)
x = normalize(x, scale, xmin)

dataset_name = "all_attacks_rand_pattern"
models_path = "./models/RNN/" #### + dataset_name + "/"
filename = models_path + dataset_name + "_settings"
np.savez(filename, data_scale=scale, data_min=xmin)

input_num = len(x[0])
X = np.reshape(x,(samples, 1, input_num))
Y = to_categorical(y, num_classes=5)

# code used for PCA analysis
#data = x[:, np.apply_along_axis(np.count_nonzero, 0, x) > 0]
#results = PCA(data)
#x = results.Y

# create RNN model
cells_per_layer_list = [20]
hidden_layers_list = [2]
batch_len = 1
epoch_num = 10

for hidden_layers in hidden_layers_list:
    for cells_per_layer in cells_per_layer_list:
        # create sequential model and add input and first hidden layer
        model = Sequential()

        if(hidden_layers!=1):
            model.add(SimpleRNN(cells_per_layer, input_shape=(1,input_num), batch_input_shape=(batch_len,1,input_num), recurrent_initializer='random_uniform', kernel_initializer='random_uniform', activation='tanh', return_sequences=True, stateful=True))
        else:
            model.add(SimpleRNN(cells_per_layer, input_shape=(1,input_num), batch_input_shape=(batch_len,1,input_num), recurrent_initializer='random_uniform', kernel_initializer='random_uniform', activation='tanh', stateful=True))

        # add extra hidden layers
        for i in range(hidden_layers - 1):
            if(i!=hidden_layers-2): # add the rest hidden layers except the last
                model.add(SimpleRNN(cells_per_layer, recurrent_initializer='random_uniform', kernel_initializer='random_uniform', activation='tanh', return_sequences=True, stateful=True))
            else: # add the last hidden layer
                model.add(SimpleRNN(cells_per_layer, recurrent_initializer='random_uniform', kernel_initializer='random_uniform', activation='tanh', stateful=True))


        # add output layer
        model.add(Dense(5, kernel_initializer='uniform', activation = 'softmax'))

        # set model name and plot model
        name = "RNN" + "_hidLayers=" + str(hidden_layers) + "_cellsPerLayer=" + str(cells_per_layer) + "_Dataset=" + dataset_name
        model.summary()
        ####plot_model(model, to_file=name + ".png", show_shapes=True)
        
        # set training method
        train_method = "SGD=0,01"
        sgd = SGD(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # train RNN model
        #hist = model.fit(X, Y, validation_split=0.2, batch_size=batch_len, epochs=epoch_num, verbose=1, callbacks=[History()], shuffle=False)

        for i in range(1, epoch_num+1):
            hist = model.fit(X, Y, validation_split=0.2, batch_size=batch_len, epochs=1, verbose=1, callbacks=[History()], shuffle=False)
            model.reset_states()

        # print results
        #print(hist.history['acc'])
        #print(X[1:6])
        #print(model.predict(X[1:6], batch_size=1))

        # save model
        model.save(models_path + name + "_" + train_method + ".h5")
    
        # save results
        results_file = "./results/RNN, " + dataset_name + ", " + train_method + ".txt"
        write_results(results_file, str(hidden_layers), str(cells_per_layer), str(hist.history['val_acc'][-1]), str(hist.history['acc'][-1]), str(hist.history['val_loss'][-1]), str(hist.history['loss'][-1]))
