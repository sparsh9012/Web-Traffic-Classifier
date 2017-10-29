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

#function that prints the percentage of samples that the network placed to each category
def print_accuracies(X):

    temp = model.predict_classes(X, batch_size = 1)
    #print(temp)

    samples = len(X)
    icmp_flood = 0
    tcp_syn_flood = 0
    udp_flood = 0
    port_scan = 0
    legit = 0
    for item in temp:
        if item==1:
            icmp_flood = icmp_flood + 1
        elif item==2:
            tcp_syn_flood = tcp_syn_flood + 1
        elif item==3:
            udp_flood = udp_flood + 1
        elif item==4:
            port_scan = port_scan + 1
        elif item==0:
            legit = legit + 1

    icmp_percentage = icmp_flood/samples
    tcp_percentage = tcp_syn_flood/samples
    udp_percentage = udp_flood/samples
    ps_percentage = port_scan/samples
    legit_percentage = legit/samples
    
    print('\n\n')
    print('Icmp_flood Tcp_syn_flood, Udp_flood Port_scan Legit\n')
    print(icmp_percentage, tcp_percentage, udp_percentage, ps_percentage, legit_percentage)


if len(sys.argv)==1:
    (x, y, samples) = create_block_input(50000, 1, True, True, True, True, True)
elif len(sys.argv)==3:
    x_file = str(sys.argv[1])
    y_file = str(sys.argv[2])
    (samples, seq_length, inp) = read_from_file(x_file)
    (samples, t, output) = read_from_file(y_file)

    x = np.array(inp)
    y = np.array(output)

#load data scale and normalize data
filename = "./models/MLP/all_attacks(progressive_aggregation)/AdaGrad=0,01/all_attacks(progressive_aggregation)_AdaGrad=0,01_Dropout=0_settings.npz"
settings = np.load(filename)
x = normalize(x, settings["data_scale"], settings["data_min"])

input_num = len(x[0])
X = np.reshape(x,(samples, input_num))
Y = to_categorical(y, num_classes=5)

#load model
model = load_model("./models/MLP/all_attacks(progressive_aggregation)/AdaGrad=0,01/MLP_hidLayers=4_cellsPerLayer=20_Dataset=all_attacks(progressive_aggregation)_AdaGrad=0,01_Dropout=0.h5")

#evaluate accuracy using commands provided by the keras library
score = model.evaluate(X[0:10000], Y[0:10000], batch_size=1, verbose=1)
print('\n\n')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print_accuracies(X[0:10000])
