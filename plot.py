import matplotlib.pyplot as plt

def plot_results(name):
    path = "./plots/"
    filename = "./results/" + name + ".txt"

    # set plot characteristics
    plt.title(name)
    myplot = plt.subplot()
    myplot.grid(True)
    myplot.set_xlabel("Cells per Layer")
    myplot.set_ylabel("Accuracy")

    # initialize counters for each line
    i = 0
    cells = []
    val_acc = []
    
    markers = ['s', 'o', '*', '.'] # marker shapes for points of different lines  
    m = 0 # marker counter

    # read results from file and plot all the lines
    in_file = open(filename, 'r')
    for line in in_file:
        col = line.split("|")
        hid_layer = int(float(col[0]))
        cells.append( int(float(col[1])) )
        val_acc.append( float(col[2]) )
        i = i+1
        if(i == 4):
            # plot current line
            myplot.plot(cells, val_acc, marker=markers[m], alpha=1, label='hidden_layers='+str(hid_layer))
            # reset counters for next line
            i = 0
            cells = []
            val_acc = []
            m = m + 1

    # move and resize the plot box in order to fit the legend box below the plot box
    box = myplot.get_position()
    myplot.set_position([box.x0*1.0, box.y0 + box.height*0.2, box.width * 1, box.height * 0.8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    plt.savefig(path+name)
    plt.show()
    plt.clf()


#plot_results("MLP, all_attacks, SGD=0,01")
#plot_results("MLP, icmp&legit, SGD=0,01")
#plot_results("MLP, tcp&legit, SGD=0,01")
#plot_results("MLP, udp&legit, SGD=0,01")
#plot_results("MLP, ps&legit, SGD=0,01")
#plot_results("MLP, tcp&ps&legit, SGD=0,01")
#plot_results("RNN, all_attacks, SGD=0,01")
#plot_results("MLP, all_attacks(aggregation), SGD=0,01")
#plot_results("MLP, tcp&ps&legit(aggregation), SGD=0,01")
#plot_results("LSTM, all_attacks_rand_pattern, SGD=0,01")
#plot_results("MLP, all_attacks(aggregation), SGD=0,01, Dropout=0,2")
#plot_results("MLP, all_attacks(aggregation), SGD=0,01, Dropout=0,4")
#plot_results("MLP, all_attacks(aggregation), AdaGrad=0,01, Dropout=0")
#plot_results("MLP, all_attacks(aggregation), AdaGrad=0,01, Dropout=0,2")
#plot_results("MLP, all_attacks(aggregation), AdaGrad=0,01, Dropout=0,4")
#plot_results("MLP, all_attacks(aggregation), RMSProp=0,001, Dropout=0")
#plot_results("MLP, all_attacks(aggregation), RMSProp=0,001, Dropout=0,2")
#plot_results("MLP, all_attacks(aggregation), RMSProp=0,001, Dropout=0,4")
#plot_results("MLP, all_attacks(progressive_aggregation), AdaGrad=0,01, Dropout=0")
