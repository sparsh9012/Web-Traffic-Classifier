import matplotlib.pyplot as plt

path = "./plots/"
name = "MLP_plot"
filename = "./results/MLP_acc.txt"

x = []
y = []

in_file = open(filename, 'r')

for line in in_file:
    col = line.split("|")
    x.append( int(float(col[0])) )
    y.append( float(col[1]) )

plt.title(name)

plot = plt.subplot()
plot.grid(True)
plot.set_xlabel("Cells per Layer")
plot.set_ylabel("Accuracy")
plot.plot(x, y)

plt.savefig(path+name)
plt.show()
