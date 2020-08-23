import numpy as np
import matplotlib.pyplot as plt

file = open("log.txt", 'r')

lambdas = []
accuracy = []

for line in file:
    line = line.strip()
    if 'Lambda: ' in line:
        lambdas.append(float(line[8:]))
    if 'Accuracy of the network on the 10000 test images:' in line:
        accuracy.append(float(line[-9:-2]))

file.close()
lambdas = np.asarray(lambdas)
accuracy = np.asarray(accuracy)

plt.figure()
plt.plot(lambdas, accuracy)
plt.title("Accuracy vs. Lambdas")
plt.xlabel("Lambdas")
plt.ylabel("Accuracy")
for i in range(len(lambdas)):
    plt.text(lambdas[i], accuracy[i], 'Lambda %f Accuracy %f' % (lambdas[i], accuracy[i]))

plt.show()