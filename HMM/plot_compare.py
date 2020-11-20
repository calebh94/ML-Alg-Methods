import matplotlib.pyplot as plt
import numpy as np


ps7 = np.load("probs_0.7.npy")
ps9 = np.load("probs_0.9.npy")
wks = np.linspace(1,39,39)
# plotting the line 1 points
plt.plot(wks, ps7, label = "q=0.7")
# plotting the line 2 points
plt.plot(wks, ps9, label = "q=0.9")
plt.xlabel('weeks')
# Set the y axis label of the current axis.
plt.ylabel('P')
# Set a title of the current axes.
# plt.title('Two or more lines on same plot with suitable legends ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.savefig("0.7_0.9_compare.png")
plt.show()