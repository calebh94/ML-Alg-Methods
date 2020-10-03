import matplotlib.pyplot as plt
import os

result_files = [ff for ff in os.listdir("results") if ff.endswith(".txt")]

static_results = [0] * len(result_files)
dynamic_results = [0] * len(result_files)
cases = []

file_cnt = 0
for file in result_files:
    case = file.split("_")[0]
    cases.append(case)

    with open(os.path.join("results", file), 'r') as f:
        firstline = f.readline().split()
        # static_results[case] = [float(firstline[1])]
        static_results[file_cnt] = float(firstline[1])
        # dynamic_results[case] = []
        dynamic_results[file_cnt] = float(0.00)
        cnts = 0
        for line in f:
            line = line.split()
            time = line[1]
            if float(time) >= 0.00:
                dynamic_results[file_cnt] = dynamic_results[file_cnt] + float(line[1])
                # dynamic_results[case].append([line[1]])
                cnts = cnts + 1
        dynamic_results[file_cnt] = dynamic_results[file_cnt] * (cnts / 1000)  # scaling cases whennot all time recorded properly
    file_cnt = file_cnt+1

# known data
# cases = ["rmat0406", "rmat0507", "rmat0608", "rmat0709", "rmat0810",
#          "rmat0911", "rmat1012", "rmat1113", "rmat1214", "rmat1315",
#          "rmat1416", "rmat1517", "rmat1618"]

edges = [64, 125, 259, 530, 1054, 2183, 4111, 9875, 29779, 106361, 372670, 1770677, 7926934]
edges_lab = [str(j) for j in edges]
x_pos = [i for i, _ in enumerate(cases)]
f1 = plt.figure()

plt.bar(x_pos, static_results, color='green')
plt.xlabel("Edge Counts / Labels")
plt.ylabel("Runtime (milliseconds)")
plt.title("Static Results (Prim's Algorithm)")
labels = []
for i in range(0, len(edges_lab)):
    labels.append(cases[i] + ": \n" + edges_lab[i])
plt.xticks(x_pos, labels)

import math
logedges = [math.log(val, 10) for val in edges]
logresults = [math.log(val, 10) for val in static_results]

f2 = plt.figure()
plt.plot(logedges, logresults)
plt.xlabel("Edge Counts")
plt.ylabel("Runtime (milliseconds)")
plt.title("Static Results (Prim's Algorithm) Log-Log")
plt.xscale("linear")
plt.yscale("linear")

f3 = plt.figure()

plt.bar(x_pos, dynamic_results, color='blue')
plt.xlabel("Edge Counts / Labels")
plt.ylabel("Runtime (milliseconds)")
plt.title("Dynamic Results (Cycle Property Recomputation)")
labels = []
for i in range(0, len(edges_lab)):
    labels.append(cases[i] + ": \n" + edges_lab[i])
plt.xticks(x_pos, labels)

import math
logedges = [math.log(val, 10) for val in edges]
logresults = [math.log(val, 10) for val in dynamic_results]

f4 = plt.figure()
plt.plot(logedges, logresults)
plt.xlabel("Edge Counts")
plt.ylabel("Runtime (milliseconds)")
plt.title("Dynamic Results (Cycle Property Recomputation) Log-Log")
plt.xscale("linear")
plt.yscale("linear")

plt.show()
