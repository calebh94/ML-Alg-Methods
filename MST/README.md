# CSE 6740 - CSE Algorithms - Homework 2 (MST)
This is the code documentation for Caleb Harris's Homework 2, Problem 3.

## Overview
The Minimum Spanning Tree is found using Prim's Algorithm and the recomputation is done by applying the Cycle Property. 

Key Functions:
* *parse_Edges*
* *computeMST*
    * *MST.insert_edge*
    * *PriorityQueue.put*
* *recomputeMST*
    * *removecycle*
    * *removecycle_recursive*

## Requirements
* Windows 10
* Python 3.5
    * Numpy - For easier/faster array formation and indexing
    
Testing was completed in a Pycharm development environment, but is not necessary.  The code should be compatible with Python 2 and Linux systems, but they have not been tested.

## Running a case
```
python.exe src\run_experiments.py data/rmat0406.gr data/rmat0406.extra results/rmat0406_MST.out
```
## Running the batch
In a Git bash command window (or linux):
```
$ ./runTests.sh
```

## Running the analysis
python.exe src\run_analysis.py