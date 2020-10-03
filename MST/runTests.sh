#!/bin/bash


graphFiles=`ls ./data/ | grep .gr`

for graph in ${graphFiles}
do
	filename=`echo ${graph} | cut -d'.' -f1`
	echo ${graph} ${filename}
	C:/Users/harri/Anaconda3/envs/py36/python.exe ./src/run_experiments.py ./data/${graph} ./data/${filename}.extra ./results/${filename}_output.txt

done
