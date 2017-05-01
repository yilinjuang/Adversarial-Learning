#!/bin/bash

#MODE="--randforest"
MODE="--gradboost"
#MODE="--adaboost"
#MODE="--cnn"
MAX_DEP=3
N_EST=10

echo mode = $MODE, max_depth = $MAX_DEP, n_estimators = $N_EST
python mnist_blackbox.py $MODE -d $MAX_DEP -n $N_EST
#for MAX_DEP in {16..25}
#do
    #echo mode = $MODE, max_depth = $MAX_DEP, n_estimators = $N_EST
    #python mnist_blackbox.py $MODE -d $MAX_DEP -n $N_EST
    #for N_EST in {20..100..20}
    #do
        #echo mode = $MODE, max_depth = $MAX_DEP, n_estimators = $N_EST
        #python mnist_blackbox.py $MODE -d $MAX_DEP -n $N_EST
    #done
#done
