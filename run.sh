#!/bin/bash

#usage: mnist_blackbox.py [-h] {cnn,adaboost,gradboost,randforest,svm}
#usage: mnist_blackbox.py adaboost [-h] n_est {dt,svm} ...
#usage: mnist_blackbox.py gradboost [-h] max_dep n_est
#usage: mnist_blackbox.py randforest [-h] max_dep n_est
#usage: mnist_blackbox.py svm [-h]
                             #[--kernel {linear,rbf,poly,rbf,sigmoid,precomputed}]
                             #C

#MODE="svm" # {cnn,adaboost,gradboost,randforest,svm}
#N_EST=10
#T_EST="dt"
#MAX_DEP=3
#C=10

#echo mode = $MODE
#echo n_estimators = $N_EST
#echo max_depth = $MAX_DEP
#echo type_estimator = $T_EST
#echo C = $C

#python mnist_blackbox.py $MODE -d $MAX_DEP -n $N_EST
for times in {1..1}
do
    #echo mode = $MODE, max_depth = $MAX_DEP, n_estimators = $N_EST
    #python mnist_blackbox.py $MODE $C
    for MAX_DEP in {2..20..3}
    do
        #echo mode = $MODE, max_depth = $MAX_DEP, n_estimators = $N_EST
        #python mnist_blackbox.py $MODE -d $MAX_DEP -n $N_EST
        echo max_depth = $MAX_DEP
        python mnist_xgb.py $MAX_DEP
    done
done
