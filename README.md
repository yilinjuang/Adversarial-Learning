# Adversarial-Learning
Individual Study under Prof. S. D. Lin.

## Requirements
- [openai/cleverhans](https://github.com/openai/cleverhans)
- keras
- matplotlib
- scikit-learn
- tensorflow

## Black-box Attack
Forked from [mnist_tutorial_tf](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_tf.md). Use FGSM attack against ensemble methods including
    - [Adaptive Boosting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
    - [Gradient Boosting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)
    - [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)

### Result
[Raw Data](https://docs.google.com/spreadsheets/d/1JOjMBLfvOUO2KTs3KoKVAsflTY2UgGV9KIjrm5FAA5Q/pubhtml)
<img src="https://github.com/frankyjuang/Adversarial-Learning/blob/master/chart/adaboost_acc_dep.png?raw=true" alt="Adaptive Boosting Accuracy-Max_Depth">
<img src="https://github.com/frankyjuang/Adversarial-Learning/blob/master/chart/adaboost_acc_est.png?raw=true" alt="Adaptive Boosting Accuracy-N_Estimators">
<img src="https://github.com/frankyjuang/Adversarial-Learning/blob/master/chart/gradboost_acc_dep.png?raw=true" alt="Gradient Boosting Accuracy-Max_Depth">
<img src="https://github.com/frankyjuang/Adversarial-Learning/blob/master/chart/gradboost_acc_est.png?raw=true" alt="Gradient oosting Accuracy-N_Estimators">
<img src="https://github.com/frankyjuang/Adversarial-Learning/blob/master/chart/randforest_acc_dep.png?raw=true" alt="Random Forest Accuracy-Max_Depth">
<img src="https://github.com/frankyjuang/Adversarial-Learning/blob/master/chart/randforest_acc_est.png?raw=true" alt="Random Forest Accuracy-N_Estimators">

