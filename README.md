[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
![Azure ML](https://img.shields.io/badge/Microsoft%20Azure-Machine%20Learning-informational)

# Optimizing an ML Pipeline in Azure

## Overview
The main goal of this project was to build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn Logistic Regression model to solve a classification problem. Hyperdrive was used to optimize the model. This was then compared to an Azure AutoML run to see which of these approaches returns the best tuned model.

**Note**: this is a fork of the template project in Udacity containing starter files. The final project in the Udacity's Azure Machine Learning Engineer Nanodegree program follows a similar template as this.

## Summary
We used a provided Logistic Regression model and the data related with direct marketing campaigns of a Portuguese banking institution to predict whether or not the client will agree to a bank term deposit. The model was optimized with HyperDrive, reaching a accuracy rate of ~ 90%.

In the Azure AutoML, the best performing model was a VotingEnsemble with 91.69% of accuracy, higher than of the HyperDrive model.

## Scikit-learn Pipeline
* The preprocessing steps were already provided in the training script named *train.py*, which includes helper functions used in the Jupyter Notebook file named *udacity_project.ipynb*.
* The TabularDatasetFactory method was used to create a tabular dataset for Azure Machine Learning.
* SKLearn estimator was used to setup a training run.
* 'Accuracy' is the metric used for the random hyperparameter tuning with HyperDrive, and Bandit Policy was used for early termination.

**Note**: since the SKLearn constructor was recently deprecated, the respective code should be replaced by the following object:
```python
from azureml.core import ScriptRunConfig
est = ScriptRunConfig(source_directory='.',
                      script='train.py',
                      compute_target=cpu_cluster,
                      environment=sklearn_env)
```

The early termination policy had as parameters:
* The ratio used to calculate the allowed distance from the best performing experiment run (set to 0.1)
* The frequency for applying the policy (set to 1)

## AutoML
The AutoML was set as classification task with experiment timeout of 30 minutes due to limited instance time. VotingEnsemble was the model selected by the AutoML, with best observed accuracy_score of 91.69%.

  *Image 1 - AutoML summary*

![alt text](https://raw.githubusercontent.com/kauvinlucas/Optimizing-a-Pipeline-in-Azure/master/Screenshots/automl_summary.PNG "AutoML summary")

  *Image 2 - Accuracy by parameters*

![alt text](https://raw.githubusercontent.com/kauvinlucas/Optimizing-a-Pipeline-in-Azure/master/Screenshots/accuracy_by_params.PNG "AutoML accuracy by parameters")

The other metrics returned by the model best voted by AutoML were the following:

  *Image 3 - AutoML metrics*

![alt text](https://raw.githubusercontent.com/kauvinlucas/Optimizing-a-Pipeline-in-Azure/master/Screenshots/metrics.PNG "AutoML metrics")

Of course, the added benefit of the AutoML was the explainability  with *model_explainability* parameter set to True by default:

  Image 4 - *Feature importance graph*
  
![alt text](https://raw.githubusercontent.com/kauvinlucas/Optimizing-a-Pipeline-in-Azure/master/Screenshots/feature_importance.PNG "AutoML feature importance graph")

It is important to acknowledge that the Azure AutoML handled the regularization and hyperparameter optimization, model complexity limitations and cross-validation practices by itself in the imbalanced classifaction problem in order to prevent over-fitting, so this may explain the performance advantage over the model tuned by HyperDrive.

## Sources:
* Train scikit-learn models at scale with Azure Machine Learning: [link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-scikit-learn?view=azure-ml-py)
* Prevent overfitting and imbalanced data with automated machine learning: [link](https://docs.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls)
* Udacity program: [link](https://www.udacity.com/course/machine-learning-engineer-for-microsoft-azure-nanodegree--nd00333)
