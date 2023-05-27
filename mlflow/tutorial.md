# MLFLOW
## Description
MLflow is a versatile, expandable, open-source platform for managing workflows and artifacts across 
the machine learning lifecycle. 

## Components of MLFlow
1. MLFlow tracking : An API for logging parameters, code versions, metrics, model environment dependencies, 
and model artifacts when running your machine learning code.
2. MLFlow Models: A model packaging format and suite of tools that let you easily deploy a trained model 
(from any ML library) for batch or real-time inference on platforms such as Docker, Apache Spark, 
Databricks, Azure ML and AWS SageMaker. 
3. MLFlow Model Registry:A centralized model store, set of APIs, and UI focused on the approval, quality assurance, 
and deployment of an MLFlow Model.
4. MLFlow Projects: A standard format for packaging reusable data science code that can be run with different 
parameters to train models, visualize data, or perform any other data science task.
5. MLFlow Recipes: Predefined templates for developing high-quality models for a variety of common tasks, 
including classification and regression.

### Add MLFlow Tracking code 
Using the supported library automatically logs the parameters, metrics and artifacts 
used for the run.

For libraries the at auto log is not yet supported use the following.
**Parameters**:
Constant values (for instance, configuration parameters)
```mlflow.log_param```, ```mlflow.log_params```

**Metrics**: 
Values updated during the run (for instance, accuracy)
```mlflow.log_metric```

**Artifacts**
Files produced by the run (for instance, model weights)
**mlflow.log_artifacts**, **mlflow.log_image**, **mlflow.log_text**

If you would like to disable autolog on supported libraries then use.
```mlflow.autolog(disable=True)```

#### Tracking online
You can be able to track the ML flow online by setting a Url on the script.
```mlflow.set_tracking_uri("http://192.168.0.1:5000")```
set the mlflow sever to ```export MLFLOW_TRACKING_URI=http://192.168.0.1:5000```