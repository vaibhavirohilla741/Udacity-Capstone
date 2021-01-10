*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Capstone Project (Udacity Microsoft Machine Learning Program)

The main goal of the project is to utilize Azure Machine Learning Hyper Drive and Auto Ml capabilities to build a machine learning model and deploy the best model based based on the performance evaluation metric.

## Project Architecture

![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/capstone-diagram.png "Logo Title Text 1")

## Dataset
The dataset can be be found on kaggle.
Short discription of the dataset id=s below.

1. Pregnancy : Number Of Pregnacies happend
2. Glucose : Blood Glucose Levels
3. Blood Pressure : Blood Pressure Levels
4. Skin ThickNess : Triceps Skin Thickness
5. Insulin : Blood Insulin Levels
6. BMI :Body Mass Index
7. Diabetes :Diabetes Function
8. Age : Age Of Patient in Years
9. Outcome : 1 0r 0 to indicate whether a patient has diabetes or Not


## Machine laerning with the dataset

Once we have registered the dataset in the azure workspace we have to run two models ie. by automl and by hyperturnning method and find out the best accuracy provided by which of these models.

### Access the dataset in the workspace
To access the dataset in the workspace we have to upload the dataset from the local files.We can also get the dataset by provideing the url of the dataset.

## Hyperdrive Run
  - The model is Logistic regression I have choosen Inverse Regularisation factor (--C) which penalises the model to prevent over fitting and maximum number of     iteration(--Max_iter) as other Hyperparameter to be tuned.
  - I have choosen Bandit policy as it terminates the model building if any of the conditions like slack factor, slack amout, delay interval are not met as per prescribed limits during maximizing the accuracy.
  - Once we pass the Hyper parameters using train.py and submit the Job Hyper Drive will create number of jobs based on the parameters given in Hyperdrive configuration using the combinations of Hyper parameters.
  - After running the modelwe need to save the model with best accuracy.In this case the model is 
  ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/screenshots/Hdrivebestrunmetrics.PNG "Logo Title Text 1")
  
  ### Hyperdrive Parameters
  Wee have define a search space with two parameters, --C and --max-iter. The --max-iter can have a uniform distribution with 300 as the best input and --C has the best value as 1 .
  ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/hdrivebestrunparams.PNG "Logo Title Text 1")
  
  
### Run Widget
![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/HyperdriveRunDetails.PNG "Logo Title Text 1")

### Results
The hyperdrive best model parameters are
![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/hdrivebestrunparams.PNG "Logo Title Text 1")
This shows the successful completion of Hyperdrive model Running along with best parameters and accuracy.

## AutoML Run
  - Here we are going to build a automl model for our problem. The Dataste is registered and converted to Tabular Dataset using Tabular dataset Factory module.
The AutomL config can be as seen below.
 ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/screenshots/automlconfig.PNG "Logo Title Text 1")
 - Once the model is run we can find different model with different accuracies.
 - the best model is
 ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/screenshots/best%20model.png "Logo Title Text 1")
 
 
## AutoMl Parameters

Various configuration used in AutoML
  1. Since the probelm is classification we choose task as classification.
  2. TimeOut is set to 30 minutes sicne the dataset is only 800 rows approximately.
  3. Primary metric is accuracy as we are trying to maximise the accuracy.
  4. label column is the column we are trying to predict here outcome .
  5. Compute target is the target cluster where the computation needs to be done
  6. N_cross_Validations=5 the number of k fold cross validations, since the dataset is small choosen 5
  7. Iterations: Number of iterations to be run 24 , so this checks 24 automl models Max_concurernt_iterations: 8 number of parallel runs at a time, choosing this too 
  8. high impact performance so choosen 8
  
## AutoML Best Model 

- After all the Runs AutomL gave voting ensemble model as best model with accuracy of 78.39 better than HYperdrive model.VotingEnsemble model works on taking the majority voting of underlying models and choose the model with highest votes as best model.
- The Fitted Model Parameters are as shown below.
![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/screenshots/best%20model.png "Logo Title Text 1")

  ### Parameters
  ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/automlfittedmodelparams.PNG "Logo Title Text 1")
  ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/automlfittedmodelparams1.PNG "Logo Title Text 1")
  
  ### Run Widget
  ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/automlmodelrunstatus.PNG "Logo Title Text 1")
 
 

### Results
- After all the Runs AutomL gave voting ensemble model as best model with accuracy of 78.39 better than *HYperdrive model
 

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
  - Before Deploying the model we need to pack all the dependencies into conda environment file whcih are included in the repository. Once we pack the dependencies a docker conatiner is built and pushed to Azure Container isntance. We need to consume the ACI instance using a rest Endpoint. The endpoint deployed will be seen in endpoints section of the Azure Machine learning studio. Before deploying an endpoint we need to define scoring script which defines the entrypoint to the deployment whcih is given in repository.
![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/screenshots/deplayement%20status.pngg "Logo Title Text 1")
 - This shows the Endpoint is successfully deployed and is healthy.Now we can consume the endpoint using scoring URL genereated after deployment.
![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/screenshots/endpointhealthy.PNG "Logo Title Text 1")

  
  
## Endpoint Consumption

  - Now we can consume the endpoint using scoring URL genereated after deployment.
  - The Endpoint is consumed using endpoint.py where we use requests library for cosnuming the endpoint.
  - The sample input to the endpoint is as below
  ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/endpoint%20input.png "Logo Title Text 1")
  - Here we are testing two datapoints and we are expecting two outputs
  ![alt text](https://github.com/vaibhavirohilla741/Udacity-Capstone/blob/main/screenshots/predicted%20values.png "Logo Title Text 1")
  
## Screen Recording

Here is the link of screencast.
https://www.youtube.com/watch?v=uFjvg4zDgIc


## Future Enhancements 

 - The model can be converted to ONNX format and deploy on Edge devices.
 - Applciation insights can be enabled.
 - Over-fitting and imbalanced data are common pitfalls when you build machine learning models. By default, Azure Machine Learning's automated machine learning provides charts and metrics to help you identify these risks, and implements best practices to help mitigate them.
 - Using more data is the simplest and best possible way to prevent over-fitting, and as an added bonus typically increases accuracy
