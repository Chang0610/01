# Distributed Climate Data Analysis and Prediction System

## Project Overview

This project demonstrates how to build and optimize distributed machine learning workflows using **Ray**—a powerful framework for scaling machine learning and AI workloads. The project covers key techniques including distributed data handling, model training with **Ray Train**, hyperparameter tuning with **Ray Tune**, and deploying models with **Ray Serve**. It also explores the use of **Accelerated DAG (ADAG)** to optimize complex machine learning workflows.

## Key Features
1. **Distributed Computing and Data Handling**:
   - Efficiently load and process large datasets in parallel using Ray’s distributed computing capabilities.
   
2. **Distributed Model Training with Ray Train**:
   - Train machine learning models such as **Ridge**, **Lasso**, **Random Forest**, and **Gradient Boosting** in a scalable fashion across multiple workers using **Ray Train**.

3. **Optimizing DAG-Based Workflows**:
   - Use Directed Acyclic Graphs (DAG) to manage complex workflows, including data preprocessing and model training, for efficient distributed execution.

4. **Hyperparameter Tuning with Ray Tune**:
   - Perform distributed hyperparameter optimization using **Ray Tune** to improve model performance. Explore a search space for parameters and schedule trials to maximize efficiency.

5. **Model Deployment with Ray Serve**:
   - Deploy machine learning models at scale using **Ray Serve** to handle real-time predictions. This ensures robust and scalable production-grade serving.

6. **Advanced Model Deployment with Accelerated DAG (ADAG)**:
   - Explore advanced usage of **Accelerated DAG** to optimize and scale machine learning pipelines beyond standard workflows.

## Installation

To run the project, you need to have Ray installed. You can install it with the following command:

```bash
pip install ray
```

For additional packages required for machine learning models (e.g., `scikit-learn`):

```bash
pip install scikit-learn
```

## Usage

### 1. Distributed Computing and Data Handling
```python
import ray
import pandas as pd

# Initialize Ray
ray.shutdown()  # Shut down any previous Ray instance
ray.init(ignore_reinit_error=True)

# Define a remote function to load data
@ray.remote
def load_data(file_path):
    return pd.read_csv(file_path)

# List of file paths for climate data
file_paths = ['total_precipitation/precipitation_data.csv', 'sunshine_duration/sunshine_data.csv']

# Load data in parallel using Ray
data_futures = [load_data.remote(file_path) for file_path in file_paths]
climate_data = ray.get(data_futures)
```

### 2. Distributed Model Training with Ray Train
```python
from ray.train.sklearn import SklearnTrainer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Define model configurations
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor()
}

# Train model in parallel using Ray Train
def train_model(config):
    model = models[config['model_type']]
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

train_config = {"model_type": "RandomForest"}
trainer = SklearnTrainer(train_func=train_model, scaling_config=ScalingConfig(num_workers=4))
result = trainer.fit(train_config)
```

### 3. Optimizing DAG-Based Workflows with Ray
```python
from ray.dag import InputNode

# Define a DAG for the preprocessing pipeline
@ray.remote
def preprocess_data(data):
    return data  # Your data preprocessing logic here

@ray.remote
def train_model_dag(data):
    return model.fit(data)  # Train the model using preprocessed data

# Create a DAG input node
data_input = InputNode()

# Define the workflow
preprocessed_data = preprocess_data.bind(data_input)
model_output = train_model_dag.bind(preprocessed_data)

# Execute the DAG
model_output.execute()
```

### 4. Hyperparameter Tuning with Ray Tune
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Define hyperparameter search space
param_space = {
    "alpha": tune.loguniform(1e-4, 1e-1),
    "fit_intercept": tune.choice([True, False])
}

def train_model_with_tune(config):
    model = Ridge(alpha=config['alpha'], fit_intercept=config['fit_intercept'])
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    tune.report(score=score)

# Run hyperparameter tuning
tune.run(train_model_with_tune, config=param_space, num_samples=10, scheduler=ASHAScheduler())
```

### 5. Model Deployment with Ray Serve
```python
from ray import serve
from sklearn.ensemble import RandomForestRegressor

# Initialize Ray Serve
serve.start()

# Define a deployment class for the model
@serve.deployment
class ModelDeployment:
    def __init__(self):
        self.model = RandomForestRegressor().fit(X_train, y_train)

    def predict(self, data):
        return self.model.predict([data])

# Deploy the model
ModelDeployment.deploy()

# Send a test request to the model
handle = ModelDeployment.get_handle()
result = ray.get(handle.predict.remote([0.5, 0.3, 0.2]))
print(result)
```

### 6. Advanced Model Deployment with Accelerated DAG (ADAG)
```python
from ray.dag import DAGNode

@ray.remote
def data_preprocessing_step(data):
    return processed_data  # Your preprocessing logic

@ray.remote
def model_training_step(processed_data):
    return trained_model  # Your training logic

# Create a DAG node
data_input = InputNode()

# Define the workflow
preprocessed_data = data_preprocessing_step.bind(data_input)
trained_model = model_training_step.bind(preprocessed_data)

# Execute the DAG
trained_model.execute()
```
