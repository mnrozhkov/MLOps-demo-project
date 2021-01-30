ML Ops demo
==============================
Demonstrate introducing reproducibility, pipelines automation and ML Ops in data science projects using DVC, MLflow, and Airflow.

***

## 1. From Jupyter to CI/CD

Transformation from a prototype in Jupyter to a reliable and reproducible ML solution:

    Step 1. Organize the project and develop the prototype
            Manage dependencies with virtual environments and Docker

    Step 2. Reorganize the code from notebooks into .py scripts that compose pipeline stages

    Step 3. Automate the pipelines using DVC

    Step 4. Introduce tests

    Step 5. Manage experiments and track metrics with DVC and MLflow

    Step 6. Automate pipelines with Airflow

    Step 7. Setup CI/CD and ML Ops

***

## 2. Mock project
A mock classifier project our ML Ops is built for

### Problem statement

- **Binary classification**
- **Target:** user device change during a month
- **Metrics:** 
   * Lift score (how better is the model compared to the random sample)
   * precision @ top k
   * recall @ top k

### Train data

`target.feather` table attributes

    - user_id
    - month: rounded date up to the end day of the month
    - target: 1/0 (1 - user changed device, 0 - nothing happened)
    - month: '2020-04-30', '2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31' 

`user_features.feather` table attributes

    — user_id
    — month: when features were gathered
    — feature_N: feature (totally 50 including categorial)
    - month: '2020-04-30', '2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31' 
    
### Scoring data

Used to emulate model behaviour in production and monitoring (not used for model fitting)

`scoring_user_features.feather`

`scoring_target.feather`

`month`: '2020-09-30'
   
### Validation

- Model must show stability during the last 4 months
- Validation takes into account the temporal structure, i.e. 4 folds (monthly)
- Model selection based on aggregated metrics over folds: mean, std, min, max 
 
 ### Target metrics
 
Assume that the entire sample (customer base) might be large, 
hundreds of thousands or millions. Therefore, from a business logic perspective, 
it is advisable to choose the **top k users with the maximum probability of the target event**.
Accordingly, we are interested in the model that performs as better as possible 
for such a sample of top k users. 
  
To calculate metrics for top k, forecasts are first sorted by predict_proba
(in descending order) and then the required metrics for those @k objects are calculated: 
 
    - Precision@k
    - Recall@k
    - Lift@k
 
k is taken as 5% out of the sample size.

### How to run

#### 1. Use virtual environment

- Сreate and activate virtual environment:
```bash
python3 -m venv venv
echo "export PYTHONPATH=$PWD" >> venv/bin/activate
source venv/bin/activate
```

- Install dependencies:

```bash
pip install -r venv_requirements.txt
```

- Add virtual environment to Jupyter notebook:
```bash
python -m ipykernel install --user --name=venv
``` 

- Run Jupyter notebook:
```bash
jupyter notebook
```

- To deactivate virtual environment: 
```bash
deactivate 
```

#### 2. Run with Docker

- Create `config/.env`
```bash
GIT_CONFIG_USER_NAME=<git user>
GIT_CONFIG_EMAIL=<git email>
```

- Build
```bash
ln -sf config/.env 
docker-compose build
```
or 

```
docker-compose  --env-file ./config/.env build
```

- Run 
```bash
docker-compose up
```

