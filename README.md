# Vanilla Matrix Factorization For Collaborative Filtering
## Algo Desc
- framework: pytorch 1.10.0
- loss: BPR(Bayesian Personalized Ranking) loss funciton
- optimizer: Adam
## Data Desc
- ml-1m: Movie Lens 1M
- process.ipynb: display the process of data preprocessing
## File Desc
- config.toml: for hyperparameter tuning and other common settings.
- dataset.py: for data loading and sampling
- evaluator.py: for evaluating the effectiveness of recommendation result
- main.py: you know
- model: implement Matrix Factorization by using torch.nn.Module as super class
- trainer: for executing the training process
- util: some useful util functions
