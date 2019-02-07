CODE FOR REPRODUCIBLE HYPERPARAMETER OPTIMIZATION

To reproduce results in paper:
All results in the paper can be reproduced using the evaluation.ipynb Jupyter Notebook. The evaluation code is written in R and a Cran R installation is required (https://cran.r-project.org/). To run R in a Jupyter notebook the R Kernel for Jupyter can be used (https://github.com/IRkernel/IRkernel). In order to reproduce the plots the GGPlot library is also required (can be installed from R).

To re-run hyperparameter optimization:
The code folder contains the results for all hyperparameter optimizations that the results are based on. If however there is a need to re-run the hyperparameter optimizations themselves, this can be done by running the runner.py file with Python in each experiment folder. The hyperparameter optimization requires Sherpa (pip install parameter-sherpa) and Keras/Tensorflow, Baselines/Tensorflow, or Scikit-Learn. Note that the output of the hyperparameter optimizations was reduced to the necessary information so that CSVs with the results are small enough to be provided here. The code to generate the reduced CSVs can be provided upon request.
