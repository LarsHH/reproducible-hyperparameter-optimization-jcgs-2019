# Code for Reproducible Hyperparameter Optimization

## Summary
Most results can be reproduced using the `main.ipynb` Jupyter notebook. The evaluation code is written in R and a Cran R installation is required (https://cran.r-project.org/). To run R in a Jupyter notebook the R Kernel for Jupyter can be used (https://github.com/IRkernel/IRkernel). In order to reproduce the plots the GGPlot library is also required (can be installed from R).

## Organization
The code in this repository is organized as follows:

* `main.ipynb`: Jupyter notebook to recreate all plots from the paper and re-run simulations.
* `distribution-of-outcomes`: Individual hyperparameter search runs whose results are used as is.
* `sherpa-seq-testing`: A fork of https://github.com/sherpa-ai/sherpa/ that includes an implementation of the proposed sequential testing method.
* `simulation-results`: Pre-computed results for hyperparameter settings that are used for simulation.
* `merge-datasets-util.ipynb`: Notebook to created merged CSV files in simulation-results folder.


## Re-running hyperparameter optimizations
This requires a Python installation. Navigate to `sherpa-seq-testing` and run `pip install -r requirements.txt`. Furthermore:

```
pip install tensorflow
pip install keras
pip install sklearn
pip install stable-baselines[mpi]
```

* **simulation-results**: to re-run pre-computed hyperparameter searches the `runner.py` file in the corresponding experiment subfolder can be run.
* **distribution-of-outcomes**: to re-run the individual hyperparameter optimizations the `runner.py` file in the corresponding experiment subfolder can be run.