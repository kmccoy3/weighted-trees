# weighted-trees
This code repository accompanies the manuscript titled "Weighted Sum-of-Trees Model for Clustered Data" by  (McCoy et al., 2025). The manuscript is currently in peer review. This README will be updated as the article gets closer to publication.

# Repository Structure

- `./simulations/`
  - `run_simulation.py`: Run simulations from *Simulation Setting 1* and *Simulation Setting 2*.
  - `new_sims.py`: Runs simulations *Simulation Setting 3*.
  - `print_plots.Rmd`: Prints nice plots in R from exported data.
  - `BART_sims.R`: This code tests BART on the synthetic data generated in `new_sims.py`.
- `./sarcoma_analysis/`
  - `data_cleaning.ipynb`: Cleans raw sarcoma files present in `./data/`.
  - `PCA.Rmd`: Prints PCA plot.
  - `sarcoma_data.ipynb`: Analyzes various methods on VIVI plots.
  - `vivi_plots.Rmd`: Prints all VIVI plots.
- `./data/`: This folder is where the user must download the external sarcoma data to run files in `./sarcoma_analysis/`.
- `./figures/`: This folder contains the figure outputs of various files.
- `./out/`: This folder will contain the results of code run in `./simulations/`.