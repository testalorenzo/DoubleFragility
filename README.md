# Replication material for "Rescuing double robustness: safe estimation under complete misspecification"

This repository contains the replication material for the paper:

Testa, L., Chiaromonte, F., Roeder, K., "Rescuing double robustness: safe estimation under complete misspecification" (2025+), available on [arXiv](https://arxiv.org/abs/2509.22446).

The Python scripts in this repository implement the simulation study and the real-data analysis described in the paper. The files are organized as follows:

- *sim_analysis.py*: This script implements the simulation study as described in the main text of our manuscript, which fully replicates the paper of [Kang and Schafer (2007)](https://projecteuclid.org/journals/statistical-science/volume-22/issue-4/Demystifying-Double-Robustness--A-Comparison-of-Alternative-Strategies-for/10.1214/07-STS227.short). This script reproduces Table 1.
- *sim_coverage_analysis.py*: This script implements the comparison analysis for coverage, again based on the simulation design of Kang and Schafer (2007). This script reproduces Table 2 in the main text and Supplementary Table C.1. 
- *sim_plot_example.py*: This script generates Figures 1 and 3 in the main text, showing results for the simulation study in the main manuscript, together with Supplementary Figures C.2, C.3, C.4, and C.5.
- *sim_W_example.py*: This script generates Supplementary Figure C.1, showing how Normal approximation and parametric bootstrap tend to provide equivalent results.
- *app_analysis.py*: This script implements the real-data analysis on Alzheimer's proteomics data. Data can be downloaded from [Merrihew et al. (2023)](https://www.nature.com/articles/s41597-023-02057-7), and then preprocessed using the code provided by [Moon et al. (2025)](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-19/issue-2/Augmented-doubly-robust-post-imputation-inference-for-proteomic-data/10.1214/25-AOAS2012.short).
- *app_gene_analysis.py*: This script post-processes previous findings collecting gene-level data. 
- *app_plot.py*: This script generates Figure 4 in the main text and Supplementary Figure D.6 using the output from the previous scripts.
