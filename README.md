# DoubleFragility

This repository contains the replication material for the paper:

Testa, L., Chiaromonte, F., Roeder, K., "Double fragility, and how to correct it" (2025+), forthcoming on [arXiv](...)

The Python scripts in this repository implement the simulation study and the real-data analysis described in the paper. The files are organized as follows:

- *sim_analysis.py*: This script implements the simulation study as described in the main text of our manuscript, which fully replicates the paper of [Kang and Schafer (2007)](https://projecteuclid.org/journals/statistical-science/volume-22/issue-4/Demystifying-Double-Robustness--A-Comparison-of-Alternative-Strategies-for/10.1214/07-STS227.full). This script reproduces Supplementary Tables C.1 and C.2, which in turn can be further summarized into Table 1 in the main text.
- *sim_coverage_analysis.py*: This script implements the comparison analysis for coverage, again based on the simulation design of Kang and Schafer (2007). This script reproduces Table 2 in the main text. 
- */sim_plot_example.py*: This script generates Figures 1 and 3 in the main text, showing results for the simulation study in the main manuscript.
- *app_analysis.py*: This script implements the real-data analysis on XXX data. Data can be downloaded from the [XXX](...).
- *app_plot.py*: This script generates Figure 4 in the main text using the output from the previous script.
