# Cogmood Analysis
------------------

Here's the code for analyzing the cogmood project.

Data is prepped first with the code in notebooks/exclusion_criteria.ipynb
Then models are fit with the code in notebooks/running_subs.ipynb and notebooks/run_flkr_hpc.ipynb

The flanker models are run from an apptainer container because they are based on old cuda code. That code is in packages/Supreme/Analysis/Flanker. The code for the othe models is in packages/SupremePilot.