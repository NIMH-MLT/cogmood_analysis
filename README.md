# Cogmood Analysis

Here's the code for analyzing the cogmood project.

Data is prepped first with the code in notebooks/exclusion_criteria.ipynb
Then models are fit with the code in notebooks/running_subs.ipynb and notebooks/run_flkr_hpc.ipynb

The flanker models are run from an apptainer container because they are based on old cuda code. That code is in packages/Supreme/Analysis/Flanker. The code for the other models is in packages/SupremePilot.

# Running analyses
1) Preprocess data with notebooks/exclusion_criteria.ipynb populating data/task/to_model
2) Run FLKR model with notebooks/run_flkr_hpc.ipynb
3) Run CAB, BART, and RDM models with notebooks/running_subs.ipynb
4) Run correlations in notebooks/task_survey_correlations.ipynb


# Generative AI use
- Claude Code with `qwen3-coder-next` was used to add argparse to subj_fit files in CAB, BART, and RDM.
- Claude's Sonnet 4.5 was used to speedup the bootstrap and permutation functions and write tests for those.