Five levers:

- QA Augmentation or instruction formatting: formatting data into question-answer or instruction-response pairs
- Scaffolding: Showing full worked examples at first and then slowly taking away the scaffolds in the working
- Skill-DAG: Establishing a dependency between different skills trained on vs different benchmarks to adjust data domain weights
- Token selection: Filtering tokens or samples by their loss curves during training to eliminate useless training on already-trained tokens or noisy tokens
- CL/Difficulty: Ordering data in terms of difficulty, measured by various metrics, and determining a difficulty curve maximizes efficiency

How to measure significance for each lever (applies to all levers):

- Record the validation loss curves on all of these methods. Compare each of these models to the control, which is cosine decay with randomly shuffled data. Fit a power law curve to the validation loss curve and use residual block bootstrapping to calculate the 95% confidence interval of the compute savings during training. Add any compute overhead, and then reject the null hypothesis if both ends of this 95% confidence interval indicate savings in total compute.
- Set the baseline benchmark score (an average across industry standard benchmarks) to be the control model’s benchmark scores at the full training token budget. Define each of the experimental models’ compute savings as the difference between the amount of total compute at which the experimental model first reaches the baseline benchmark and the full training token budget.
- If the model fails to reach the baseline benchmark score or the baseline validation loss within the training token budget, then automatically fail to reject the null hypothesis
