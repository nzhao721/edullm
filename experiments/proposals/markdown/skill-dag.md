Skill-DAG / prerequisite-ordered sequencing: ordering data by dependency structure, NOT naive easy-to-hard. Evidence: [Skill-It](https://www.google.com/url?q=https://arxiv.org/abs/2307.14430&sa=D&source=editors&ust=1784950268276931&usg=AOvVaw2GTKL7RT5fyzeDT5O_-5JD) (1B tokens ≈ 3B uniform, ~3x), [DoReMi](https://www.google.com/url?q=https://arxiv.org/abs/2305.10429&sa=D&source=editors&ust=1784950268277158&usg=AOvVaw2baH0OdgWlqyxt3eU6snQk) (2.6x fewer steps).

## Evidence

- [RegMix: Data Mixture as Regression for Language Model Pre-training](https://www.google.com/url?q=https://arxiv.org/abs/2407.01492&sa=D&source=editors&ust=1784950268277584&usg=AOvVaw27t66oODXTBvykAqxIxi6y) (Liu et al., 2024) — Optimizes domain mixtures using a regression model trained on 512 tiny models, matching DoReMi with 10% compute.
- Data domain mixture can be framed as a regression model problem. The researchers in this study trained a large number (512) of micro models of 1M parameters to predict the best domain weights for their dataset. They chose random distributions of domain weights on a Dirichlet distribution (keeping the weights realistic) and fit a lightGBM regression model to predict val loss from domain weights. They found that the val loss to domain weights model predicted by the regression model scaled well to a 1B test model.
- [Data Mixing Laws](https://www.google.com/url?q=https://arxiv.org/abs/2403.16952&sa=D&source=editors&ust=1784950268279012&usg=AOvVaw3uzLdQbFgTvYuxuBxiHWC-) (Ye et al., 2024) — Predicts large model performance on massive data using small-scale training runs to optimize mixture proportions.
- Executes a similar procedure to the RegMix paper in using small training runs to fit a model that predicts validation loss from domain weights. The power law can be fitted to extrapolate the loss depending on both number of parameters and number of training tokens according to the formula E + B/Sa, where S is the number of steps or the number of parameters and the other numbers are fitted parameters. Using this extrapolation, the final loss at the target model size can be estimated from incomplete training runs of smaller models. These projected losses from each domain weight combination are then used to fit the data mixing law

![image](images/image6.png)

tij, ki, and ci are all fitted parameters, rj are domain weights

- [Skill-it!](https://www.google.com/url?q=https://arxiv.org/abs/2307.14430&sa=D&source=editors&ust=1784950268281311&usg=AOvVaw1vDTrLXtDtzNFufgXKyhZx) A Data-Driven Skills Framework
- This paper establishes the idea of a prerequisite graph showing how much different skills depend on different datasets. The dependency graph is an adjacency matrix where the weights represent how much training on dataset domain i improves skill j. After building this matrix, training is split into segments and the dataset domain weights are initialized and updated throughout training like this

![image](images/image2.png)![image](images/image5.png)

The calculation of the matrix weights can be done either by isolating each pair (training on datasets i and j vs only j and then evaluating on skill j) or by training on one data domain and evaluating effects on all skills (more efficient but less accurate)

### Where the current evidence is strong

- Using something like an adjacency matrix of data domains and eval domains to select domain weights is an effective strategy
- Results on small models (order of magnitude 100M) scale well to moderate-size models on the 1B order of magnitude in terms of data domain experiments
- These data sampling methods result in a moderate reduction in compute (~30–35%) and a few pp improvement in benchmark scores

### Where the current evidence is conflicting and issues exist

- The process of determining the domain weights can be computationally heavy, and different papers have different ways of optimizing the compute usage in determining the dataset balance
- Whether the domain weights should be chosen upfront or adaptively reweighted

### Where the current evidence is weak

- The specific loss curve: whether the initial loss curve is very steep. Most research labs are more concerned with achieving exactly the same benchmark scores with slightly reduced compute or a few pp higher on benchmark scores with equal compute, not sacrificing a few pp on performance for an order of magnitude reduction in compute
- What stage or checkpoint of training this best applies to (some are pretraining from scratch, some are continually pretraining, some are fine-tuning)

## Hypothesis

Null hypothesis: Pre-training with a dataset with adaptive domain weights requires the same amount of compute to achieve the same validation loss compared to pre-training with a dataset with fixed domain weights.

Alternative hypothesis: Pre-training with a dataset with adaptive domain weights requires less compute to achieve the same validation loss compared to pre-training with a dataset with fixed domain weights.

## Proposed experiment

Do pre-training from a very early checkpoint of OLMo-1B. Compare the following methodologies as the independent variable

- Natural weighting of the dataset
- Select optimal domain weight based on the regression model and keep those weights throughout training. Use incomplete training runs of various 50-100M sized models and extrapolation with the power laws to fit a lightGBM regression for val loss as a function of domain weights. Then minimize that function
- Same as (2), but with the data mixing law
- Select optimal domain weight to start, and then adjust the weights 5 times throughout pre-training according to the skill-it formula. Use an approximately 100M proxy model and probing every pair of domain combinations.
- Same as (4), but with T-LITE-style behavioral clustering over domains (see below). Skill-It and Aij operate on clusters rather than raw domains.

### Behavioral clustering (T-LITE-inspired)

Borrow the abstraction idea from T-LITE [Duong et al., ISCA 2024](https://www.google.com/url?q=https://www.cs.utexas.edu/~lin/papers/isca24.pdf&sa=D&source=editors&ust=1784950268289315&usg=AOvVaw1pY8Ccavzexb9UzIyyH6Yh): instead of letting the weighting policy act on every raw domain, group domains that have similar transfer behavior and share one weight / one adjacency row per group.

- Represent each domain by a cheap behavioral signature — e.g. its estimated effect on each skill (a row of A), or ![image](images/image3.png) from the data mixing law — not by topic embedding similarity.
- Cluster domains with k-means into a fixed number of behavioral clusters K.
- Run Skill-It (and selective probing) at the cluster level: learn/adapt K weights, then sample within a cluster by natural weighting.
- Optionally mirror T-LITE’s candidate selection as well: for each skill, only probe the top-N candidate prerequisite clusters by the derivative estimate (ties into the further step below).

This is mainly a compute lever on arm (4): fewer units to probe and reweight, with the hope of matching full Skill-It val-loss curves at lower probing cost. Success is measured the same way as the cheaper-Aij follow-up — less probing/model-fitting compute than pairwise Skill-It, and not statistically significantly worse pretraining compute to target validation loss.

### Another further step to try if the null hypothesis is rejected in the previous experiment

- We can try to estimate Aij using the derivative of the data mixing function with respect to one domain weight. This enables us to much more cheaply calculate Aij than if we used probing.
- Additionally, since the Skill-it formula is exponential in Aij and ignores negative values, we can run probing for only the highest estimated Aij’s, decreasing the cost with negligible impact to error if we find that entirely using the derivative introduces too much error

![image](images/image7.png)

### How we can measure the success of this new formula

- This approach must use less compute during the probing and model-fitting stage (not including training) than a full skill-it style probing of the adjacency matrix
- It must not yield statistically significantly worse outcomes to the amount of pre-training compute required to achieve the same validation loss at any value of validation loss near convergence and must not yield statistically significantly worse convergence validation loss
