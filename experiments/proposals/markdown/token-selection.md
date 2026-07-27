Difficulty selection: [RHO-1](https://www.google.com/url?q=https://arxiv.org/abs/2404.07965&sa=D&source=editors&ust=1784950268294608&usg=AOvVaw0KcUbUc8RGjp6gzqwX2QiL) reached parity training on 3% of tokens by selecting the right ones.

## Evidence

- [RHO-1](https://www.google.com/url?q=https://arxiv.org/abs/2404.07965&sa=D&source=editors&ust=1784950268295071&usg=AOvVaw3n9ru9KOcazpqLnywAMhGU)
- This paper claims that not all tokens are equally mathematically valuable to learn from. They calculate “excess loss” by taking the loss on the training model minus the loss on a proxy model trained on high-quality data. They only counted the top 60–70% of the tokens by excess loss towards the loss function and achieved the same benchmark scores with 3% of the training budget
- [[2310.15389] Irreducible Curriculum for Language Model Pretraining](https://www.google.com/url?q=https://arxiv.org/abs/2310.15389&sa=D&source=editors&ust=1784950268296034&usg=AOvVaw0qsfs0I3NE88D5bQ4ii0Qk)
- This paper proposes the concept of learnability, which effectively yields the same effect as RHO-1’s token filtering. They train a small 80M proxy model on a high-quality held-out dataset and save an early and a late checkpoint, and then use the delta of loss on each sample of the large training dataset to determine how learnable it is, with higher learnability meaning the token is less likely to be noise.
- [Centrifuge (ICLR 2026)](https://www.google.com/url?q=http://openreview.net/forum?id%3DeshXwEnENV&sa=D&source=editors&ust=1784950268297008&usg=AOvVaw1qDF7i5--bLYLwH1orlIV5):
- Identifies a critical weakness in how RHO-1 style efficiency claims: loss masking does not save training time, because the model still performs backward pass computation on the unselected tokens. Centrifuge changes how the backward pass is implemented so that filtered tokens also skip internal computation. Tested across a wide range of model sizes (1.1B to 40B parameters), filtering 50% of tokens reduced backward pass time by up to 49.9% and end-to-end training time by up to 34.7%, while retaining the accuracy/utility gains of RHO-style filtering.
- [LaCy (ICLR 2026 Workshop)](https://www.google.com/url?q=https://machinelearning.apple.com/research/lacy&sa=D&source=editors&ust=1784950268298182&usg=AOvVaw0zFGrlcIML_KX6WQRU0ZLk)
- LaCy directly compares with RHO-style loss masking and challenges RHO-1’s selection signal. It argues that loss alone cannot identify which tokens a small model should learn because a high-loss token may have several valid continuations rather than being noisy. Its grammar-plus-loss method outperforms the RHO baseline on factuality in a 334M model cascade experiment.
- [Investigating Data Pruning for Pretraining LLMs at Scale](https://www.google.com/url?q=https://arxiv.org/pdf/2309.04564&sa=D&source=editors&ust=1784950268299226&usg=AOvVaw0rd4RqUK8O1RJmVFQsR-sR)
- Tested three ways to prune pretraining data before training (perplexity, EL2N, and memorization scores) across models from 124M to 1.5B parameters. They found that keeping just 30–50% of data from the middle of the distribution based on perplexity could outperform training on the full dataset by up to 2.1%. They also found that larger, cleaner data reference models scored higher and that an early, non-fully-trained reference model checkpoint performed nearly as well as a fully converged one.
- [[2606.18650] BLADE: Scalable Bi-level Adaptive Data Selection for LLM Training](https://www.google.com/url?q=https://arxiv.org/abs/2606.18650&sa=D&source=editors&ust=1784950268300539&usg=AOvVaw0Ro22MakpvNPQs1CPygFj-)
- This research aims to solve a problem with excess-loss calculations based on a reference model. The reference model drifts farther away from the training model as training progresses, making the loss information less representative. BLADE makes the reference model dynamic, performing a synchronization every t steps. During a synchronization, the reference model first copies the weights from the training model exactly, and then trains on a mix of the training data and a high-quality corpus for K steps. In their experiment, they used K=300 and t=1000
- [ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning](https://www.google.com/url?q=https://arxiv.org/pdf/2510.18250&sa=D&source=editors&ust=1784950268301856&usg=AOvVaw1668Aa-_vjiPjdhsAukVdg)
- This study introduces two more ways of measuring excess loss: one based on loss deltas, and one based on attention. For the REL (retrospective excess loss), they compare the loss on the current model to the loss on an EMA of all past checkpoints to compute excess loss. For the attention-based method, they took each prompt token (the keys) and summed up all of the attention weights (the key-query dot products). They averaged this for the prompt token across all heads of attention to determine its attention-based score. They also explored a weighted combination of attention and REL and found that 50/50 or 75% REL were best.

### Where the current evidence is strong

- Not all tokens or documents are equally valuable for a gradient update, and filtering can be done cheaply using a reference model’s loss as a yardstick. This is proven at our ~1B param scale.
- The mechanism has a legible, checkable signal (excess loss = current loss - reference loss); it can be inspected token by token and audited.
- Reaching the same target performance 5-10x faster is a large, hard-to-dismiss effect size. Various scoring metrics independently corroborate the general principle of learnability-based filtering.

### Where the current evidence is conflicting, and issues exist

- The headline “5-10x” and especially the widely repeated “3% of tokens” framing conflate two different comparisons. One is a genuine same-corpus token selection result, the other is a cross-dataset comparison against an entirely different, much larger training set. These should not be treated as the same claim, and any use of this evidence needs to cite the within-corpus 60-70% selection number, not the cross-dataset “3%” framing.
- RHO-1’s result is from a math-specific continued pretraining setting; it's untested whether the same excess loss signal transfers cleanly to a general, mixed subject educational corpus where "easy" tokens (e.g., function words, common phrasing) may still carry pedagogically important signal that a narrow excess loss filter would discard.
- The two scoring approaches (excess loss vs. a frozen reference model; learnability gap vs. a proxy model's own early/late checkpoints) haven't been directly compared against each other on the same corpus - it's unclear whether they select overlapping or divergent token sets, or which produces a better selection signal.

### Where the current evidence is weak

- Masked targets still require a full forward pass over the entire sequence (all tokens remain in context), so the compute savings are specifically from skipping loss computation and backward pass gradient contribution on non-selected tokens - this is a narrower efficiency claim than "training on fewer tokens" might suggest, and needs to be measured explicitly rather than assumed.
- No published work has tested whether a raw difficulty selection baseline (highest loss tokens under the current model, with no reference model at all) captures most of RHO-1's benefit on its own, or whether the reference model comparison is doing real additional work; this is exactly what Condition 3 vs. Condition 4 below is designed to isolate.
- Centrifuge's speedup numbers are implementation- and hardware-specific. It’s unverified whether the same 30%+ end-to-end speedup materializes on our testbed setup.

## Hypothesis

Null hypothesis: Pruning tokens from a dataset yields no reduction in the amount of compute required to reach the same performance compared to both training on the entire dataset or training on a randomly selected k% of the dataset.

Alternative hypothesis: Selective token loss reaches the same mastery at lower total cost than both controls.

## Proposed experiment

Continually pretrain the same OLMo 2 checkpoint (fully trained) and also do pre-training from scratch on the same frozen curriculum corpus. Every condition must use identical sequences of data samples:

## Experimental conditions

- Entire dataset (control)
- Random k%
- Top k% by raw loss
- Top k% by ssToken-style retrospective loss
- Top k% by RHO-1 style excess loss
- Middle k% by perplexity
- Top k% by attention score
- Top k% by learnability
- Top k% using dynamic reference model

Default to k = 60% (found to be optimal or nearly optimal in many of the studies), but if compute allows, try values of k between 40% and 80% for each experimental condition.

## Follow-up experiments, if null hypothesis is rejected for any of the conditions

- Keep track of the scores for each of the metrics tested in conditions 2 through 6: if one of these is highly correlated to another, they are likely measuring the same thing and can be combined
- For the conditions that worked well (null hypothesis was rejected), try weighted average blends of those metrics, ensuring that each metric is normalized to a score between 0 and 1 using softmax before it is combined with the other scores.
