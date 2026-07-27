Context anchors: instruction-formatted corpora let an [8B model match a 70B](https://www.google.com/url?q=https://arxiv.org/abs/2406.14491&sa=D&source=editors&ust=1784950268313656&usg=AOvVaw1h3z-etRw821wmB4_Hc71b) (~9x); curation alone is worth [6.6–10x](https://www.google.com/url?q=https://arxiv.org/abs/2406.11794&sa=D&source=editors&ust=1784950268313902&usg=AOvVaw1VnB9OxSwaL4YRI_4v78C3). Important negative result: naive developmental "easy-to-hard" curricula were rejected (BabyLM challenge: "largely unsuccessful," β = −3.6).

## Evidence

- [Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](https://www.google.com/url?q=https://arxiv.org/abs/2511.18903&sa=D&source=editors&ust=1784950268314665&usg=AOvVaw1VvZUk76CgfKW1uh2DRXfv)(Luo et al., 2025) — Demonstrates that curriculum learning fails under standard LR decay; proposes Curriculum Model Averaging (constant LR + weight averaging).
- Traditional learning rate decays (like the cosine decay curve) make curriculum learning (saving highest-quality data for last) disadvantageous since it deweights the highest-quality data. The main purpose of learning rate decay is to eliminate noise in gradient descent towards the end of training, and another way to do this is to take an average of late-training checkpoints. This paper found that the best results came from taking a simple or exponential moving average of the last 6 checkpoints’ weights (they did not specify where the checkpoints were) with either constant learning rate (with warmup) or WSD with decay phase comprising 15–20% of total training and ending learning rate ⅓ of peak
- [Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning](https://www.google.com/url?q=https://arxiv.org/abs/2506.11300&sa=D&source=editors&ust=1784950268316666&usg=AOvVaw0N8OjIvn_I5U9kwDWqJT5I) (Zhang et al., 2025) — Systematic investigation identifying compression ratio and lexical diversity as the most effective difficulty signals for LLM curricula.
- Ordering data based on difficulty (curriculum learning) works well if done with the right pacing curves. “Vanilla” CL (naive strict sorting of data from easy to hard) does not work because the model struggles with the pure hard material at the end of training and catastrophically forgets easy concepts from early training. Instead, training should use a pacing curve, where there is always a mix of easy and hard data, but the average difficulty of the data increases. A linear or quadratic curve was found to work the best. Interleaving (splitting the training budget into sections and applying the pacing curve independently in each section) also showed strong results. All of these achieved faster results, but plateaued at the same level. Applying CL to only a warmup phase and using random shuffled data later resulted in both the initial speedup and 3.5% higher final performance
- [“Findings of the Second BabyLM Challenge”](https://www.google.com/url?q=https://aclanthology.org/2024.conll-babylm.1.pdf&sa=D&source=editors&ust=1784950268318837&usg=AOvVaw2pzGGrnADt8n440a-sL2L1) (2024):
- This paper took a lot of submissions of different LLMs and correlated various aspects of how they were trained to their benchmarking scores. They found that models that used curriculum learning did worse at a borderline significant, p = 0.055 level. However, there were no specifics on how CL was applied.
- [Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM Training](https://www.google.com/url?q=https://arxiv.org/abs/2607.04969&sa=D&source=editors&ust=1784950268319791&usg=AOvVaw2VwwZqb_F5y7NzhSDJpU4r) (July 2026)
- This paper investigated spaced repetition curricula for LLM pre-training. It quantified a forgetting curve for LLMs and also an overfitting curve to determine the optimal epoch length or spacing between repeating the same training sample. They compared the loss on a random unseen piece of data to a piece of data seen a time tau ago. They plotted this delta as a function of time tau and found that this delta decayed very quickly at an interval that they deemed the forgetting threshold. To quantify overfitting, they measured benchmark scores over time. For small epoch sizes, the scores quickly peaked and then declined due to overfitting. They increased this epoch size until the benchmark score curve no longer changed depending on epoch size and deemed this the threshold for overfitting.
- [Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics](https://www.google.com/url?q=https://arxiv.org/abs/2601.21698&sa=D&source=editors&ust=1784950268321640&usg=AOvVaw2l7uaCb5Ekja5gwth8MDfn)
- Researchers investigated sorting data from easy to hard in terms of three different metrics: Verb Variation, Word Frequency, and Age-of-Acquisition. They also calculated the loss of these on a checkpoint of a base model and found that difficulty on these metrics was correlated with the loss on that base model, suggesting that could be another way of calculating difficulty. They found that CL helped smaller models reach stability very effectively, but diminishing returns or even negative effects for larger models beyond 410M parameters.
- [[2310.15389] Irreducible Curriculum for Language Model Pretraining](https://www.google.com/url?q=https://arxiv.org/abs/2310.15389&sa=D&source=editors&ust=1784950268323134&usg=AOvVaw1p6S7WzB2JNm1X792BT4s2)
- Instead of measuring difficulty by human measures, such as vocabulary, rarity, or complexity, this paper argues that LLMs do not perceive difficulty the same way as humans and proposes measuring difficulty based on learnability. They train a small 80M proxy model on a high-quality held-out dataset and save early and late checkpoints, and then use the delta of loss on each sample of the large training dataset to determine how learnable it is, with higher learnability meaning “easier”. Then, the data is sorted from high to low learnability.

There is not much decisively strong evidence in this category. Both research methodology and conclusions widely vary across the evidence, and it is hard to synthesize and pin down the exactly ablation that is leading to any improvements or regressions.

### Where the current evidence is conflicting

- Whether curriculum learning in any form actually produces reliable results at all. There are numerous studies on both sides of this.
- The metrics that should be used to measure the “difficulty” of data samples. The existing research has used many different metrics of both human-perceived and loss-based difficulty and different studies have shown different metrics as being the best.
- The pacing of difficulty sorting. Evidence leans away from naive sorting of the entire dataset regardless of what difficulty metric is being used. Evidence leans towards more nuanced schedules that manipulate the proportion of easy and hard data while maintaining some random mixture.
- Scaling with model size: some of the evidence cited above was conducted on models slightly smaller than our ~1B param target size. It is inconclusive whether their results will scale well to our target size, and there is evidence that the benefits of CL diminish with larger models
- Learning rate decay as a confounding variable: curriculum learning fundamentally depends on placing harder data at the end, so a time-decaying learning rate may very likely nullify any benefits that the harder data at the end provides

## Hypothesis

Null hypothesis: Curriculum learning in any pacing with any measure of difficulty has no effect on the amount of compute required during pre-training to reach the same validation loss compared to the same dataset randomly shuffled.

Alternative hypothesis: There exists a setup of curriculum learning with a certain pacing schedule and metric of difficulty for which the amount of compute required to reach the same validation loss is smaller for the curriculum learning compared to the same dataset randomly shuffled.

## The proposed experiment

Do pre-training from scratch with the OLMo-1B architecture. Compare the following 3D matrix of independent variables:

Learning rate decay:

- Warmup + cosine learning rate decay
- Warmup + constant learning rate with exponential moving average of the weights of the last 6 checkpoints over the last 20% of the training token budget

Pacing:

- Control (random shuffle)
- Vanilla (naive easy-to-hard sort)
- Linear
- Warmup (CL for half the budget, then shuffle)

Metrics of difficulty:

- Control (randomly shuffled data)
- Learnability, as measured by the loss deltas on a smaller proxy model
- Compression ratio
- Flesch reading ease
- Measure of Textual Lexical Diversity

Paragraph draft

P1 - Formatting the Data Corpus to Increase Pre-training EfficiencyQA Augmentation: Facts can be memorized yet stay unanswerable until reformatted; we compare repetition, paraphrase, and QA-formatted restatements at matched budgets.

Scaffolding: Motivated by evidence that scaling worked solutions improves math accuracy, we pretrain on matched problem families under a fixed budget, comparing bare answers, complete worked solutions, a pre-registered fading schedule, and a shuffled-fade control using the same scaffold lengths in random order.

Skill-DAG: Dissecting our dataset domains and using cheap probing runs on ~50M parameter models to construct a dependency graph linking training on each domain to downstream performance on other domains. This graph will dynamically adjust domain weights during training in response to loss.Token Selection: Even in cleaned/filtered data, not every token is worth training on; some are learned early on while others are noisy and do not contribute to the learning signal. We are analyzing the loss curves of every token and selecting the most valuable tokens to train on at every step.Curriculum Learning (CL): Some research has proven that CL (easy-to-hard ordering of data) does not work, but there are inconsistencies in this evidence. We construct a rigorous sweep targeting a matrix of various learning rate curves, metrics of difficulty, and difficulty pacing strategies.

Hi Nathan I have this rough draft in for now on the doc i can help shorten the paragraph later or if you get to it before I do just replace what I have put in in the main doc

QA Augmentation: we'll continually pretrain OLMo-1B on synthetic data, Scaffolding: pretrains on matched problem families under a fixed token budget, comparing bare answers, worked solutions, and fading schedules, crediting fading only if it beats full worked solutions on unscaffolded accuracy and transfer. Skill-DAG: uses cheap ~50M-parameter probing runs to build a dependency graph linking domains to downstream performance, dynamically reweighting domains during training. Token Selection: analyzes per-token loss curves to filter out noisy or already-learned tokens, training only on the most valuable ones each step. Curriculum Learning: tests paced difficulty schedules against random shuffling, given mixed prior evidence on whether ordering data by difficulty actually helps.
