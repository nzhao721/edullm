Worked Examples with Faded Scaffolds - Showing a complete, step by step solved problem instead of an unsolved one, then gradually stripping steps out across a problem sequence so the model gets less hand holding over time

## Evidence

[Sweller & Cooper (1985) / Sweller (1988)](https://www.google.com/url?q=https://andymatuschak.org/files/papers/Sweller%2520-%25201988%2520-%2520Cognitive%2520load%2520during%2520problem%2520solving.pdf&sa=D&source=editors&ust=1784950268240185&usg=AOvVaw0OqXjgGROIit9kKCYpUp0w): Origin study: Algebra students who studied fully worked solutions learned faster than those who solved problems unaided and did better on transfer problems, ruling out "it's just memorization."

- Theory: Novices solving toward a fixed goal must juggle current state, goal state, the gap between them, and a subgoal stack all in working memory at once, which overloads it.
- Computational proof of the load difference: Sweller's PRISM model showed a goal-specific solver needs 4 productions/5 cycles/29 conditions matched for a 3-step problem, vs. just 1/3/17 for a nonspecific goal solver, a concrete measure of how much less mental machinery goal-free solving requires.
- Key experiment (24 trigonometry students, dual task design): goal-specific and goal-free groups solved equally well and equally fast on the main task, but the goal-free group recalled significantly more structural problem elements afterward (with no advantage on irrelevant details), evidence that freed up mental capacity gets spent building a schema, the same mechanism worked examples are thought to exploit.

[Barbieri et al. meta-analysis](https://www.google.com/url?q=https://www.danamillercotto.com/uploads/4/7/7/2/47725475/barbieri_et_al__2023__we_meta-analysis.pdf&sa=D&source=editors&ust=1784950268242543&usg=AOvVaw07fPgwvFBDw-wC231kJeEm):

What it is: Combined 55 studies (181 results, screened from 8,000+ candidates) testing whether worked examples beat solving alone in math. Headline effect: g = 0.48 (CI 0.36 - 0.60), a solid, medium, real effect, not a statistical fluke. Statistical care taken: Used RVE to avoid overcounting studies that reported multiple related results; checked for publication bias via funnel plot + Egger's test (found some bias), corrected with trim and fill (barely changed the estimate: 0.48 to 0.44), and confirmed with fail-safe N (~13,915 hidden null studies would be needed to erase the effect, which is implausible).

Big caveat: Massive variation between studies (I² = 93.7%); the average hides very different results depending on design.

What explained some of that variation:

- Correct-only examples beat incorrect/mixed correctness examples.
- Self-explanation prompts hurt rather than helped (likely shallow self-explanations or repetition).
- Timing (new topic vs. practice) didn't matter.

Why it doesn't fully transfer to P1:

- Comparisons are across studies, not one controlled experiment; suggestive, not causal proof for any single design choice.
- Measures only test accuracy, never efficiency (tokens/compute), the currency Incept cares about.
- No dosage data, no curve for how benefit scales with amount of worked example exposure.

Takeaway: g ≈ 0.48 is a reasonable real-world baseline expectation, but P1's own ablations need to measure token efficiency, dosage scaling, and isolate design choices directly rather than assume this transfers as is.

[Xwin-Math / GAIR-Abel (LLM-specific):](https://www.google.com/url?q=https://arxiv.org/abs/2403.04706&sa=D&source=editors&ust=1784950268246091&usg=AOvVaw3R-GUHOPknKW5HoY_QhRxQ) The paper's core manipulation is step-by-step chain-of-thought solutions versus terse final answer-only responses, and its core lever is scaling the volume of that step-by-step data. Method: starting from real GSM8K/MATH training questions, GPT-4 Turbo generates new questions (with a verification pass that solves and checks each generated question before keeping it), then generates a full worked chain of thought solution for each. Scaling this from 7.5K real examples up to 960K (GSM8K) / 480K (MATH) synthetic examples took LLaMA-2 7B from 50.2%/8.4% to 82.6%/40.6% on the two benchmarks, with no sign of saturation at the scales tested. The paper distinguishes Pass@256 (does the correct answer appear in any of 256 sampled generations a ceiling capability measure) from PassRatio@256 (what fraction of the 256 generations are correct a reliability measure), and finds that the scaling worked solution data mostly moves the reliability number, while the capability ceiling was already high with very little data. In other words, more worked example data made the model more consistently correct, not more capable of solving harder problems it couldn't touch at all before. The paper also decomposes errors into reasoning errors versus calculation errors and finds calculation errors are corrected faster than reasoning errors as data scales a useful secondary metric template for our own error analysis.

[[2402.04004] Understanding the Effect of Noise in LLM Training Data with Algorithmic Chains of Thought](https://www.google.com/url?q=https://arxiv.org/abs/2402.04004&sa=D&source=editors&ust=1784950268248714&usg=AOvVaw3RHGzqMryw1NXaFJYqPWsr)

- The authors fine-tune a pretrained Pythia 410M model on 20,000 synthetic arithmetic and median problems. The direct arm learns from final answers only. The algorithmic chain of thought arm learns from verified intermediate states and final answers.
- Resulting accuracy

Task

Final answer only

Verified algorithmic trace

Five-digit multiplication

39.6%

90.7%

Ten number median

30.6%

97.0%

Mixed arithmetic

60.4%

95.8%

[Distilling Algorithmic Reasoning from LLMs via Explaining Solution Programs](https://www.google.com/url?q=https://arxiv.org/pdf/2404.08148&sa=D&source=editors&ust=1784950268252189&usg=AOvVaw2rGLO6VoeUtktmQbl7iFVl)

- Experiment conducted in the domain of coding (algorithmic programming)
- An “Explainer” annotates explanations given <problem, solution> —> a “Reasoner” learns to reason through a problem —> a “Coder” implements the solution from the reasoning
- Explainer: frontier model (GPT-4)
- Reasoner: smaller model fine-tuned on <problem, explanation> pairs generated from the Explainer that is given a set of <problem, solution> pairs (closed: GPT-3.5-Turbo, open: DeepSeek Coder 7B)
- Coder: same model as reasoner w/o fine-tuning

### Where the current evidence is strong

- Sweller & Cooper (1985) showed the benefit isn't just memorization, since students also did better on transfer problems they'd never seen worked out.
- There's a plausible, quantified mechanism behind it (cognitive load / working memory conservation), not just a correlational finding Sweller's PRISM model gives a concrete computational accounting of how many "moves" a goal directed solver needs versus a goal free one (4 productions/5 cycles/29 conditions vs. 1/3/17), and the dual task recall experiment shows this freed up capacity gets spent specifically on encoding problem structure (the schema building elements), not irrelevant surface details.
- The effect size is replicated at scale, not a one-off: the Barbieri meta-analysis pools 55 independent studies (181 effect sizes) and lands on g = 0.48 with a confidence interval that stays clearly above zero, and this survives aggressive publication bias correction (fail-safe N ≈ 13,915).
- There's a direct LLM specific analog with a strong, unambiguous scaling result: Xwin Math/GAIR Abel show that scaling step by step worked solutions (not just terse answers) from 7.5K to ~960K examples took LLaMA 2 7B from 50% to 83% and 8% to 41% on two benchmarks, with no saturation this is the closest thing in the bundle to direct evidence that the effect transfers to a training corpus for an LLM specifically.

### Where the current evidence is conflicting, and issues exist

- The human literature disagrees on which supplementary design features help. The Barbieri meta-analysis found self-explanation prompts hurt on average (β = −0.24), which cuts against other work suggesting self-explanation is a beneficial addition. Since I² = 93.7% shows a huge amount of that variation is genuine rather than sampling error.
- The mechanism story doesn't cleanly match between the human and LLM evidence. Sweller's account is about freeing working memory to build a schema; Xwin-Math's result is explicitly framed as improving reliability (PassRatio@256) rather than raising the capability ceiling (Pass@256), i.e., the model was already capable of solving harder problems with little data, and more worked example data mostly made it get the right answer more consistently. It's unclear whether this is the "same" effect as human schema building, or a different phenomenon (better calibration/consistency) that happens to look similar on benchmarks.
- Whether correctness and fading specifically (vs. worked examples generally) matter is unresolved. The meta-analysis says correct-only examples beat incorrect/mixed ones, but nothing in this bundle directly tests whether faded scaffolds outperform static, non-faded worked examples. The evidence supports "worked examples help" and "gradual guidance removal is a plausible model of skill acquisition," but doesn't isolate fading as an independently tested lever.

### Where the current evidence is weak

- No dosage response curve. Neither the human meta-analysis nor Xwin-Math (despite scaling to 960K examples with "no sign of saturation") tells us the shape of the return curve: is it linear, log-linear, or does it plateau at some point relevant to a 1B parameter training budget? Xwin-Math only shows it hasn't saturated yet at the scales tested, not where it would.
- No efficiency metric that matches Incept's currency. All the human evidence measures test accuracy; Xwin-Math measures benchmark accuracy and pass rate reliability; none of it is expressed in tokens per unit of learning or compute cost, which is what "learning gain per token" and "mastery per dollar" require.
- Fading itself is untested here. The bundle justifies worked examples in general and frames fading as a natural analogy to human skill acquisition (via the phase models referenced elsewhere), but no cited study actually compares faded scaffold sequences against fixed-format worked examples at the same total data volume; this is arguably the single biggest gap between "what the evidence supports" and "what P1 is proposing to build."
- Error type decomposition (reasoning vs. calculation errors) has not been tested under a fading manipulation. Xwin-Math shows this decomposition is a useful lens (calculation errors close faster than reasoning errors as data scales), but this was measured under a chain of thought volume scaling manipulation, not a faded scaffold one it's a metric template to borrow, not existing evidence about fading itself.

## Hypothesis

Null hypothesis: At a fixed total training-token budget for continued pretraining, spending that budget on complete or faded worked solutions produces the same held-out problem-solving accuracy as spending it on bare problem–answer pairs (which fit more distinct problems and repetitions into the same budget).

Alternative hypothesis: At the same fixed token budget, complete and/or faded worked solutions produce higher held-out accuracy than bare problem–answer pairs (depth beats breadth per token).

## Proposed experiment

Do continual pretraining on an early checkpoint using matched problem families (procedural/math style problems grouped so each family shares an underlying skill, split into held-out and trainable instances the same way the QA augmentation lever splits Ptrain/Ptest). Compare the following as the independent variable:

- Arm 1 - Bare problem-answer pairs (control): each training document contains a problem followed only by its final answer, with no intermediate steps and no general expository text.
- Arm 2 - Complete worked solutions: each training document contains a problem followed by a complete, step-by-step solution and final answer.
- Arm 3 - Ordered faded completions: each problem family appears repeatedly under a pre-registered sequence of decreasing scaffold lengths, where the provided scaffold is context and the omitted continuation is the training target.
- Arm 4 - Shuffled-fade control: Arm 4 uses the same problem instances, scaffold-length multiset, and loss masking as Arm 3, but randomizes scaffold lengths within each family rather than presenting them from most to least support.

All four arms draw from the same pre-registered problem-family roster and use the same trainable and held-out split.

For Arms 2, 3, and 4, every family appears the same pre-registered number of times under the same distinct-instance policy.

Arm 2 presents complete solutions on those appearances.

Arm 3 presents the fixed fading schedule.

Arm 4 presents the identical scaffold-length multiset in randomized within-family order.

### Training-data validation

Before continued pretraining, validate the construction of the worked solutions and scaffolds. Where a deterministic or symbolic verifier is available, verify every final answer and intermediate step. For items that cannot be checked automatically, conduct a pre-registered, blinded audit sampled across problem families and scaffold lengths. Validate that each shown scaffold is a correct prefix of the complete solution and that the omitted continuation completes the same valid solution. Regenerate or remove any example that fails validation before the main training runs.

The same set of problem families and the same problem counts are used across all three arms this is what everything else kept identical refers to: which skills are covered, how many problem families, and the held out/trainable split. This is a matter of coverage, not compute, and it does not imply equal token counts per arm (see below).Matching compute as an equal token budget, not equal problem coverage: Worked solutions contain far more tokens per problem than bare answers, so "same problems, same counts" does not give arm 1, 2, and 3 the same number of training tokens, arm 2/3 would naturally run longer than arm 1 if every arm trained through its full problem set exactly once. Each arm is instead given the same fixed total token budget to train on (e.g., the same number of training steps × batch size). Arm 1 spends that budget by cycling through more distinct real problems and/or repeats of them (capped as above) from the same pool arms 2 and 3 draw from, it is not padded with filler text of any kind, since mixing in unrelated exposition would confound "no steps" with "different kind of content" and make it impossible to tell which one drove any difference in results. Arm 1 will therefore touch more distinct problems and/or repetitions within that budget than arm 2 or 3, since its examples are shorter; that's expected and fine, because coverage isn't the thing being held constant here, token spend is. Each arm's accuracy is logged as a curve over the course of training (checkpointed at regular token intervals) rather than measured only at the end. This turns the comparison into an efficiency question: not just who ends up more accurate at the same budget, but how many tokens does each arm need to reach a given level of performance.

### What to Measure

Primary metric: Unscaffolded accuracy on held-out problem instances from trained-on families.

At evaluation, every model receives only the bare problem, with no supplied solution steps or scaffolds.

Each item's score is the fraction of N sampled generations that are correct, using verifier checking or exact match.

This measures whether training with worked examples or fading helps the model solve new problems without continued help.

Secondary metric 1: Transfer: Accuracy on problem families never seen in any form during training. Distinguishes genuine schema learning from surface pattern matching (this is the LLM analog of Sweller & Cooper's original transfer test).

Secondary metric 2: Ceiling vs. reliability: For each arm, sample N generations per held out problem and report both:

- Pass@N: does the correct answer appear in any generation (capability ceiling)
- PassRatio@N: what fraction of generations are correct (consistency)

This separates the idea that "the model can now solve harder problems" from "the model gets the same problems right more often."

Secondary metric 3: Error type: Classify wrong answers as reasoning errors vs. calculation errors. Checks whether any gain specifically reduces reasoning errors (the type tied to schema acquisition) rather than just arithmetic slip-ups.

Decision rules:

- Reject the null if 2 and/or 3 beats 1 (bare answer) with non-overlapping 95% CIs.
- Only conclude fading adds value if arm 3 beats arm 2; specifically, beating 1 alone isn't enough.
- Confidence intervals: bootstrap the held-out set 1,000 times per arm; arms differ significantly only if their 95% CIs don't overlap.

### Follow Up Experiment (if the null is rejected)

If 2 and/or 3 beats the bare answer control, run a fading schedule sweep: compare multiple fading paces against each other to find the best one.

- Fast fading steps removed after 1-2 exposures
- Slow fading steps removed gradually across many exposures
- Fixed 50% steps permanently held at half shown, no further fading

Whichever schedule produces the best held-out accuracy wins. This step is necessary because neither the human literature (Atkinson et al.) nor any LLM study has established an optimal fading pace.

### If the null is not rejected (worked/faded solutions don't beat the control)

deprioritize authoring full step-by-step solutions; they're expensive to generate and verify, and shift effort to cheaper interventions like broadening problem answer coverage.

### Success Criteria for the Fading Schedule Sweep

A winning schedule must satisfy all three:

- Matches or beats the best result. Accuracy is not statistically significantly worse than the best single arm (2 or 3) from the main experiment.
- Same token budget. It uses no more total solution tokens per problem family than arm 2 (complete worked solutions) did; any gain must come from how the fixed budget is paced, not from adding more content.
- Genuine mechanism, not just reliability. It shows a real shift in Pass@N (ceiling) or in the reasoning error rate, not just PassRatio@N (reliability).
