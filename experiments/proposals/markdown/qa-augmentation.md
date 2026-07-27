QA-augmentation: restating facts as question–answer pairs. Known result: facts a model has "read" but never seen in QA form can be memorized but not extractable — 0% recall until augmented

## Evidence

[Physics of Language Models: Part 3.1, Knowledge Storage and Extraction (Allen Zhu & Li, 2023)](https://www.google.com/url?q=https://arxiv.org/abs/2309.14316&sa=D&source=editors&ust=1784950268206448&usg=AOvVaw2nBy5IIyaJjgLOOCYvScTY)

- Built a synthetic dataset of 100k fake people with fake biographies (birthdate, city, university, employer, etc.) so the researchers knew exactly what was and wasn't in the training data no risk of the model having seen the question before.
- Core finding: a model can hit 99%+ next-token accuracy reciting a biography verbatim, and still score 0% answering "Where was this person born?" about that same person. The fact got memorized as a fixed string, not stored in a way that's queryable from a different angle.
- The fix was knowledge augmentation showing the same fact multiple ways during pretraining:
- multiplicity: generate 2-5 differently worded biography entries per person
- permute: shuffle the order of the sentences within an entry
- fullname: replace pronouns with the person's full name every time, not just once
- They used two probing techniques to explain why this works:
- P probing: checks how early in a biography the model has already "figured out" an attribute, using its position in the sentence as the signal
- Q probing: checks whether the attribute is linearly readable straight off the hidden state of the person's name alone (no context needed)
- With enough augmentation, Q probing accuracy approaches 100%; the model learns to attach every attribute directly to the name, rather than inferring it from surrounding sentence structure. That’s what makes the fact retrievable by a differently phrased question later, including QA style ones. They also showed the effect holds when a small "celebrity" subset of the corpus is heavily augmented with augmentation on some entities improves extraction accuracy for other, non-augmented entities too, because it shapes how the model tends to store any name attribute pair.

[Adesope, Trevisan & Sundararajan 2017 Rethinking the Use of Tests](https://www.google.com/url?q=https://www.researchgate.net/publication/315706448_Rethinking_the_Use_of_Tests_A_Meta-Analysis_of_Practice_Testing&sa=D&source=editors&ust=1784950268210697&usg=AOvVaw0PQebVdGt3hMJmWUngnNBn)

- Not about AI about how humans learn. Pooling across decades of studies, students who take a practice test on material before a final test consistently outperform students who only restudy the same material, with a mean weighted effect size of d = 0.74. The effect held across test formats, participant ages, and lab vs. classroom settings, though the size varied depending on those factors.
- Relevance here: this is the human cognition analog of what Allen Zhu & Li found in transformers that retrieval practice produces more usable, flexible storage of that fact than just re-exposure to it in its original form.

[Demystifying Synthetic Data in LLM Pre-training Kang et al., FAIR at Meta, 2025 1000 LLMs, >100k GPU hours](https://www.google.com/url?q=https://arxiv.org/abs/2510.01631&sa=D&source=editors&ust=1784950268212302&usg=AOvVaw3JZmgjS7qdvhfdkG_PQSVO)

- This paper supports "format of training data matters" but in a narrower way than the Allen Zhu & Li result, and it's the main source of tension for this lever.
- QA style rephrasing is one of the synthetic data types they test (alongside high quality rephrasing and textbook style generation). Findings that bear directly on this lever:
- Pure QA rephrased data alone does not clearly beat training on natural web text in some scaling regimes; it looks worse (highest irreducible loss among all mixtures tested, second only to pure CommonCrawl)
- The best performing setups mix a minority of synthetic/reformatted data with natural text; optimal ratios converge to ~30% synthetic, not full conversion to QA format
- Benefit is sensitive to model size and data budget, not a fixed universal ratio
- So this paper is evidence that "reformatting facts helps" but is a caution against "more QA reformatting is strictly better" or "convert the whole corpus to QA pairs." It's testing a different question than Allen Zhu & Li (general pretraining corpus efficiency at scale, not out-of-distribution fact extraction accuracy on a controlled entity set), which is likely why the two papers land in different places, worth keeping in mind when designing the experiment below.

[Cheng et al. 2024 - Instruction Pre-Training](https://www.google.com/url?q=https://aclanthology.org/2024.emnlp-main.148/&sa=D&source=editors&ust=1784950268215098&usg=AOvVaw1yeFua_fJAYYwwuCbXBnVK)

- Used a model to synthesize raw internet text into instruction response pairs via few-shot prompting, then pretrained on those instruction response pairs with ordinary next-token prediction instead of raw web text. Reported large gains in downstream multitask utility from this reformatting step. Directionally consistent with the QA augmentation lever, though it's converting general text into instruction format rather than restating known facts multiple ways, a related but distinct mechanism.

[How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data](https://www.google.com/url?q=https://arxiv.org/abs/2604.13977&sa=D&source=editors&ust=1784950268216630&usg=AOvVaw10WsuQoHxdI3JlDKsEw7WA)

- The design of rephrasing prompts primarily determines performance; structured pedagogical formats exceed the utility of simple paraphrasing; generator capacities beyond 1B parameters yield negligible gains for most rephrasing prompts; source quality has minimal impact when paired with robust mix in data; and generative diversity outweighs formatting consistency.
- While synthetic tokens provide logical depth, original web data remains essential to preserve commonsense reasoning and linguistic variety. These principles culminate in FinePhrase, a 486 billion token dataset that achieves up to a 30× cost reduction while outperforming established baselines.
- ![image](images/image1.png)

[FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale](https://www.google.com/url?q=https://arxiv.org/abs/2601.22146&sa=D&source=editors&ust=1784950268218535&usg=AOvVaw2uCMswh4DHt5WQDvf620Un)

- Took ~18M human user queries and turned them into templates to be used for synthetic instruction response generation, and they keep responses as 80+% grounded from the source text
- Given pretraining documents, they match them with compatible templates, meaning the document has enough information to instantiate the query and provide a grounded answer, before quality checking the instruction answer pairs with a judge model

![image](images/image4.png)

↑ Benchmark performance of 1.8B models pretrained on FineInstructions ↑

### Where the current evidence is strong

- Reformatting a known fact into multiple surface forms (including QA pairs) during pretraining measurably increases the model's ability to answer differently phrased questions about that fact later shown with a controlled synthetic entity set with no train/test leakage possible (0% to 96.6%)
- The mechanism has an explanation, not just a correlation: augmentation changes where the fact is stored (linearly on the entity's name, vs. distributed across surrounding tokens), verified via two independent probing methods
- The general principle retrieval/testing produces more flexible, retrievable memory than passive re-exposure replicates in an entirely separate domain (human learning science) across two independent meta-analyses

### Where the current evidence is conflicting, and issues exist

- Demystifying Synthetic Data finds pure QA format data does not clearly outperform natural web text at pretraining scale, and the best mixtures use QA/rephrased content as a minority (~30%) of the corpus; this pushes against a "convert everything to QA" version of this lever
- It's not established whether the benefit in Allen Zhu & Li is specifically about the QA format, or about phrasing diversity in general multiplicity and permutation (no QA structure at all) produced similarly large gains in their ablations, so QA pairs may not be doing anything a well-shuffled paraphrase set wouldn't also do
- Unclear whether "extractability of a fact under differently phrased questions" (the Allen Zhu & Li metric) and "general pretraining efficiency/loss" (the Demystifying Synthetic Data metric) are even measuring compatible things a lever could win on one and lose on the other

### Where the current evidence is weak

- Both controlled studies use small, synthetic, single-domain entity sets (100k fake people); no direct evidence yet on whether this transfers to a messy, real, multi-topic educational corpus at eduLLM's scale
- No evidence on optimal QA augmentation ratio or placement specifically for continued pretraining on an existing checkpoint (Allen Zhu & Li pretrain from scratch; OLMo continued pretraining is a different regime)
- Unclear whether QA augmentation's benefit is additive with the other P1 levers (e.g., Skill DAG ordering, DoReMi weighting) or largely redundant with them, since several levers touch "how facts are exposed to the model"

## Hypothesis

Null hypothesis: On a fixed set of N entities (same people and attributes in every condition), with matched per-attribute restatement counts (and matched name-mention counts), continued pretraining with QA-formatted restatements produces the same accuracy on held-out question phrasings as paraphrase-only restatements of those same facts. Under this null, surface-form diversity alone explains any gain, and QA structure adds nothing beyond it.Alternative hypothesis: On that same fixed entity set and matched restatement counts, QA-formatted restatements produce higher held-out QA accuracy than paraphrase-only i.e., QA structure itself adds value, not just rewording or covering more entities. Condition 1 (single-form prose) is a repetition/format floor and sanity check only. The primary claim is “QA beyond paraphrase” (condition 3 vs. condition 2). A win over condition 1 alone could just be more rehearsal. Why not “same total tokens per format” as the sole budget: QA pairs and prose paraphrases differ in length. Matching only total tokens can let the shorter format fit more people or more attribute exposures into the stream; evaluating on “everyone who appeared” then favors that format for the wrong reason. So we fix N (and the attribute set) first, match how often each of those facts is restated, and only then equalize any remaining step/token count with shared background or filler and not by adding extra entities to the shorter condition.

## Proposed experiment

Keep training an early OLMo-1B checkpoint on a fixed set of N fake people each with the same attribute schema.

Train/test split:

Train and test on the same people. Hold out new ways of asking the questions — not new people. (The old “never train on the test people” design would make every condition score near 0%, and wouldn’t test what we care about: answering questions about facts the model already saw.)

Fixed augmentation budget and controls

Choose N (and which attributes) so that both paraphrase and QA renderings of that full set fit inside the planned synthetic-token budget for every condition. Conditions always train on exactly those N people never “as many people as fit in T tokens,” which would give the shorter format more trained items.

- Condition 2 spends those R slots on paraphrases (reworded prose, permuted sentence order).
- Condition 3 spends those R slots on QA pairs
- Condition 1: the same N people, one canonical bio each, repeated R times (identical repeats zero diversity). Sanity floor only.

Without matching budget or fixed N, 2 and 3 just show each fact more often than 1 so a “win” could just be extra practice. Keep full-name use the same in conditions 2 and 3. Questions naturally use the full name; paraphrases often use “he/she.” Allen Zhu & Li found full names matter a lot — if we don’t match this, QA can “win” for the wrong reason. No fourth main condition that says “~30% of each person’s tokens are QA.” That mixes up Demystifying’s finding. How to split QA vs paraphrase within the budget is Follow-up A. How much of the whole stream is fake bios vs normal text is Follow-up B.

## Generation method (Stage-1)

Use simple templates/rules (Allen Zhu style): fill in names and facts into fixed bio / paraphrase / QA patterns. Hold out some question wordings for the test.

A Cheng-style / FineInstructions-style synthesizer adds dropped or invented attributes, variable multiplicity, pronoun vs. full-name skew, and quality differences across conditions — fighting the point of a synthetic entity set with known ground truth. Save model-based synthesis for Stage-2 on real edu text.

Shared mix with normal text

Every condition trains on:

- that condition’s fake-person docs, plus
- the same natural or educational background text (same amount for everyone).

That matches how we’d use this for an edu model, and avoids “train only on fake bios” weirdness. Change the overall fake-vs-real mix only in Follow-up B.

### Conditions (main experiment)

- Single-form prose (repetition floor): One canonical biography (fixed order, never reworded, never QA), same N people.
- Paraphrase only: Same N people; R paraphrase restatements per attribute; no QA.
- Same N people; R QA restatements per attribute (hard prose rule above).

There is no fourth main condition that caps QA at ~30% of an entity’s tokens. That misuses Demystifying’s corpus-level finding. Entity-level QA vs. paraphrase allocation is Follow-up A; corpus-level synthetic vs. natural mix is Follow-up B.

### What we measure:

- Main score: Exact-match accuracy on questions about trained people, using question wordings never seen in training. Score per fact; when estimating uncertainty, resample people (facts about the same person aren’t independent).
- Name probe (Q-probe): Can we read the fact from the model’s representation of the name alone? Checks storage, not just “got used to question style.”
- Bio recitation: Can the model still recite the original bio? Keeps the “memorized but can’t answer” gap visible (99% recite / 0% answer).

### Decision rules

- Main comparison: condition 3 vs. 2 (does QA beat paraphrase?).
- Condition 1: sanity check only. If 2 and 3 don’t both beat 1, the rewrite pipeline is broken.
- Stats: Compare conditions on the same people and same questions; report a confidence interval on the difference. Don’t just check whether separate score bars’ intervals overlap.

### Follow-up A: mix of QA vs paraphrase (if QA beats paraphrase)

If condition 3 beats 2, keep the same total budget/N people and vary how much of it is QA vs paraphrase (e.g. 0 / 25 / 50 / 75 / 100% QA). If QA does not beat paraphrase, prefer plain rewriting for P1 it’s simpler to make at scale.

### Success for Follow up A:

- A clear best mix (or a clear rising/peak pattern) not just “not clearly worse than the best”
- That best mix still beats pure paraphrase at the same budget
- No extra text beyond condition 3’s budget better split, not more data

### Follow up B: mix of fake bios vs normal text

Separately, vary how much of the training stream is fake-person data vs the shared edu/natural background (include a ~30% synthetic setting). Keep how each person’s facts are rewritten fixed. Report both question-answering and a general training-quality / usefulness check. Do not treat Follow-up A (% QA inside a bio) and Follow-up B (% synthetic in the whole mix) as the same dial.

Stage 2 real educational text: A win on fake bios is not enough for our eduLLM. If Stage 1 looks good (or paraphrase wins and we still care about product use), run Stage 2:

- Data: real educational passages
- Rewrites: Cheng-style and/or FineInstructions-style (grounded in the source text)
- Same idea: paraphrase vs QA at the same text budget; same background mix; test with new question wordings or essentially test with actual educational material
- Success: better answers about content from those edu passages, not only better scores on the fake bio test
