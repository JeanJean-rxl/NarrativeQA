# NarrativeQA

### 🧠 Project Overview

Test project explores retrieval-augmented generation (RAG) under the lens of Floridi’s scope-certainty trade-off published on 11/06/2025: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5289884

<pre>
**Key Conjecture**  
There exists a constant *k* > 0, independent of any specific system, such that every sufficiently expressive AI mechanism *M* satisfies:  

**1 − C(M) · S(M) ≥ k**
</pre>


### 📊 Experiment Findings
Dataset: https://huggingface.co/datasets/deepmind/narrativeqa

Baseline Updated on 13/06/2025:

Without fine-tune, 8 runs across two models (flan-t5-base, flan-t5-large) and four chunk sizes (64–512):
	•	Larger scope does not guarantee better answers — performance plateaus or declines as context grows.
	•	Model scaling brings marginal gains, but insufficient to offset poorly scoped input.


### ❓ TODO

1.	Control (- We are here -)
Establish a high-certainty baseline under minimal scope using NarrativeQA data.
→ How certain can a system be when it knows little?

2.	Extend
Expand contextual scope while attempting to preserve certainty.
→ How much more can it know without becoming uncertain?

3.	Understand
How and why certainty collapses...
