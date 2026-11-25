// Proposal talk, 5 minutes. Content only: the theme supplies the styling.
// "Say:" comments give the spoken line and the time budget for each slide.

= Introduction / Motivation

== The question
// Say (0:40): accuracy gains are the usual evidence; we want a test of use, not of benefit.

Latent reasoning helps audio QA:
- Continuous "thoughts" before the answer raise accuracy (CoAT: Qwen2-Audio on MMAU *52.50% → 66.90%*).
- But the model *still hears the original audio* while it answers.
- Higher accuracy does not show that the answer *depends* on the latent. The latent may be bypassed.

*When an audio language model can still access the original audio, do its latent thoughts affect what it answers?*

== How we got here
// Say (0:50): we kept the detour because it gave us a baseline and two warning signs.

+ *Original idea:* the LLM writes what it heard into K latents, then answers from them (latent reasoning + a capacity bottleneck).
+ *What we built first:* a small external compressor (Q-Former-like) squeezes audio features into K slots for a *frozen* Qwen. It is easy to control and measure.
+ *The gap we noticed:* that is audio-token compression, not latent reasoning. The LLM never writes the latent.

#table(
  columns: 3,
  [], [*What we built*], [*What we meant to study*],
  [Who writes the latent], [external compressor], [the LLM itself],
  [Knows the question?], [only if we feed it the stem], [already read it],
  [LLM trained?], [frozen], [LoRA],
)

== What the detour taught us
// Say (0:50): keeping evidence and using it are different, so we measure use directly.

- *A few vectors are enough.* Official Clotho-AQA test, 514 yes/no questions, reader sees only the slots: text only 54.3%, K=4 69.0%, *K=16 79.6%*, full audio 82.5%.
- *Readable ≠ used.* On controlled one- vs two-segment clips, a linear probe reads the cue from K=16 latents *89–90%* of the time on held-out sources. The frozen decoder answers "yes" to all 100 clips.
- *A small compressor can't pick evidence.* Feeding it the question gave no question-specific gain on Clotho. A synthetic multi-event task showed a signal at K=4/16 but failed our pre-registered bar.

→ Keeping evidence in a latent and using it are different. *Measure use directly.*

// Clotho numbers: locked plan, paired bootstrap over recordings.
// Probe and question-guidance results are development-set findings.

= Related Works

== Related works
// Say (0:35): CoAT reports gains from adding the block; we ask whether the answer moves when the latent changes.

- *Latent reasoning:* Coconut (continuous thoughts from last-layer hidden states); CoAT (audio latent thoughts, audio kept in context).
- *Does reasoning look at audio?* SPARE: long reasoning chains may pull attention away from the audio.
- *Compressing into K tokens:* BLIP-2 Q-Former, Flamingo resampler (external); VoCo-LLaMA (inside the LLM).
- *Decodable ≠ used:* probing with control tasks. A probe finding information does not mean the model relies on it.

Our angle: *freeze a trained model and replace only its latent*, an ablation of the latent itself rather than of a training component.

= Methods

== One writer, three readers
// Say (0:45): same writer in all three; only the reader's access to audio changes.

- *Pass 1 (writer):* [audio] [question + options] [start] [K learned queries] → last-layer hidden states = *K latents*
- *Pass 2 (reader):* [audio?] [question + options] [start] [K latents] [end] → answer-letter logits

#table(
  columns: 3,
  [*Reader*], [*Audio at answer time*], [*What it tells us*],
  [Soft], [always], [Is the latent bypassed? (main test)],
  [Hard], [never], [How much evidence K latents can carry],
  [Mid], [dropped with p = 0.5 in training], [Can training force reliance?],
)

The bottleneck is a *tool*: it controls which path audio can take to the answer. Qwen2-Audio-7B with LoRA, trained on AVQA.

== How we test whether the latent is used
// Say (0:40): the plain-LoRA and K=0 controls separate "the latent helps" from "fine-tuning helps".

Fix the trained model, audio, question and options. *Replace only the latent:* normal, swap (another item's latent), dataset mean, zero, added noise. For soft/mid readers, also remove the audio.

#table(
  columns: 2,
  [*Observation (soft reader)*], [*Reading*],
  [answers unchanged under swap / mean / zero], [latent is *bypassed*: gains come from fine-tuning],
  [accuracy drops, answers move with the latent], [latent is *used* even with audio available],
  [mid reader moves more than soft], [audio dropout *creates* reliance],
)

Controls: plain LoRA (no latent), K=0 (start/end tokens only), writer blind to the question, soft + CoAT-style reconstruction loss.

// Both option orders, with writer and reader using the same order. Text-only baseline reported alongside.

= Next Steps

== Status and next steps
// Say (0:30): close by repeating the question; the first run tells us whether the soft reader bypasses the latent.

*Ready*
- Data: AVQA train 3,991 / dev 500; MMAU test-mini 1,000 for development; *MMAR held out* until the plan is locked.
- Base Qwen2-Audio on MMAU: *65.1%* with audio, 41.9% text-only.
- Pipeline and latent interventions run end to end.

*Next*
+ First run: seed 17, K=4, all readers and controls.
+ Lock success criteria in a plan; add seeds and K = 1 / 4 / 16.
+ Evaluate MMAR once.
+ Later: do the latents carry *multi-step* reasoning, beyond audio evidence?

_Early hint, not evidence:_ in a 40-question, 600-step smoke test, the soft reader's answers did not change under swap, mean or zero latents.
