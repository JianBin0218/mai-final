#set text(
  font: "New Computer Modern Sans"
)

#import "@preview/touying:0.7.3": *
#import themes.stargazer: *


#show: stargazer-theme.with(
  aspect-ratio: "16-9",
  config-common(new-section-slide-fn: none),
  config-info(
    title: [Do Audio Language Models Use Their Own Latents?],
    subtitle: [Team 29 · MAI 2026 Final Project Proposal],
    author: [Cheng-Hsian Hsieh,  Yu-Syuan Lee, Jian-Bin Wu],
    date: "2026-11-26",
    institution: [Dept. of Computer Science & Information Engineering, National Taiwan University],
  ),
)

#set heading(numbering: none)

#let ink-blue = rgb("#075a9e")
#let pale-blue = rgb("#edf5fc")
#let pale-gray = rgb("#f2f4f6")
#let accent(body) = text(fill: ink-blue, weight: "bold", body)
#let panel(tone, body) = block(width: 100%, fill: tone, inset: (x: 12pt, y: 9pt), radius: 4pt, body)

// 5-minute proposal. Timing includes the title and outline slides (total about 4:35).
// Say (0:10): We study whether audio language models actually use the latent vectors they write before answering.
#title-slide()

// Say (0:05): Introduction, related work, our setup and pilot results, and next steps.
#outline-slide()

= Introduction

== Motivation
// Say (0:30): Latent reasoning lets a model think in continuous vectors instead of text. CoAT brings this to audio and improves MMAU a lot (CoAT, Table 1). But its reader still sees the original audio, so the gain may come from the latents, or from fine-tuning while the model keeps reading the audio. Accuracy alone cannot tell these apart.

*Latent reasoning:* the model writes continuous vectors before answering, instead of a text chain of thought.

#panel(pale-blue)[
  #accent[CoAT on MMAU] · Qwen2-Audio accuracy *52.50% → 66.90%*
]

#align(center, image("figures/fig-motivation.svg", width: 90%))

== Research Question and Contributions
// Say (0:25): Our question: when direct audio is available, does an audio model use its own latents? Our pilot already suggests the answer is no, the same latent bypass reported for images.

#panel(pale-blue)[
  *Research question.* When direct audio is available, do audio language models use their own latents to answer?
]

+ *Causal latent test for audio.* Fix the trained model and all inputs; change only the latents.
+ *Controls that separate fine-tuning from latents.* Plain LoRA, K = 0, and a reconstruction loss.
+ *Reader conditions.* Compare readers with full, partial, or no direct audio.

*Pilot finding:* audio latent reasoning shows the same bypass reported in vision.

= Related Work

== Latent Reasoning in Vision
// Say (0:25): In vision, several papers already intervene on latents and find they are bypassed. Two lines of fixes exist: making latents necessary by removing the image at answer time, and optimizing latents at inference time. We test whether the same bypass holds for audio.

#text(size: 0.85em)[
#table(
  columns: (1.1fr, 1fr), inset: (x: 8pt, y: 5pt),
  [*Work*], [*Finding*],
  [Imagination Helps (ICML 2026)], [Changing latents barely changes answers],
  [What's Holding Back Latent Visual Reasoning?], [Zero / noise / random latents: accuracy almost unchanged],
  [Latent Visual Reasoning in Silence (NeurIPS 2026)], [Replacing or removing latents barely hurts],
  [Visual Latents Know More (NeurIPS 2026)], [Silent latents; fix by inference-time optimization],
  [Reason Through the Latent! (CVRR)], [Removes image states so latents become necessary],
)
]

== Latent Reasoning in Audio
// Say (0:20): Audio has latent reasoning methods with accuracy gains, but none of them fixes the model and changes only the latents. That is the gap we fill.

#text(size: 0.9em)[#table(
  columns: (auto, 1fr, auto), inset: (x: 8pt, y: 5pt),
  [*Work*], [*Approach*], [*Latent intervention*],
  [CoAT], [Continuous thinking positions; Qwen2-Audio], [No],
  [JELAR], [WavJEPA features as latent targets; Audio-Reasoner], [No],
  [DeALR], [Training-free latent optimization; Qwen2-Audio, Qwen2.5-Omni], [No],
  [SPARE (Interspeech 2026)], [Attention drifts from audio during long reasoning], [No],
)]

#panel(pale-blue)[
  *Gap.* None of these audio studies fixes a trained model and intervenes only on its latents.
]

= Methodology and Experiments

== Two-Pass Write–Read Architecture
// Say (0:35): Qwen2-Audio-7B-Instruct writes latents in the first pass and answers in the second, with one shared LoRA (q and v projections, rank 8). Three readers share the architecture and are trained separately; they differ only in direct audio. Soft matches CoAT and is our main test.

#align(center, image("figures/fig-architecture.svg", width: 76%))

#table(
  columns: (auto, 1fr), inset: (x: 8pt, y: 5pt),
  [*Reader*], [*Direct audio at the reader*],
  [*Soft · main test*], [Available, as in CoAT],
  [Hard], [Removed; audio reaches the answer only through latents],
  [Mid], [Masked with p = 0.5 in training; evaluated with and without],
)

== Interventions, Controls, and Data
// Say (0:25): After training we fix the model and all inputs and change only the latents, four ways. We report how many answers change and how accuracy changes. Plain LoRA and K = 0 show how much of the gain needs latents at all. Each question is scored in the original and reversed option order, then averaged.

#align(center, image("figures/fig-interventions.svg", width: 78%))

#panel(pale-gray)[#accent[Controls] · Plain LoRA (no latents) · K = 0 (boundary tokens only) · reconstruction loss · question-blind writer]

#table(
  columns: (auto, 1fr), inset: (x: 8pt, y: 5pt),
  [*Split*], [*Data*],
  [Train], [AVQA-R1-6K, 3,991 questions],
  [Development], [AVQA dev 500 · MMAU test-mini 1,000],
  [Held out], [MMAR, final evaluation only],
)

== Pilot Results: Latent Bypass with Direct Audio
// Say (0:35): Soft scores the same as plain LoRA and K = 0, so the gain comes from fine-tuning, not from latents. All four interventions change at most 0.9% of answers, and accuracy moves by at most 0.1 points. Even latents forced to encode the audio are ignored. This mirrors the bypass found in vision. This is a pilot: one seed, K = 4, development evaluation.

#table(
  columns: (1fr, auto, auto), inset: (x: 9pt, y: 6pt), align: (left, center, center),
  [*Model (MMAU test-mini)*], [*Accuracy*], [*Answers changed by interventions*],
  [Original Qwen2-Audio], [65.1%], [—],
  [Plain LoRA, no latents], [66.8%], [—],
  [K = 0, boundary tokens only], [66.0%], [—],
  [*Soft*], [*66.0%*], [*0.1–0.9%*],
)

#panel(pale-blue)[
  With a reconstruction loss the latents encode the audio, yet interventions still change only *0.7–1.2%* of answers.
]

== Pilot Results: Reader Conditions
// Say (0:30): Same pilot setting; a swap means latents written for another question. Once direct audio is removed, the swap matters. Hard answers well from latents alone and collapses under a swap. Mid keeps using latents even when audio is present. But the writer sees the question and options, so reliance does not yet show that latents carry audio evidence.

#table(
  columns: (1fr, auto, auto, auto), inset: (x: 9pt, y: 6pt), align: (left, center, center, center),
  [*Reader at evaluation*], [*Accuracy*], [*After latent swap*], [*Answers changed*],
  [Soft · audio], [66.0%], [+0.1 pt], [0.2%],
  [Mid · audio], [63.6%], [−2.9 pt], [19.7%],
  [Mid · no audio], [63.8%], [−33.6 pt], [67.5%],
  [Hard · latents only], [66.5%], [−40.1 pt], [74.0%],
)

#panel(pale-gray)[
  *Reliance is not yet evidence.* The writer sees the question and options, so latents may relay a chosen answer.
]

= Next Steps

== Proposed Experiment: Paired Evidence Swap
// Say (0:35): Next, we ask what the latents carry. We fix the reader's audio and question, and swap in latents written from a paired clip with the opposite answer. If the score moves toward the source clip's answer, the latents carry evidence. A question-blind writer and independent option orders rule out answer shortcuts.

#align(center, image("figures/fig-paired-swap.svg", width: 76%))

#panel(pale-blue)[
  *Measure:* does the Yes-minus-No score move toward the source clip's answer?
]

*Answer-shortcut checks:* a question-blind writer and independent option orders at writer and reader.
