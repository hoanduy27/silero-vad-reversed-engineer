# Does the SileroVAD encoder carry speaker identity? (negative result)

Goal: check whether `VADRNN.encoder` (the 4 `SileroVadEncoderBlock`s, decoder RNN
bypassed via `VADRNN.forward_embed`) could double as a cheap speaker embedding,
before investing in any dedicated speaker-verification model. If the VAD's own
convolutional features already separated speakers, that would be a useful free
side effect; if not, don't bother reusing this encoder for that purpose.

## Setup

- Corpus: VIVOS (Vietnamese read speech, 16 kHz), via
  `silero/speaker_sim/download_vivos.py`.
- Subset: 46 speakers x 50 utterances (utterance-level run), 10 speakers x 5
  utterances (frame-level run) — see `egs/se/vivos/run.sh`.
- Embedding: `VADRNN.forward_embed(x)` — STFT -> encoder only, no decoder/LSTM.
  - `utterance` level: mean-pool the encoder's per-frame output over time -> one
    128-d vector per utterance.
  - `frame` level: keep every encoder time step as its own 128-d point, tagged
    with its parent utterance's speaker (`return_individuals=True`).
- Visualization: t-SNE to 2D (`silero/speaker_sim/plot_tsne.py`), colored by
  speaker.
- Cluster-vs-speaker check: KMeans (k=2) fit directly on the saved 2D t-SNE
  projection, then measured how much each cluster's points agree with speaker
  identity (`silero/speaker_sim/analyze_clusters.py`).

Reproduce with `cd egs/se/vivos && ./run.sh` (add `--level frame` for the
frame-level run); outputs land in `egs/se/vivos/exp/` (gitignored — rerun to
regenerate, nothing there is committed).

## Result

Both the utterance-level (2300 points, 46 speakers) and frame-level (10
speakers) t-SNE plots collapse onto a single continuous, winding curve with
every speaker's color fully interleaved along its whole length — not into
per-speaker islands. That shape is the classic t-SNE signature of one dominant
non-categorical factor (here, most likely something utterance-level like
duration/energy/phonetic content that correlates with position along the VAD's
own feature axis) rather than a set of discrete, separable classes.

The KMeans(k=2) check initially looks like a mixed signal — "22/46 speakers
have >=90% of their points in a single cluster" — but this is an artifact, not
evidence of speaker structure: since all the points lie on one continuous
curve, *any* 2-means split just cuts that curve in half by position, and
whichever arc a speaker's utterances happen to land on determines their
"cluster," regardless of speaker identity. A speaker whose few utterances
happen to cluster at one end of the curve will look "confined" to one cluster
by coincidence, not because the model encoded who they are. There is no region
of either plot where a single speaker (or small group of speakers) is visually
isolated from the rest.

## Conclusion

**No meaningful speaker-discriminative signal.** `VADRNN.encoder`'s
representation is not a usable speaker embedding as-is — expected, since
nothing in Silero VAD's training objective (speech/non-speech decision) gives
it a reason to preserve speaker identity, and if anything a VAD benefits from
being speaker-*invariant*. Don't reuse `forward_embed` for speaker
verification/diarization without dedicated speaker-discriminative fine-tuning
(e.g. a speaker-classification or metric-learning loss on top of the encoder);
treat this probe as ruling that shortcut out rather than something to retest.
