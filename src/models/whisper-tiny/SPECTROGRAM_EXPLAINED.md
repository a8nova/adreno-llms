# The Spectrogram — Sound Made Visible

> *Why a neural net "sees" a picture of your voice, what a mel bin actually is,
> why we pick 80 of them, and the 350-year idea that ties sound to light.*

A companion to `WHISPER_ARCHITECTURE.md` §2.1 (the front-end that turns audio into
the `[80, 3000]` log-mel image Whisper consumes).

---

## 1. The one big idea: a wave is a sum of pure tones

A microphone records **one number over time** — air pressure. Plotted, it's a wiggly
line (a *waveform*):

```
pressure
  +1 │      ╭╮      ╭╮      ╭╮            a "waveform" — amplitude vs TIME.
   0 │─╮╭──╮╯╰╮╭──╮╯╰╮╭──╮╯╰─   You can SEE loudness, but not PITCH/timbre.
  -1 │ ╰╯  ╰─ ╰╯  ╰─ ╰╯
     └────────────────────────▶ time
```

The waveform hides the thing we care about for speech: **which frequencies are
present, and how strong each one is.** A vowel like "ee" vs "oo" looks like a similar
squiggle but is a totally different *mix of frequencies*.

**Fourier's theorem** (Joseph Fourier, 1822): *any* signal, however complicated, can
be rebuilt by adding together **pure sine waves** of different frequencies, each with
its own strength. So instead of describing sound as "pressure over time," we can
describe it as "**how much of each frequency** is in it." That recipe — strength per
frequency — is the **spectrum**.

```
   one complex sound      =      sum of pure tones (its spectrum)

      ╱╲╱╲╱╲                       ───  low freq, strong   ████
     ╱      ╲      ≈   add up:     ∿∿∿  mid freq, medium   ██
    ╱  messy  ╲                    ∿∿∿∿∿ high freq, weak    █
```

The math gadget that takes a chunk of waveform and returns its spectrum is the
**Fourier Transform** (in practice the **FFT** — Fast Fourier Transform, Cooley &
Tukey, 1965; the trick was hiding in Gauss's notebooks from ~1805).

---

## 2. What a "bin" is

The FFT doesn't give you a smooth, infinitely-detailed spectrum — it gives you a
**finite set of buckets**, and each bucket holds the energy in a small band of
frequencies. **Each bucket is a "bin."**

Think of it like a row of jars, each catching a different slice of pitch:

```
 frequency →   0 Hz        1 kHz        2 kHz        ...        8 kHz
              ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
 FFT bins:    │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │ b8 │ b9 │ ...
              └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
                ▲                                          each jar's "fill level"
            energy in                                      = how much sound energy
            this band                                        sits in that band
```

If you run a 400-sample FFT (Whisper's window), you get **201 raw frequency bins**
spanning 0 → 8000 Hz (half the 16 kHz sample rate — the *Nyquist limit*, the highest
frequency you can represent). Bin spacing is linear: each raw bin is ~40 Hz wide.

**A bin is just a discrete frequency channel.** "More bins" = finer frequency
resolution = a sharper picture, but more data and more compute.

---

## 3. From a spectrum to a SPECTROGRAM (add time back)

One FFT describes one short instant (~25 ms). Speech changes constantly, so we slide a
short window along the audio and take an FFT at each step:

```
 waveform  ▁▂▃▅▇▅▃▂▁▂▃▅▇▆▄▂▁▃▅▇▇▅▃▁ ...
            └─┘ └─┘ └─┘ └─┘ └─┘          overlapping ~25ms windows,
            FFT FFT FFT FFT FFT          hop ~10ms apart
             │   │   │   │   │
             ▼   ▼   ▼   ▼   ▼
          spectrum per window, stacked side by side →
```

Stack those spectra as **columns**, with frequency going **up** and time going
**right**, and color each cell by energy → a **spectrogram**: a 2-D image of sound.

```
        a spoken word as a spectrogram  (▓ loud · ░ quiet)
 freq
  ▲  8k │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← hiss of "s","f" lives up here
  │  4k │░░░▒▒░░░░░░▒▒▒▒░░░░░░░░░▒▒▒░░░
  │  2k │░▒▓▓▒░░░▒▓▓▓▓▓▒░░░▒▒▓▓▓▓▓▒░░   ← formants: the bands that make
  │  1k │▒▓▓▓▓▒░▒▓▓▓▓▓▓▓▒░▒▓▓▓▓▓▓▓▓▒░     a vowel sound like "ah" vs "ee"
  │ 500 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒   ← voice pitch / its harmonics
  │   0 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     └──┴───────────────────────────▶ time
          "M  i  s  t  e  r    Q  u  i  l ..."
```

This is *exactly* the representation Whisper's encoder reads. Speech recognition
became a **computer-vision problem**: recognize patterns in an image. You can almost
*read* a spectrogram — vowels are stacks of horizontal bands (formants), "s"/"sh" are
bright clouds high up, silences are dark.

---

## 4. The mel scale — and why **80** bins

### Human hearing is logarithmic, not linear

You easily hear the difference between 200 Hz and 300 Hz (a musical interval). But
6000 Hz vs 6100 Hz? Same 100 Hz gap — yet it's nearly inaudible. **We perceive pitch
logarithmically:** doubling the frequency sounds like "one step up" whether you go
100→200 or 2000→4000.

A linear FFT wastes resolution: it spends just as many bins on 7000–8000 Hz (where we
barely care) as on 200–1200 Hz (where vowels and speech live). The fix is the **mel
scale** (Stevens, Volkmann & Newman, 1937) — a frequency axis warped to match
*perceived* pitch. Equal steps in mels sound like equal steps in pitch.

### The mel filterbank — squashing 201 raw bins into 80 mel bins

We don't FFT directly into mel bins. We take the 201 linear FFT bins and **group them
with triangular weighting functions** — wide triangles at high frequency (lump lots of
linear bins together), narrow triangles at low frequency (keep detail where it
matters). Each triangle's output = one **mel bin**.

```
 weight
   1 │   ╱╲    ╱╲     ╱──╲       ╱────╲          ╱──────╲
     │  ╱  ╲  ╱  ╲   ╱    ╲     ╱      ╲        ╱        ╲     ← 80 triangular
     │ ╱    ╲╱    ╲ ╱      ╲   ╱        ╲      ╱          ╲      mel filters
   0 │╱      ╳     ╳        ╲ ╱          ╲    ╱            ╲
     └────────────────────────────────────────────────────▶ frequency (Hz)
       narrow & dense          getting wider & sparser →
       (low freq, much detail)         (high freq, coarse)
        mel 0,1,2,3 ...                          ... mel 79
```

Each mel bin = (FFT energies) weighted by one triangle, summed. So **80 mel bins** is
a *perceptually smart compression* of the 201 raw bins down to 80 numbers per time
step — keeping detail where the ear (and speech) needs it, throwing it away where it
doesn't.

### Why 80 specifically?

It's an engineering sweet spot, not a law of nature:

| Mel bins | Used by | Trade-off |
|----------|---------|-----------|
| 13–23 | classic MFCC speech recognition (1980s–2010s) | tiny, fast, loses detail — fine for phone-quality ASR |
| 40 | early deep-learning ASR, smaller Whisper-era models | good balance |
| **80** | **Whisper, most modern ASR/TTS** | rich enough to capture fine spectral structure; still compact |
| 128 | some high-fidelity audio / music models | more detail, more compute, diminishing returns for speech |

80 captures the formants, harmonics, and fricative detail that distinguish phonemes,
while keeping the encoder input small (80 × 3000 ≈ 240k numbers for 30s, vs millions of
raw samples). More bins past ~80 buys little for speech but costs memory and compute —
so 80 became the de-facto standard. **In this port it's `NUM_MEL_BINS = 80`** and the
encoder's first conv reads exactly those 80 channels.

### "log"-mel: one last human-hearing trick

We also take the **logarithm** of each mel energy. Loudness perception is logarithmic
too (that's what decibels are). A whisper and a shout differ by a factor of millions in
raw energy; log compresses that into a comfortable range the network can learn from.
Hence **log-mel spectrogram**.

---

## 5. Sound *and* light: the 350-year-old idea you sensed

Your intuition is exactly right — **"spectrum" is one idea that spans sound and light.**
Here's the through-line:

- **1666 — Isaac Newton** passes sunlight through a glass prism and sees it fan out into
  red→violet. He borrows the Latin word **_spectrum_** ("appearance / apparition") for
  the band of colors. A prism **physically sorts light by its frequency** (wavelength).
  This is the *first* spectrum.

- **1822 — Joseph Fourier** proves *mathematically* that any vibration is a sum of pure
  frequencies. This makes "spectrum" a general concept: not just colors, but the
  frequency recipe of *anything that oscillates*.

- **light and sound are both waves.** Light is an electromagnetic wave; sound is a
  pressure wave in air. Different physics, **same math**. "Frequency" means color for
  light and pitch for sound. A **prism is to light what a Fourier transform is to
  sound** — both decompose a wave into its constituent frequencies.

```
        LIGHT                                  SOUND
   white light                            complex sound
        │                                       │
     ┌──┴──┐ prism                          ┌───┴───┐ Fourier transform
     ▼     ▼                                ▼       ▼
   red ... violet                        bass ... treble
   (low → high frequency)               (low → high frequency)
        =  SPECTRUM  =                        =  SPECTRUM  =
```

So a spectrogram is, quite literally, the **"colors" of your voice** plotted over
time. A rainbow is a single light spectrum; a spectrogram is a *movie* of sound
spectra. Same concept Newton named, Fourier generalized, and your phone now computes
60–100 times a second.

- **1940s — the spectrograph.** During WWII, engineers at **Bell Telephone
  Laboratories** built the **sound spectrograph**, a machine that drew spectrograms on
  paper to make speech visible. **Ralph Potter, George Koenig, and Harriet Green Kopp**
  published it in 1947 as **_Visible Speech_** — the direct ancestor of the image
  Whisper reads. (Astronomers, meanwhile, had been reading *light* spectrograms for a
  century to discover what stars are made of — same instrument idea, other end of the
  wave spectrum.)

So when this port runs an FFT and a mel filterbank on your audio, it's standing on a
chain that runs **Newton's prism → Fourier's theorem → Bell Labs' spectrograph →
the FFT → a neural net.** You're not just preprocessing audio; you're taking the
spectrum of a sound the same way a prism takes the spectrum of light.

---

## 6. The pipeline in this codebase (where to look)

`src/mel_frontend.cpp` does steps 1–4 above, on-device, in C++:

```
raw waveform [16 kHz mono]
   │  frame: slide a 400-sample (25ms) window, hop 160 samples (10ms)
   ▼
windowed frames
   │  FFT each frame → 201 linear frequency bins (power)
   ▼
linear power spectrum [201, T]
   │  multiply by 80 triangular MEL filters, sum
   ▼
mel spectrogram [80, T]
   │  log10 + normalize
   ▼
LOG-MEL [80, 3000]   ──►  Whisper encoder  (WHISPER_ARCHITECTURE.md §2.2)
```

Validated at **cosine 1.0** against HuggingFace `WhisperProcessor`, and it costs only
~0.14s — the cheap, elegant front door to the whole model.

---

### One-paragraph summary

A **spectrogram** turns sound into a picture by chopping audio into tiny overlapping
windows and, for each, computing its **spectrum** (the strength of every frequency) via
the **Fourier transform** — stacking those spectra over time. A **bin** is one discrete
frequency bucket. Because human hearing is logarithmic, we re-bucket the raw FFT bins
into **80 mel bins** (triangular filters, dense at low pitch, coarse at high) and take
a log — giving the compact, perceptually-tuned **log-mel** image that modern speech
models read. The whole idea is the same "**spectrum**" Newton saw in a prism in 1666
and Fourier turned into math in 1822: light and sound are both waves, and a spectrum is
simply a wave broken into its pure frequencies.
