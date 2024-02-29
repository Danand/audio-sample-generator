# Audio Sample Generator

This app can generate new audio samples using the Stable Diffusion model (LoRA) for training.

Steps:

1. Convert existing audio samples to spectrograms.
2. Train a small Stable Diffusion model (LoRA) on spectrograms.
3. Generate new audio samples with a specified prompt.

<!-- installation: begin -->

## Prerequisites

- [Python](https://www.python.org/downloads)

## How to install

```bash
git clone git@github.com:Danand/audio-sample-generator.git
cd audio-sample-generator
chmod +x run.sh
```

## How to launch

```bash
./run.sh
```

<!-- installation: end -->

## How to use

Simply follow all pages from the sidebar sequentially.

_Advanced settings are skipped here for convenience._

### Extract Spectrograms

1. Open audio files.
2. Click the **Extract** button.
3. Review the spectrograms extracted from the audio files.
4. Proceed to the next page.

### Prepare Dataset

1. Specify for each spectrogram:
   - **Subject**
   - **Caption** (comma-separated keywords)
   - _Optional:_ **Weight**
2. Click the **Save** button.

### Train LoRA

1. Click the **Train** button.

### Generate Audio with Stable Diffusion

1. Type in the **Prompt**.
2. Specify the **Amount** of audio to generate.
3. Click **Generate**.
4. Listen and save the generated samples if desired.

## Extras

### Batch Convert to Audio

That page is convenient for batch converting spectrograms to audio samples. You can experiment with any images of the respective size, not necessarily spectrograms.
