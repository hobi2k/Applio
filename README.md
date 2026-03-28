# ComfyUI Applio

[Applio](https://applio.org) voice conversion as ComfyUI custom nodes — a fork of [IAHispano/Applio](https://github.com/IAHispano/Applio).

Applio is a high-quality RVC (Retrieval-based Voice Conversion) tool. This fork exposes its full feature set as chainable ComfyUI nodes: voice conversion, audio effects, voice blending, TTS, and audio analysis.

> **Note:** Applio uses pre-trained speaker models — it is **not** zero-shot. You need a trained `.pth` model for the target voice.

---

## Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hobi2k/Applio.git
```

### 2. Install dependencies

```bash
cd Applio
pip install -r requirements.txt
```

`torch` / `torchaudio` / `torchvision` are **not** included in `requirements.txt` — they are managed by your ComfyUI environment.

### 3. Download prerequisite models (automatic)

On first use, the following models are automatically downloaded to `ComfyUI/models/rvc/`:

| File | Size | Purpose |
|------|------|---------|
| `predictors/rmvpe.pt` | ~200 MB | F0 prediction (default) |
| `predictors/fcpe.pt` | ~80 MB | F0 prediction (FCPE method) |
| `embedders/contentvec/pytorch_model.bin` | ~360 MB | Feature extraction |
| `embedders/contentvec/config.json` | — | Model config |

### 4. Place your voice models

```
ComfyUI/models/rvc/
├── voices/        ← trained .pth files go here
├── index/         ← .index files go here
├── predictors/    ← auto-downloaded
└── embedders/     ← auto-downloaded
```

---

## Nodes

### `Applio Model Loader`
Loads a trained speaker model. Reads from `models/rvc/voices/` and `models/rvc/index/` by default — filenames or absolute paths both accepted.

| Input | Type | Description |
|-------|------|-------------|
| `pth_path` | STRING | `.pth` checkpoint filename or absolute path |
| `index_path` | STRING | `.index` file (leave empty to skip) |
| `embedder_model` | COMBO | `contentvec` / `spin` / `spin-v2` / `chinese-hubert-base` / `japanese-hubert-base` / `korean-hubert-base` / `custom` |
| `embedder_custom_path` | STRING | Path when `embedder_model = custom` |

**Output:** `APPLIO_MODEL`

---

### `Applio Voice Convert`
Full RVC inference with all parameters.

| Input | Default | Description |
|-------|---------|-------------|
| `model` | — | From Model Loader |
| `audio` | — | Input audio |
| `pitch` | 0 | Semitone shift (−24 ~ +24) |
| `f0_method` | rmvpe | `rmvpe` / `crepe` / `fcpe` |
| `index_rate` | 0.75 | FAISS index blend (0 = off, 1 = full) |
| `volume_envelope` | 1.0 | Output volume matching |
| `protect` | 0.5 | Consonant protection (0.5 = max) |
| `hop_length` | 128 | F0 extraction hop length |
| `split_audio` | False | Split long audio before inference |
| `clean_audio` | False | Noise reduction post-process |
| `clean_strength` | 0.5 | Noise reduction strength |
| `f0_autotune` | False | Snap pitch to nearest note |
| `f0_autotune_strength` | 1.0 | Autotune blend |
| `formant_shifting` | False | Formant shift (gender/character) |
| `formant_qfrency` | 0.8 | Formant quefrency |
| `formant_timbre` | 0.8 | Formant timbre |
| `sid` | 0 | Speaker ID (multi-speaker models) |

**Output:** `AUDIO`

---

### Audio Effects (chainable)

All effects take `AUDIO` → `AUDIO` and can be chained freely.

| Node | Key Parameters |
|------|---------------|
| `Applio Reverb` | room_size, damping, wet_level, dry_level, width, freeze_mode |
| `Applio Chorus` | rate_hz, depth, center_delay, feedback, mix |
| `Applio Delay` | delay_s, feedback, mix |
| `Applio Compressor` | threshold (dB), ratio, attack_ms, release_ms |
| `Applio Gain` | gain_db |
| `Applio Limiter` | threshold_db, release_ms |
| `Applio Pitch Shift FX` | semitones |
| `Applio Distortion` | drive_db |
| `Applio Bitcrush` | bit_depth |
| `Applio Clipping` | threshold_db |

Powered by [Pedalboard](https://github.com/spotify/pedalboard).

---

### `Applio Voice Blender`
Blend two `.pth` models by ratio. Output model can be used directly in Voice Convert (no `.index` needed).

| Input | Default | Description |
|-------|---------|-------------|
| `output_name` | blended_model | Output filename (no extension) |
| `pth_path_a` | — | First model path |
| `pth_path_b` | — | Second model path |
| `ratio` | 0.5 | 0 = A only, 1 = B only |

**Outputs:** `message` (STRING), `output_pth_path` (STRING)

---

### `Applio TTS (EdgeTTS)`
Text-to-speech via Microsoft Edge TTS. Requires internet. Output can be piped directly into Voice Convert.

| Input | Default | Description |
|-------|---------|-------------|
| `text` | — | Input text (multiline) |
| `tts_voice` | en-US-AriaNeural | Voice name |
| `speed` | 0 | Speed offset in % (−100 ~ +100) |

**Voice examples:** `en-US-AriaNeural`, `ja-JP-NanamiNeural`, `ko-KR-SunHiNeural`, `zh-CN-XiaoxiaoNeural`

Full voice list: `python -m edge_tts --list-voices`

**Output:** `AUDIO`

---

### `Applio Audio Analyzer`
Spectrogram + waveform + spectral features.

| Input | Description |
|-------|-------------|
| `audio` | Any AUDIO |

**Outputs:** `info` (STRING with stats), `spectrogram` (IMAGE)

---

## Example Workflows

Ready-to-use workflow JSON files are in the [`workflows/`](workflows/) folder. Load them via **ComfyUI → Load**.

| File | Description |
|------|-------------|
| [`01_basic_voice_convert.json`](workflows/01_basic_voice_convert.json) | Load audio → voice convert → preview + save |
| [`02_tts_voice_convert.json`](workflows/02_tts_voice_convert.json) | EdgeTTS → voice convert → preview + save |
| [`03_voice_convert_effects_chain.json`](workflows/03_voice_convert_effects_chain.json) | Voice convert → Compressor → Reverb → Limiter → save |

**Workflow 01 — Basic voice convert**
```
LoadAudio ──────────────────────────┐
                                    ├→ ApplioInfer → PreviewAudio
ApplioModelLoader (pth + index) ────┘             → SaveAudio
```

**Workflow 02 — TTS → voice convert**
```
ApplioTTS (text → speech) ──────────┐
                                    ├→ ApplioInfer → PreviewAudio
ApplioModelLoader ──────────────────┘             → SaveAudio
```

**Workflow 03 — Effects chain**
```
LoadAudio ──┐
            ├→ ApplioInfer → ApplioCompressor → ApplioReverb → ApplioLimiter → PreviewAudio
ModelLoader ┘                                                                → SaveAudio
```

---

## Project Structure

```
custom_nodes/Applio/
├── __init__.py          # ComfyUI entry point
├── nodes.py             # 15 ComfyUI nodes
├── requirements.txt     # Dependencies (no torch)
└── applio/              # Applio backend (IAHispano/Applio fork)
    ├── rvc/             # RVC inference, pipeline, models
    ├── tabs/            # Gradio UI tabs (not used by nodes)
    ├── app.py           # Original Gradio app
    └── ...
```

---

## Credits

- [IAHispano/Applio](https://github.com/IAHispano/Applio) — original Applio (MIT License)
- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — RVC architecture
- [Spotify Pedalboard](https://github.com/spotify/pedalboard) — audio effects
- [Microsoft Edge TTS](https://github.com/rany2/edge-tts) — TTS backend

See [applio/LICENSE](applio/LICENSE) and [applio/TERMS_OF_USE.md](applio/TERMS_OF_USE.md) for licensing details.
