# Which parts run where

One repository, three layers. Not three repositories: the pipeline below is
identical on every machine, and duplicating it to keep the platforms apart
would mean fixing every bug twice and finding out about the second copy later.

```
   ┌─────────────────────────────────────────────────────┐
   │  CAPTURE            Windows only, needs the adapter │
   │  audio.py  hotkey.py  tray.py  notify.py            │
   └────────────────────────┬────────────────────────────┘
                            │  16 kHz mono audio
   ┌────────────────────────▼────────────────────────────┐
   │  PIPELINE                    portable, no OS calls  │
   │  vad.py  transcribe.py  diarize.py                  │
   │  extract.py  fields.py  workorder.py                │
   └────────────────────────┬────────────────────────────┘
                            │  work order text
   ┌────────────────────────▼────────────────────────────┐
   │  DELIVERY               popup, clipboard, output/   │
   └─────────────────────────────────────────────────────┘
```

| Layer | Files | Runs on |
|---|---|---|
| **Capture** | `audio.py`, `hotkey.py`, `tray.py`, `notify.py` | Windows, at the phone |
| **Pipeline** | `transcribe.py`, `diarize.py`, `extract.py`, `vad.py`, `fields.py`, `workorder.py` | anywhere — Windows, macOS, Linux, CPU or GPU |
| **Plumbing** | `config.py`, `storage.py`, `pipeline.py`, `models.py`, `update.py`, `accel.py` | anywhere |

The capture layer is the only Windows-specific code, and only because the
LRX-40USB is plugged into the office PC. Everything expensive is in the
portable middle, which is why it can be moved to a faster machine without the
capture following it.

## Three ways to arrange it

### 1. Standalone

Everything on one machine. What the office PC does today, and what a Windows
box with an RTX 3090 does best — capture is cheap, and the pipeline gets the
GPU.

```toml
# nothing to set; this is the default
```

### 2. Split extraction

Capture and transcription stay at the phone; the language model runs
elsewhere. **One line of config, no code, works today** — Ollama is already an
HTTP service and this already talks to it over HTTP.

```toml
[extract]
base_url = "http://192.168.1.50:11434"
allow_lan = true
```

On the machine doing the work: `OLLAMA_HOST=0.0.0.0 ollama serve`.

This moves the single heaviest stage — three minutes of an eleven-minute call
on the old office PC — and costs nothing to try.

### 3. Split everything

Capture at the phone, pipeline on the fast machine. Needs a service that
accepts audio and returns a work order, which does not exist yet. Worth
building only once (2) has been measured and the remaining wait still matters.

## Choosing an accelerator

`provider` under `[transcribe]` and `[diarize]` selects where the ONNX models
run: `cpu`, `cuda`, `coreml`, or `auto`.

Everything chosen automatically falls back to the CPU rather than failing. A
graphics driver without the runtime libraries behind it is the normal state of
a machine nobody set up for GPU work, and it is not a reason to refuse to
transcribe a call.

| | Windows + NVIDIA | Mac (Apple Silicon) | CPU only |
|---|---|---|---|
| Whisper | **CUDA**, float16 | CPU only, permanently | CPU, int8 |
| Parakeet | **CUDA** | CoreML | CPU |
| Speaker labelling | **CUDA** | CoreML | CPU |
| Ollama | **CUDA** | Metal | CPU |

**NVIDIA is the only platform where all four accelerate.** CTranslate2 — what
faster-whisper is built on — has no Metal backend and is not getting one, so
Whisper stays on the CPU on a Mac however fast the Mac is. A Mac mini is an
excellent Ollama server and half a speech machine.

Two things to know before expecting CUDA to work:

- **sherpa-onnx from PyPI is CPU-only.** Parakeet and speaker labelling need
  the GPU build from the k2-fsa index; without it `provider = "cuda"` falls
  back to the CPU and says so in the log.
- **faster-whisper needs the CUDA runtime, not just the driver.** cuBLAS and
  cuDNN ship with the CUDA toolkit. Missing them is what produces `Library
  cublas64_12.dll is not found`, which is handled — it falls back to the CPU —
  but the GPU sits idle until they are installed.

`doctor.bat` prints where each stage will actually run, which is the only
reliable way to know the setting took effect.

## Testing without a phone

`test` accepts a folder as well as a file:

```
run.bat test C:\calls\
```

Every recording in it, one after another, work orders to `output/` and to the
console. The popup is suppressed — twenty windows over an hour is not a
feature. This is how to exercise the pipeline on a machine with no adapter
attached to it at all.
