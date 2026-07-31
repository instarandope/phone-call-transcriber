# Call Transcriber

Turns an incoming phone call into a service work order, on your own computer,
without you touching anything.

A call comes in. When you hang up, a window pops up with the customer's name,
address, the problem, access notes and what still needs asking — already copied
to your clipboard, ready to paste. The recording is never saved.

```
WORK ORDER - Ridgeline Plumbing & Heating
Call received Fri 31 Jul 2026, 03:30 PM  (6m 14s)
------------------------------------------------------------
CUSTOMER      Maria Delgado (Delgado Property Mgmt)
PHONE         555 240 1188
ADDRESS       418 Rosewood Ave, Apt 2B, Springfield IL 62704
ISSUE         No hot water, tenant reports water pooling near heater
DETAILS
  Started yesterday evening. Tenant heard a click then nothing.
  Pilot light will not stay lit after three attempts. Small puddle
  at the base, roughly a dinner plate wide, not spreading.

EQUIPMENT     Rheem gas water heater, 40 gal, installed 2017
URGENCY       URGENT
ACCESS
  Lockbox on the railing, code 5512. Tenant home after 4pm. Small
  dog, friendly.

AVAILABLE     Tomorrow after 4pm, or Thursday morning
SCHEDULED     Thursday 1 Aug, 9-11am window
EXISTING      YES
PRICING       $89 diagnostic fee, waived if repair proceeds

FOLLOW-UP
  - Text the tenant the arrival window
  - Quote a replacement if the tank is cracked

STILL NEEDED
  - Model number
  - Whether the shutoff valve was closed
------------------------------------------------------------
```

Everything runs locally. No audio, no transcript and no customer detail is sent
anywhere — there is no account, no API key and no cloud service involved.

## Nothing to click

There is no record button. The adapter is live whenever the handset is off the
cradle, so the app listens to it continuously and works out for itself when a
call starts and stops:

1. Speech on the line for a third of a second → it starts recording.
2. The line going properly quiet → the handset is back on the cradle, the call
   is over, and processing begins.
3. Calls under ten seconds are thrown away as wrong numbers and false starts.

Step 2 is deliberately not "six seconds of silence". A pause is not a hangup —
callers go quiet all the time to find a model number or check a calendar, and
cutting the recording there would split one call into two half-useless work
orders. An open phone line is never truly silent; it carries line noise and
room tone even when nobody is speaking, and that only stops when the handset
goes down. So the app watches the **line level**, not the absence of speech.

There is a second, much longer timeout (45 seconds with no speech at all) as a
backstop for lines that keep humming after the far end hangs up.

During processing it is still listening, so a call arriving right behind
another one is not missed — see [Back-to-back calls](#back-to-back-calls).

## What you need

| | |
|---|---|
| **Adapter** | VEC LRX-40USB, plugged into a USB port, with the handset cord running through it. Corded phones only — it cannot tap a cordless handset. |
| **Windows** | 10 or 11. |
| **Python** | 3.11 or newer, from [python.org](https://www.python.org/downloads/). Tick **Add python.exe to PATH** on the first installer screen. |
| **Ollama** | From [ollama.com/download](https://ollama.com/download). This runs the local model that reads the transcript. |
| **Disk** | About 5 GB for the two models. |
| **RAM** | 8 GB works. 16 GB is comfortable. |

No graphics card is needed.

## Setting it up

Double-click **`install.bat`**. It creates the Python environment, installs
everything, downloads both models, and offers to start the app automatically
with Windows. It finishes by checking the whole setup and telling you about
anything that is wrong.

Then find your adapter:

```
run.bat devices
```

You will see something like `[2] Microphone (USB Audio Device) (1ch @ 44100 Hz)`.
Open `config.toml`, and set `device_match` to a distinctive piece of that name:

```toml
[audio]
device_match = "USB Audio"
```

Check everything is ready:

```
run.bat doctor
```

Then calibrate the call detection against your actual phone line — this takes
45 seconds and is worth doing properly:

```
run.bat levels
```

It shows a live level meter and asks you to leave the handset on the cradle,
then lift it and stay quiet, then talk. From those three it prints the two
threshold values to paste into `config.toml`. The defaults are reasonable
guesses; these are measurements.

Once every `doctor` line reads `[ok]`, you are done. If you let the installer
add it to startup, it is already running invisibly in the background —
otherwise double-click **`run.bat`**.

## Trying it without waiting for a real call

Record yourself talking through a fake service call, save it as a `.wav`, and
run it through the whole pipeline:

```
run.bat test my-test-call.wav
```

You get the same popup and the same files you would get from a real call. This
is the fastest way to see whether the extraction is picking up what you need.

To hear what the adapter is actually capturing, turn on `keep_audio` in
`config.toml` for a day, take a few calls, and listen to the `call.wav` saved
next to each work order. **Turn it back off when you are done** — and run
`run.bat purge` to shred the recordings you kept.

## Back-to-back calls

Recording and processing are separate threads, so the app keeps listening while
it works on the previous call. Nothing is missed when one call lands right on
top of another.

What can happen on a busy stretch is that work orders start arriving late.
Transcription and extraction are queued and run one at a time — deliberately,
because running two at once on the same CPU makes both slower rather than
finishing sooner. So if a call takes two minutes to process and you take three
calls in five minutes, the third work order shows up a few minutes after you
hang up. The queue drains during the first gap.

If that lag becomes a problem, the fix is to make each call cheaper rather than
to run more at once: drop `transcribe.model` to `base.en`, or `extract.model`
to `gemma3:1b`. Watch `processing_s` in any `extracted.json` to see where you
actually stand — if it is comfortably under the length of the call itself, you
will keep up with anything short of continuous talking.

Queued calls hold their audio in memory, around 2 MB per minute. Even a deep
backlog is nowhere near your RAM.

## Changing what it pulls out

Everything the app looks for lives in one file:
[`src/call_transcriber/fields.py`](src/call_transcriber/fields.py). Each entry
is a field on the work order. Adding one is a single block:

```python
Field(
    name="referred_by",
    label="REFERRED BY",
    prompt="How the caller heard about the business, if they say.",
),
```

That is the whole change. The instructions given to the model, the JSON it must
return, and the layout of the printed work order are all generated from that
list, so nothing else needs editing. Delete an entry and the field disappears
just as cleanly.

The wording in `prompt` matters — it is passed to the model more or less
verbatim. Be specific about what counts and what does not.

## What gets saved, and what doesn't

Each call produces a folder:

```
output/2026-07-31/152214-maria-delgado/
    work_order.txt     the text you paste
    transcript.txt     the full conversation
    extracted.json     the same fields, for feeding into other software
```

**The recording is not among them.** The audio is transcribed directly from
memory and is never written to disk unless you deliberately switch `keep_audio`
on. There is no temporary file and no window during which a copy exists.

Transcripts and work orders *are* kept, because they are the point. If you want
those gone too:

```
run.bat purge --all
```

Both purge modes overwrite files before deleting them. That defeats ordinary
recovery, but it is not a guarantee against forensic recovery — on an SSD,
wear-levelling can leave the original blocks somewhere the filesystem can no
longer reach. If that matters for your client data, turn on BitLocker.

`.gitignore` is set up so that no recording, transcript or work order can be
committed to this repository by accident.

## Recording calls legally

Consent rules for recording phone calls vary by state and country. Some places
require only one party to know; others require everyone on the line to consent.
Check what applies where you operate, and if you need an announcement at the
start of the call, make sure it is in place before you rely on this.

## Configuration worth knowing about

Full documentation is in the comments of
[`config.example.toml`](config.example.toml). The settings people actually
change:

| Setting | Default | Change it when |
|---|---|---|
| `detect.line_dead_dbfs` | `-60.0` | Calls end during pauses (lower it), or hangups take too long to notice (raise it). `run.bat levels` measures the right value. |
| `detect.line_dead_s` | `3.0` | Your line dips to silence briefly mid-call — raise it. |
| `detect.hangup_silence_s` | `45.0` | Your line stays loud after a hangup, so calls only end on this timeout — lower it. Never below ~20. |
| `detect.min_call_s` | `10.0` | Short but real calls are vanishing — lower it. |
| `detect.noise_floor_dbfs` | `-50.0` | Line noise starts phantom recordings (raise to `-40`), or quiet callers are missed (lower to `-60`). |
| `transcribe.model` | `small.en` | Transcription is too slow (`base.en`) or not accurate enough (`medium.en`). |
| `extract.model` | `gemma3:4b` | Extraction is slow (`gemma3:1b`) or sloppy with addresses (`gemma3:12b`). |
| `business.name` | — | Set it so the model never mistakes your own name or address for the customer's. |

## When something is wrong

Start with `run.bat doctor`. It checks each piece and says what to do about
anything broken. Beyond that:

**No device matching 'LRX'** — the adapter reports itself under a generic name
on many machines. Run `run.bat devices` and copy part of whatever you actually
see into `device_match`.

**Nothing happens during calls** — check that the handset cord runs *through*
the adapter rather than around it, then confirm Windows can hear it: Settings →
System → Sound → Input, pick the adapter, and watch the level bar while you
talk. If the bar does not move, it is a wiring problem, not a software one.

**Recordings start when nobody is calling** — line noise is crossing the
threshold. Run `run.bat levels`, or just raise `noise_floor_dbfs` to `-40`.

**Calls get cut in half** — the line is dropping below `line_dead_dbfs` during
pauses, so a hesitation reads as a hangup. Run `run.bat levels` while
deliberately staying quiet on an open line; if that quiet sits near the dead
threshold, lower `line_dead_dbfs` by 5–10, or raise `line_dead_s` to `5.0`.
Each half of a split call is saved separately, so check the folder either side
of the one you noticed — the customer's name and address are usually in the
first half.

**Calls take ages to end after you hang up** — the dead-line test is not
firing, so it is falling through to the 45-second backstop. Your line stays
noisy on hook: run `run.bat levels`, note the on-cradle level, and set
`line_dead_dbfs` a few dB above it.

**The transcript is good but the fields are empty** — that is the extraction
step, not the speech step. Check Ollama is running, then try a larger model.

**"cannot reach Ollama"** — it is installed but not started. Open Ollama from
the Start menu; it lives in the system tray.

**Transcription is very slow** — a call should process in well under its own
length. If not, drop `transcribe.model` to `base.en`.

## How it works

```
LRX-40USB  →  capture  →  detect  →  whisper  →  gemma  →  work order
              16 kHz      start/end   on-device  on-device   clipboard
                                                             + popup
                                                             + files
```

Three threads: the audio callback, a detector that decides where calls begin
and end, and a worker that transcribes and extracts. They are separate so a
long transcription cannot cause the next call to be missed.

Speech-to-text is [faster-whisper](https://github.com/SYSTRAN/faster-whisper);
field extraction is [Gemma](https://ollama.com/library/gemma3) through
[Ollama](https://ollama.com), constrained to a fixed JSON schema so the reply
is always well-formed. The extraction prompt is deliberately strict about not
inventing values — a blank address is a nuisance, but a confidently wrong one
sends a tech to the wrong house.

If you were expecting [FluidVoice](https://github.com/altic-dev/FluidVoice)
here: it is a macOS dictation app driven by a hotkey, with no way to run
headless or process a recording, so it cannot do this job. This uses the same
class of on-device speech model it does, without the dictation UI.

## Development

```
pip install -e ".[dev]"
pytest
```

The tests stub out the two model calls, so the suite runs in under a second and
needs neither Ollama nor a downloaded whisper model.

## Licence

MIT — see [LICENSE](LICENSE).
