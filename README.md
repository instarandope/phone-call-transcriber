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

## Two ways to run it

Set `mode` under `[control]` in `config.toml`.

**`manual`** — a global hotkey (default **Ctrl+Alt+R**) starts recording, and
pressing it again stops. Two rising beeps confirm the start, two falling ones
the stop, so you never have to look at the screen. Nothing is recorded until
you say so.

Use this in a workshop, a shop floor, or any room where people talk near the
phone. Automatic detection assumes the loudest thing near the adapter is the
phone; in a noisy room that is false, and you get work orders generated from
conversations that were never calls.

**`auto`** — the app decides, as described below. Good in a quiet office.

## How auto mode decides

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
| **Python** | **3.12** is the one to install, from [python.org](https://www.python.org/downloads/) — scroll past the yellow button to the version list. 3.11 and 3.13 work too. Tick **Add python.exe to PATH** on the first installer screen. Not from the Microsoft Store. |
| **Ollama** | From [ollama.com/download](https://ollama.com/download). This runs the local model that reads the transcript. |
| **Disk** | About 5 GB for the two models. |
| **RAM** | 8 GB works. 16 GB is comfortable. |

No graphics card is needed.

## Wiring the adapter

Get this right before touching any settings. A misconnected adapter produces
symptoms that look exactly like software bugs.

The LRX-40USB sits **in the middle of the handset connection**, not alongside
it:

```
phone's handset jack ──cord──► LRX-40USB ◄──curly cord── handset
                                   │
                                  USB ──► PC
```

Per the manufacturer's instructions: plug your handset's curly cord into the
LRX-40USB's modular jack, plug the LRX-40USB's own cord into the telephone,
then the USB into the computer.

**Use the handset jack — often labelled HAC — not the headset jack.** They look
identical and both accept the plug. The headset jack is a separate circuit that
is live under different conditions, and an adapter on it will hear the room
while the phone is hung up.

The adapter has two controls, and both matter:

| Control | What it does |
|---|---|
| **Selector switch** | Matches the handset wiring configuration, which differs by phone brand. There is no way to know the right position from the outside — test both. |
| **Record level dial** | Sets the level of *the calling party* specifically. Too low and the customer is faint in the transcript while your side is fine. Start at the midpoint. |

The multi-colour LED is not documented beyond being a "mode indicator", so
judge by the level meter rather than by the light.

**Verify with `levels.bat`.** With the handset on the cradle the meter
should stay flat, even when you talk in the room. Lift the handset and it
should jump on dial tone. Try both switch positions and keep the one that
behaves that way. If neither does, the adapter is on the wrong jack.

**Then check the switch with the dial, which is the faster test.** Record a
call with `keep_audio = true`, play it back, and turn the record level dial
while listening:

- **Both voices change level** → correct position. The adapter is on the line.
- **Only your own voice changes** → wrong position. That one taps the
  mouthpiece alone, and every transcript will be missing the customer.

The second case is the dangerous one, because it produces transcripts that look
fine until you notice the only half present is yours. Nothing in the audio
announces it — you have to go looking.

Aim for speech peaking around **-12 dBFS**. Too far right is worse than too far
left: clipping destroys the waveform, and whisper copes with a quiet recording
far better than a distorted one. The app warns about both after each call.

## Setting it up

Double-click **`install.bat`**. It creates the Python environment, installs
everything, downloads both models, and offers to start the app automatically
with Windows. It finishes by checking the whole setup and telling you about
anything that is wrong.

Then find your adapter by double-clicking **`devices.bat`**.

You will see something like `[2] Microphone (USB Audio Device) (1ch @ 44100 Hz)`.
Open `config.toml`, and set `device_match` to a distinctive piece of that name:

```toml
[audio]
device_match = "USB Audio"
```

Check everything is ready by double-clicking **`doctor.bat`**.

Then calibrate the call detection against your actual phone line by
double-clicking **`levels.bat`**. This takes 45 seconds and is worth doing
properly.

It shows a live level meter and asks you to leave the handset on the cradle,
then lift it and stay quiet, then talk. From those three it prints the two
threshold values to paste into `config.toml`. The defaults are reasonable
guesses; these are measurements.

Once every `doctor.bat` line reads `[ok]`, you are done. If you let the installer
add it to startup, it is already running invisibly in the background —
otherwise double-click **`run.bat`**.

## The double-clickable scripts

Everything can be run by double-clicking, so you never need a terminal:

| File | What it does |
|---|---|
| **`install.bat`** | One-time setup. Safe to re-run; it skips whatever is already done. |
| **`run.bat`** | Start listening for calls. |
| **`devices.bat`** | List audio inputs, to find the adapter's name for `config.toml`. |
| **`doctor.bat`** | Check everything: packages, adapter, models, output folder. |
| **`levels.bat`** | Meter the phone line and print the threshold values to use. |

The less common ones need a terminal, because they take an argument. Open the
folder in File Explorer, click the address bar, type `cmd` and press Enter — a
Command Prompt opens already pointed at the folder. Then:

```
run.bat test somecall.wav    process an existing recording
run.bat purge                shred any kept audio
run.bat purge --all          shred transcripts and work orders too
```

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

Everything the model is asked to find lives in one file:
[`src/call_transcriber/fields.py`](src/call_transcriber/fields.py). Open it in
Notepad. Each entry is one field on the work order, and they print in the order
they appear.

**To capture something new**, add a block anywhere in the list:

```python
Field(
    name="referred_by",
    label="REFERRED BY",
    prompt="How the caller heard about the business, if they say.",
),
```

- `name` — lowercase, no spaces. The key in `extracted.json`.
- `label` — what prints on the work order.
- `prompt` — what you are telling the model to look for.

That is the whole change. The instructions sent to the model, the JSON schema
it must fill in, and the printed layout are all generated from this list, so
nothing else needs editing. Restart the app and the next call has the field.

**To stop capturing something**, delete its block. **To reword something**,
edit its `prompt`.

**Three kinds of field** are available:

```python
Field(name="po_number", label="PO NUMBER", prompt="..."),                    # text

Field(name="parts_needed", label="PARTS", prompt="...", kind="list"),        # a list

Field(name="paid", label="PAID", prompt="...",                               # fixed choices
      kind="enum", choices=("yes", "no", "unknown")),
```

The wording of `prompt` is what actually does the work — it reaches the model
close to verbatim. Be specific about what counts and what does not, the way the
existing entries are. "Any figures quoted or agreed: service call fee, estimate
range, deposit" gets you far better results than "pricing", because it tells
the model where the boundary is.

Two things worth knowing:

- The model is instructed never to invent a value. If a caller did not say
  something, the field comes back empty and its label appears under **STILL
  NEEDED** instead. That is deliberate — a blank address is a nuisance, a
  confidently wrong one sends a tech to the wrong house.
- Adding a lot of fields makes each call slower to process, since the model has
  more to produce. A dozen is comfortable.

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

Start with `doctor.bat`. It checks each piece and says what to do about
anything broken. Beyond that:

**No device matching 'LRX'** — the adapter reports itself under a generic name
on many machines. Double-click `devices.bat` and copy part of whatever you actually
see into `device_match`.

**Nothing happens during calls** — check that the handset cord runs *through*
the adapter rather than around it, then confirm Windows can hear it: Settings →
System → Sound → Input, pick the adapter, and watch the level bar while you
talk. If the bar does not move, it is a wiring problem, not a software one.

**Recordings start when nobody is calling** — line noise is crossing the
threshold. Run `levels.bat`, or just raise `noise_floor_dbfs` to `-40`.

**Calls get cut in half** — the line is dropping below `line_dead_dbfs` during
pauses, so a hesitation reads as a hangup. Run `levels.bat` while
deliberately staying quiet on an open line; if that quiet sits near the dead
threshold, lower `line_dead_dbfs` by 5–10, or raise `line_dead_s` to `5.0`.
Each half of a split call is saved separately, so check the folder either side
of the one you noticed — the customer's name and address are usually in the
first half.

**Calls take ages to end after you hang up** — the dead-line test is not
firing, so it is falling through to the 45-second backstop. Your line stays
noisy on hook: run `levels.bat`, note the on-cradle level, and set
`line_dead_dbfs` a few dB above it.

**The transcript is good but the fields are empty** — that is the extraction
step, not the speech step. Check Ollama is running, then try a larger model.

**"cannot reach Ollama"** — it is installed but not started. Open Ollama from
the Start menu; it lives in the system tray.

**Transcription is very slow** — a call should process in well under its own
length. If not, drop `transcribe.model` to `base.en`.

**`install.bat` fails with compiler errors** — "Microsoft Visual C++ 14.0 or
greater is required", "Building wheel for … error", or "no matching
distribution found". Your Python is newer than some dependencies have shipped
ready-built wheels for, so pip is trying to compile C++ from source. Install
Python 3.12, delete the `.venv` folder, and run `install.bat` again. Newest is
not best here — pick the version the ecosystem has caught up with.

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
