# Guitar Synthesizer

Created by **Reza Afrasyabi** — [rezaafrasyabi.com](https://rezaafrasyabi.com)

Realistic guitar synthesis in Python using the **Karplus-Strong** plucked string algorithm. Three guitar types, two song formats, and real-time chord display synced with playback.

---

## Guitar Types

| Type | Sound | Character |
|------|-------|-----------|
| **Classic** | Classical nylon-string | Warm, soft attack, wooden body resonance |
| **Metal** | Acoustic steel-string | Bright, snappy, ringing overtones |
| **Electric** | Electric with overdrive | Sharp attack, long sustain, tanh distortion |

## Song Formats

**Chord-based (strum)** — JSON files with chord names. The synth renders full 6-string strummed chords with humanized timing.

**Note-based (fingerpick)** — JSON files with individual `[string, fret]` notes. Each note is plucked independently and rings naturally, overlapping with surrounding notes to create the fingerpicking sound.

## Usage

```
python guitar.py
```

1. Pick a guitar type (classic / metal / electric)
2. Choose an action:
   - `J` — play a song from a JSON file (chords/notes display in terminal synced with playback)
   - `S` — render a JSON song to a `.wav` file
   - `T` — switch guitar type
   - Type a chord name (e.g. `Am`, `G`, `Em`) to play a single chord

## Included Songs

| File | Song | Format |
|------|------|--------|
| `the_last_of_us_main_theme.json` | The Last of Us — Main Theme (Gustavo Santaolalla) | Fingerpick (506 notes) |

### Example Output

**The Last of Us — Main Theme** (Classical Nylon guitar):

<audio controls>
  <source src="the_last_of_us_main_theme.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

[Download the_last_of_us_main_theme.wav](the_last_of_us_main_theme.wav)

## Adding Your Own Songs

**Chord-based** (strum):

```json
{
  "title": "My Song",
  "duration_per_chord": 1.2,
  "sections": [
    {
      "name": "Verse",
      "chords": ["Am", "G", "F", "Em"]
    }
  ]
}
```

**Note-based** (fingerpick):

```json
{
  "title": "My Song",
  "format": "notes",
  "note_interval": 0.25,
  "note_ring": 2.0,
  "sections": [
    {
      "name": "Intro",
      "notes": [
        [3,12], [4,0], [5,0],
        [3,9], [4,0], [5,0]
      ]
    }
  ]
}
```

Each note is `[string_index, fret]` where string 0 = low E through string 5 = high e. For simultaneous notes, nest them: `[[0,0], [5,3]]`.

## Supported Chords

`A` `Am` `A7` `B` `Bm` `C` `C7` `D` `Dm` `D7` `E` `Em` `E7` `F` `G` `G7`

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

`numpy` and `sounddevice` are required for real-time playback. The synth can render to `.wav` files without `sounddevice`, but `numpy` is needed for fast rendering.

## Requirements

- Python 3.10+
- numpy
- sounddevice
