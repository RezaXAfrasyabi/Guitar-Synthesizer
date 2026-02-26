import io
import json
import math
import random
import struct
import sys
import threading
import time
import wave

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

SAMPLE_RATE = 44100

GUITAR_TYPES = {
    "classic": {
        "label": "Classical (Nylon)",
        "warmth": (4, 2),
        "brightness": (0.9920, 0.9957),
        "body": (0.18, 0.12),
        "volume": (0.40, 0.75),
        "fade_ms": 3.0,
        "distortion": 0.0,
    },
    "metal": {
        "label": "Acoustic Steel-String",
        "warmth": (1, 0),
        "brightness": (0.9960, 0.9992),
        "body": (0.08, 0.04),
        "volume": (0.50, 0.85),
        "fade_ms": 1.0,
        "distortion": 0.0,
    },
    "electric": {
        "label": "Electric (Overdrive)",
        "warmth": (0, 0),
        "brightness": (0.9980, 0.9996),
        "body": (0.0, 0.0),
        "volume": (0.55, 0.90),
        "fade_ms": 0.5,
        "distortion": 0.40,
    },
}

DEFAULT_GTYPE = "classic"


def _string_params(string_idx: int, gtype: str = DEFAULT_GTYPE) -> tuple:
    gt = GUITAR_TYPES.get(gtype, GUITAR_TYPES[DEFAULT_GTYPE])
    t = string_idx / 5.0
    bright = gt["brightness"][0] + t * (gt["brightness"][1] - gt["brightness"][0])
    warmth = round(gt["warmth"][0] + t * (gt["warmth"][1] - gt["warmth"][0]))
    body = gt["body"][0] + t * (gt["body"][1] - gt["body"][0])
    vol_range = gt["volume"][1] - gt["volume"][0]
    vol_lo = gt["volume"][0] + t * vol_range * 0.4
    vol_hi = gt["volume"][0] + vol_range * 0.5 + t * vol_range * 0.5
    return (bright, vol_lo, vol_hi, warmth, body)


OPEN_STRINGS = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]


def _note_freq(string_idx: int, fret: int) -> float:
    return OPEN_STRINGS[string_idx] * (2 ** (fret / 12.0))


_CHORD_FRETS = {
    "C":  [(1,3),(2,2),(3,0),(4,1),(5,0)],
    "D":  [(2,0),(3,2),(4,3),(5,2)],
    "E":  [(0,0),(1,2),(2,2),(3,1),(4,0),(5,0)],
    "F":  [(0,1),(1,3),(2,3),(3,2),(4,1),(5,1)],
    "G":  [(0,3),(1,2),(2,0),(3,0),(4,0),(5,3)],
    "A":  [(1,0),(2,2),(3,2),(4,2),(5,0)],
    "B":  [(1,2),(2,4),(3,4),(4,4),(5,2)],
    "AM": [(1,0),(2,2),(3,2),(4,1),(5,0)],
    "BM": [(1,2),(2,4),(3,4),(4,3),(5,2)],
    "EM": [(0,0),(1,2),(2,2),(3,0),(4,0),(5,0)],
    "DM": [(2,0),(3,2),(4,3),(5,1)],
    "C7": [(1,3),(2,2),(3,3),(4,1),(5,0)],
    "G7": [(0,3),(1,2),(2,0),(3,0),(4,0),(5,1)],
    "D7": [(2,0),(3,2),(4,1),(5,2)],
    "E7": [(0,0),(1,2),(2,0),(3,1),(4,0),(5,0)],
    "A7": [(1,0),(2,2),(3,0),(4,2),(5,0)],
}

CHORDS = {name: [_note_freq(s, f) for s, f in frets] for name, frets in _CHORD_FRETS.items()}


def _chord_key(name: str) -> str:
    s = name.strip().upper()
    if len(s) >= 2 and s[1] == "M" and s[0] in "ABCDEFG":
        return s
    return s


def _ks_pluck(freq: float, duration: float, sample_rate: int = SAMPLE_RATE,
              brightness: float = 0.996, volume: float = 0.7,
              warmth: int = 3, body_mix: float = 0.15,
              fade_ms: float = 3.0) -> list:
    if freq <= 20:
        return [0.0] * int(sample_rate * duration)
    period = max(2, int(round(sample_rate / freq)))
    n = int(sample_rate * duration)

    buf = [random.uniform(-volume, volume) for _ in range(period)]
    for _ in range(warmth):
        for i in range(period):
            nxt = (i + 1) % period
            buf[i] = 0.5 * (buf[i] + buf[nxt])

    out = [0.0] * n
    for i in range(n):
        idx = i % period
        out[i] = buf[idx]
        nxt = (idx + 1) % period
        buf[idx] = brightness * 0.5 * (buf[idx] + buf[nxt])

    fade_samples = min(int(fade_ms * 0.001 * sample_rate), n)
    if fade_samples > 0:
        for i in range(fade_samples):
            out[i] *= i / fade_samples

    if body_mix > 0:
        decay = 4.0 / max(duration, 0.01)
        two_pi_f = 2.0 * math.pi * freq
        for i in range(n):
            t = i / sample_rate
            out[i] += body_mix * volume * math.exp(-decay * t) * math.sin(two_pi_f * t)

    return out


def _apply_distortion(samples, amount: float):
    if amount <= 0:
        return samples
    gain = 1.0 + amount * 10.0
    if HAS_NP:
        arr = np.array(samples, dtype=np.float64) if not isinstance(samples, np.ndarray) else samples.copy()
        peak = np.max(np.abs(arr))
        if peak > 0:
            arr /= peak
        arr = np.tanh(arr * gain)
        return arr
    peak = max(abs(s) for s in samples) or 1.0
    return [math.tanh(s / peak * gain) for s in samples]


def _strum_chord(freqs: list, duration: float, sample_rate: int = SAMPLE_RATE,
                 strum_time: float = 0.045, down: bool = True,
                 gtype: str = DEFAULT_GTYPE) -> list:
    gt = GUITAR_TYPES.get(gtype, GUITAR_TYPES[DEFAULT_GTYPE])
    n = int(sample_rate * duration)
    mixed = [0.0] * n
    num = len(freqs)
    if num == 0:
        return mixed
    delay_per = strum_time / max(1, num - 1) if num > 1 else 0
    order = list(range(num)) if down else list(range(num - 1, -1, -1))
    for idx, s in enumerate(order):
        delay = int(idx * delay_per * sample_rate)
        bl, bh = gt["brightness"]
        bright = random.uniform(bl, bh)
        vl, vh = gt["volume"]
        vol = random.uniform(vl * 0.9, vh * 0.9)
        remaining = duration - delay / sample_rate
        if remaining < 0.05:
            continue
        w = round((gt["warmth"][0] + gt["warmth"][1]) / 2)
        bd = (gt["body"][0] + gt["body"][1]) / 2
        tone = _ks_pluck(freqs[s], remaining, sample_rate, bright, vol,
                         warmth=w, body_mix=bd, fade_ms=gt["fade_ms"])
        for i in range(len(tone)):
            if delay + i < n:
                mixed[delay + i] += tone[i]
    return mixed


def render_chord_song(song: dict, duration_per_chord: float | None = None,
                      gtype: str = DEFAULT_GTYPE) -> list:
    dur = duration_per_chord or song.get("duration_per_chord", 1.2)
    all_samples = []
    down = True
    for section in song.get("sections", []):
        for c in section.get("chords", []):
            cn = _chord_key(c)
            if cn not in CHORDS:
                all_samples.extend([0.0] * int(SAMPLE_RATE * dur))
                continue
            d = dur + random.uniform(-0.02, 0.02)
            chunk = _strum_chord(CHORDS[cn], d, SAMPLE_RATE,
                                 random.uniform(0.025, 0.05), down, gtype)
            all_samples.extend(chunk)
            down = not down
    return all_samples


def render_notes_song(song: dict, gtype: str = DEFAULT_GTYPE) -> list:
    gt = GUITAR_TYPES.get(gtype, GUITAR_TYPES[DEFAULT_GTYPE])
    interval = song.get("note_interval", 0.22)
    ring = song.get("note_ring", 2.0)

    events = []
    t = 0.0
    for section in song.get("sections", []):
        sec_interval = section.get("note_interval", interval)
        sec_ring = section.get("note_ring", ring)
        if "notes" in section:
            for note in section["notes"]:
                events.append((t, note, sec_ring))
                t += sec_interval + random.uniform(-0.006, 0.006)
        elif "chords" in section:
            dur = section.get("duration_per_chord", song.get("duration_per_chord", 1.4))
            for chord in section["chords"]:
                cn = _chord_key(chord)
                if cn in CHORDS:
                    events.append((t, ("strum", cn, dur), 0))
                t += dur

    total_sec = t + ring + 0.3
    n_total = int(SAMPLE_RATE * total_sec)

    if HAS_NP:
        out = np.zeros(n_total, dtype=np.float64)
    else:
        out = [0.0] * n_total

    count = len(events)
    for ei, (time, data, note_ring) in enumerate(events):
        if ei % 50 == 0:
            print(f"\r  Rendering: {ei}/{count} notes...", end="", flush=True)
        start = int(time * SAMPLE_RATE)

        if isinstance(data, tuple) and data[0] == "strum":
            cn, dur = data[1], data[2]
            chunk = _strum_chord(CHORDS[cn], dur, SAMPLE_RATE,
                                 random.uniform(0.03, 0.05), True, gtype)
            end = min(start + len(chunk), n_total)
            if HAS_NP:
                out[start:end] += np.array(chunk[:end - start])
            else:
                for j in range(end - start):
                    out[start + j] += chunk[j]
            continue

        notes_list = [data] if isinstance(data[0], int) else data
        for nd in notes_list:
            s_idx, fret = nd[0], nd[1]
            freq = _note_freq(s_idx, fret)
            bright_base, vol_lo, vol_hi, warmth, body = _string_params(s_idx, gtype)
            bright = bright_base + random.uniform(-0.0006, 0.0006)
            vol = random.uniform(vol_lo, vol_hi)
            tone = _ks_pluck(freq, note_ring, SAMPLE_RATE, bright, vol,
                             warmth=warmth, body_mix=body, fade_ms=gt["fade_ms"])
            end = min(start + len(tone), n_total)
            if HAS_NP:
                out[start:end] += np.array(tone[:end - start])
            else:
                for j in range(end - start):
                    out[start + j] += tone[j]

    print(f"\r  Rendering: {count}/{count} notes... done.")

    if HAS_NP:
        return out
    return out


def render_song(song: dict, duration_per_chord: float | None = None,
                gtype: str = DEFAULT_GTYPE) -> list:
    has_notes = any("notes" in s for s in song.get("sections", []))
    if has_notes:
        samples = render_notes_song(song, gtype)
    else:
        samples = render_chord_song(song, duration_per_chord, gtype)

    dist = GUITAR_TYPES.get(gtype, GUITAR_TYPES[DEFAULT_GTYPE])["distortion"]
    if dist > 0:
        print("  Applying distortion...")
        samples = _apply_distortion(samples, dist)

    return samples


def normalize_and_pack(samples) -> bytes:
    if HAS_NP:
        arr = np.array(samples, dtype=np.float64) if not isinstance(samples, np.ndarray) else samples
        peak = np.max(np.abs(arr)) or 1.0
        arr = (arr / peak * 30000).clip(-32768, 32767).astype(np.int16)
        return arr.tobytes()
    if not samples:
        return b""
    peak = max(abs(s) for s in samples) or 1.0
    scale = 30000 / peak
    packed = [max(-32768, min(32767, int(s * scale))) for s in samples]
    return struct.pack(f"<{len(packed)}h", *packed)


def save_wav(path: str, data: bytes) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(data)


def mix_chord(freqs: list, duration: float, sample_rate: int = SAMPLE_RATE,
              gtype: str = DEFAULT_GTYPE) -> bytes:
    chunk = _strum_chord(freqs, duration, sample_rate,
                         random.uniform(0.025, 0.045), True, gtype)
    dist = GUITAR_TYPES.get(gtype, GUITAR_TYPES[DEFAULT_GTYPE])["distortion"]
    if dist > 0:
        chunk = _apply_distortion(chunk, dist)
    return normalize_and_pack(chunk)


def play_wav(path: str) -> None:
    try:
        import sounddevice as sd
        with wave.open(path, "rb") as wav:
            raw = wav.readframes(wav.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16) if HAS_NP else list(struct.unpack(f"<{len(raw)//2}h", raw))
        sd.play(np.array(samples, dtype=np.int16) if not HAS_NP else samples, SAMPLE_RATE)
        sd.wait()
    except ImportError:
        print(f"Saved: {path}  (install sounddevice + numpy to play)")


def play_chord_to_wav(chord_name: str, duration: float = 1.0,
                      out_path: str | None = None, gtype: str = DEFAULT_GTYPE) -> str:
    name = _chord_key(chord_name)
    if name not in CHORDS:
        raise ValueError(f"Unknown chord: {chord_name}. Use one of: {', '.join(sorted(CHORDS))}")
    path = out_path or f"chord_{name}.wav"
    data = mix_chord(CHORDS[name], duration, SAMPLE_RATE, gtype)
    save_wav(path, data)
    return path


def load_song_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OPEN_MIDI = [40, 45, 50, 55, 59, 64]


def _fret_to_note_name(string_idx: int, fret: int) -> str:
    midi = OPEN_MIDI[string_idx] + fret
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:04.1f}"


def _build_timeline(song: dict, duration_per_chord: float | None = None) -> list:
    timeline = []
    interval = song.get("note_interval", 0.22)
    default_dur = duration_per_chord or song.get("duration_per_chord", 1.2)
    t = 0.0

    for section in song.get("sections", []):
        sec_name = section.get("name", "")

        if "notes" in section:
            sec_interval = section.get("note_interval", interval)
            timeline.append((t, sec_name, "section"))
            melody_notes = []
            for note in section["notes"]:
                if isinstance(note[0], int):
                    s_idx, fret = note[0], note[1]
                    melody_notes.append((t, _fret_to_note_name(s_idx, fret)))
                else:
                    names = [_fret_to_note_name(n[0], n[1]) for n in note]
                    melody_notes.append((t, "+".join(names)))
                t += sec_interval
            for mt, mname in melody_notes:
                timeline.append((mt, mname, "note"))

        elif "chords" in section:
            dur = section.get("duration_per_chord", default_dur)
            if sec_name:
                timeline.append((t, sec_name, "section"))
            for chord in section["chords"]:
                timeline.append((t, chord, "chord"))
                t += dur

    return timeline


def _sync_display(timeline: list, stop_event: threading.Event):
    start = time.time()
    prev_section = ""
    prev_chord = ""

    for evt_time, text, kind in timeline:
        if stop_event.is_set():
            break
        now = time.time() - start
        wait = evt_time - now
        if wait > 0.01:
            stop_event.wait(wait)
            if stop_event.is_set():
                break

        if kind == "section":
            if text != prev_section:
                print(f"\n  [{_fmt_time(evt_time)}]  -- {text} --")
                prev_section = text
        elif kind == "chord":
            if text != prev_chord:
                print(f"  [{_fmt_time(evt_time)}]  {text}")
                prev_chord = text
        elif kind == "note":
            print(f"\r  [{_fmt_time(evt_time)}]  {text:<12}", end="", flush=True)

    print()


def play_song_from_json(path: str, duration_per_chord: float | None = None,
                        gtype: str = DEFAULT_GTYPE) -> None:
    song = load_song_json(path)
    title = song.get("title", song.get("title_fa", path))
    total = sum(len(s.get("chords", s.get("notes", []))) for s in song.get("sections", []))
    gt_label = GUITAR_TYPES.get(gtype, GUITAR_TYPES[DEFAULT_GTYPE])["label"]
    print(f"Rendering: {title}  ({total} events)  [{gt_label}]")
    samples = render_song(song, duration_per_chord, gtype)
    data = normalize_and_pack(samples)
    secs = len(data) // 2 / SAMPLE_RATE
    print(f"Playing ({secs:.1f}s)...")

    timeline = _build_timeline(song, duration_per_chord)

    try:
        import sounddevice as sd
        audio = np.frombuffer(data, dtype=np.int16) if HAS_NP else list(struct.unpack(f"<{len(data)//2}h", data))
        stop_evt = threading.Event()
        display_thread = threading.Thread(target=_sync_display, args=(timeline, stop_evt), daemon=True)
        sd.play(np.array(audio, dtype=np.int16), SAMPLE_RATE)
        display_thread.start()
        sd.wait()
        stop_evt.set()
        display_thread.join(timeout=2)
    except ImportError:
        out = "song_output.wav"
        save_wav(out, data)
        print(f"Saved: {out}")
    print("Done.")


def save_song_from_json(json_path: str, wav_path: str,
                        duration_per_chord: float | None = None,
                        gtype: str = DEFAULT_GTYPE) -> None:
    song = load_song_json(json_path)
    title = song.get("title", song.get("title_fa", json_path))
    total = sum(len(s.get("chords", s.get("notes", []))) for s in song.get("sections", []))
    gt_label = GUITAR_TYPES.get(gtype, GUITAR_TYPES[DEFAULT_GTYPE])["label"]
    print(f"Rendering: {title}  ({total} events)  [{gt_label}]")
    samples = render_song(song, duration_per_chord, gtype)
    data = normalize_and_pack(samples)
    save_wav(wav_path, data)
    print(f"Saved: {wav_path}  ({len(data) // 2 / SAMPLE_RATE:.1f}s)")


def _ask_guitar_type() -> str:
    print("  Guitar type:")
    for i, (key, gt) in enumerate(GUITAR_TYPES.items(), 1):
        print(f"    {i}. {gt['label']}")
    while True:
        choice = input("  Choose (1/2/3, Enter=classic): ").strip()
        if not choice:
            return DEFAULT_GTYPE
        keys = list(GUITAR_TYPES.keys())
        if choice in ("1", "2", "3"):
            return keys[int(choice) - 1]
        if choice.lower() in GUITAR_TYPES:
            return choice.lower()
        print("  Invalid choice, try again.")


def main():
    print("=" * 45)
    print("  Guitar Synthesizer (Karplus-Strong)")
    print("=" * 45)
    print()
    gtype = _ask_guitar_type()
    gt_label = GUITAR_TYPES[gtype]["label"]
    print(f"\n  Selected: {gt_label}")
    print(f"  Chords: {', '.join(sorted(CHORDS))}")
    print(f"  Type 'T' anytime to switch guitar type.\n")
    while True:
        cmd = input(f"[{gt_label}] Chord / [L]ist / [J] play JSON / [S] save JSON->WAV / [T]ype / [Q]uit: ").strip()
        if not cmd:
            continue
        c = cmd.upper()
        if c == "Q":
            break
        if c == "T":
            gtype = _ask_guitar_type()
            gt_label = GUITAR_TYPES[gtype]["label"]
            print(f"  Switched to: {gt_label}\n")
            continue
        if c == "L":
            print("Chords:", ", ".join(sorted(CHORDS)))
            continue
        if c == "J":
            jpath = input("JSON file path: ").strip()
            if not jpath:
                continue
            try:
                dur_str = input("Seconds per chord (Enter=auto): ").strip()
                dur = float(dur_str) if dur_str else None
            except ValueError:
                dur = None
            try:
                play_song_from_json(jpath, duration_per_chord=dur, gtype=gtype)
            except FileNotFoundError:
                print(f"File not found: {jpath}")
            except Exception as e:
                print(f"Error: {e}")
            continue
        if c == "S":
            jpath = input("JSON file path: ").strip()
            if not jpath:
                continue
            wpath = input("Output WAV path: ").strip()
            if not wpath:
                continue
            try:
                dur_str = input("Seconds per chord (Enter=auto): ").strip()
                dur = float(dur_str) if dur_str else None
            except ValueError:
                dur = None
            try:
                save_song_from_json(jpath, wpath, duration_per_chord=dur, gtype=gtype)
            except FileNotFoundError:
                print(f"File not found: {jpath}")
            except Exception as e:
                print(f"Error: {e}")
            continue
        if _chord_key(cmd) not in CHORDS:
            print(f"Unknown. Use one of: {', '.join(sorted(CHORDS))}")
            continue
        cn = _chord_key(cmd)
        try:
            dur_str = input("Duration [1.0]: ").strip() or "1.0"
            duration = max(0.1, min(10, float(dur_str)))
        except ValueError:
            duration = 1.0
        out = input("Save as (Enter = play only): ").strip()
        path = play_chord_to_wav(cn, duration, out if out else None, gtype)
        print(f"Saved: {path}")
        play_wav(path)


if __name__ == "__main__":
    main()
