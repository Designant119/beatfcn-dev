"""Judge-line show layer: turn a single-static-line event chart into a multi-line staged chart.

BeatFCN r5 outputs rhythm events (onset / type / tail); `phira_layout` then puts every note on
ONE non-moving judge line. This module is the missing performance stage. It reads the finished
positioned events and decides:

  * how many lines are lit at each moment (a small pool, only 1-4 visible at a time),
  * where the carrying line sits and how it is tilted (each song section gets its own pose),
  * which support lines fade in for the loud sections.

Rules taken from 9.6-9.8 rated reference charts:
  - stage a pool of lines, but keep only 1-4 visible at once (measured median 1.8-12.7),
  - leave real holes: quiet sections get no support line at all,
  - poses change in one hard cut (0.10-0.15 s), never a slow drift,
  - the carrying line stays readable: tilt <= 25 deg so notes keep falling straight.

Input events: [{"t_us": int, "end_us": int, "type": "tap|hold|flick|drag", "x": float}]
Output: RPE judgeLineList (list of dicts), ready for chart["judgeLineList"].

Pure stdlib. bpm is a serialization clock only (60 => one beat == one second).
"""
from __future__ import annotations

import math
from collections import defaultdict

KINDS = {"tap": 1, "hold": 2, "flick": 3, "drag": 4}

# song sections in normalised time: (until_fraction, carrying-line y, tilt, support lines)
SECTIONS = (
    (0.09, -200.0, 0.0, 0),      # intro: one line, low, nothing else
    (0.32, 0.0, 0.0, 1),         # verse 1
    (0.37, -300.0, -12.0, 0),    # transition drop: line falls away, no support
    (0.49, 60.0, 0.0, 1),        # verse 2
    (0.58, 240.0, 0.0, 0),       # quiet valley: line high up, deliberately empty
    (0.84, -120.0, 16.0, 3),     # chorus: line mid, tilted, three support lines
    (1.00, -200.0, 0.0, 0),      # outro: back to the lonely line
)

SUPPORT_SLOTS = (
    (-420.0, -18.0, 45),
    (330.0, 0.0, 55),
    (-480.0, 12.0, 40),
    (460.0, -8.0, 50),
)


def _beat(seconds, bpm):
    return seconds * bpm / 60.0


def _bt(v, den=19200):
    total = round(v * den)
    whole = int(total // den)
    num = total - whole * den
    if num == 0:
        return [whole, 0, 1]
    g = math.gcd(num, den)
    return [whole, int(num // g), int(den // g)]


def _ev(v0, v1, b0, b1, easing=1):
    return {
        "startTime": _bt(b0), "endTime": _bt(b1),
        "start": round(v0, 3), "end": round(v1, 3),
        "easingType": easing, "easingLeft": 0.0, "easingRight": 1.0,
        "bezier": 0, "bezierPoints": [0.0, 0.0, 0.0, 0.0], "linkgroup": 0.0,
    }


def _alpha_chain(points, bpm):
    """points: [(second, alpha, easing_into_next)] -> alphaEvents"""
    return [_ev(points[i][1], points[i + 1][1], _beat(points[i][0], bpm),
                _beat(points[i + 1][0], bpm), points[i][2])
            for i in range(len(points) - 1)]


def _dedupe(points):
    out = []
    for p in sorted(points, key=lambda q: q[0]):
        if not out or p[0] - out[-1][0] > 1e-4:
            out.append(tuple(p))
        else:
            out[-1] = tuple(p)
    return out


def _pose_keys(dur, cuts, hard_cut=0.12):
    """[(t, y, tilt)] with a hard cut: hold the previous pose, then jump in hard_cut seconds."""
    keys = [(0.0, cuts[0][1], cuts[0][2])]
    for t, y, tilt, _count in cuts[1:]:
        keys.append((max(t - 0.001, 0.0), keys[-1][1], keys[-1][2]))
        keys.append((t + hard_cut, y, tilt))
    keys.append((dur, keys[-1][1], keys[-1][2]))
    return keys


def build_show(events, *, bpm=60.0, max_support=4):
    """events -> judgeLineList. One carrying line holds every note; support lines only decorate."""
    if not events:
        raise ValueError("no events")
    events = sorted(events, key=lambda e: e["t_us"])
    dur = max(e["end_us"] for e in events) / 1e6 or 1.0
    cuts = [(frac * dur, y, tilt, sup) for frac, y, tilt, sup in SECTIONS]
    cuts[-1] = (dur, cuts[-1][1], cuts[-1][2], cuts[-1][3])

    keys = _pose_keys(dur, cuts)
    my = [_ev(keys[i][1], keys[i + 1][1], _beat(keys[i][0], bpm), _beat(keys[i + 1][0], bpm),
              8 if keys[i + 1][0] - keys[i][0] < 0.3 else 1) for i in range(len(keys) - 1)]
    mr = [_ev(keys[i][2], keys[i + 1][2], _beat(keys[i][0], bpm), _beat(keys[i + 1][0], bpm),
              8 if keys[i + 1][0] - keys[i][0] < 0.3 else 1) for i in range(len(keys) - 1)]
    mx = [_ev(0.0, 0.0, 0.0, _beat(dur, bpm), 1)]

    # carrying line brightness: dim in quiet parts, bright in the chorus
    main_alpha = [(0.0, 0, 4), (0.6, 150, 4)]
    for frac, _y, _tilt, _sup in SECTIONS[1:-1]:
        t = frac * dur
        level = 190 if 0.32 <= frac <= 0.84 else 120
        main_alpha += [(t - 0.001, 150, 4), (t + 0.10, level, 1)]
    main_alpha += [(dur - 1.6, 150, 4), (dur, 0, 4)]
    main_alpha = _dedupe(main_alpha)

    lines = [{
        "Group": 0, "Name": "Carrying line", "Texture": "line.png",
        "alphaControl": [{"alpha": 1.0, "easing": 1, "x": 0.0}], "bpmfactor": 1.0,
        "eventLayers": [{"moveXEvents": mx, "moveYEvents": my, "rotateEvents": mr,
                         "alphaEvents": _alpha_chain(main_alpha, bpm),
                         "speedEvents": [_ev(9, 9, 0, _beat(dur, bpm), 1)]}],
        "father": -1, "group": 0, "isCover": 0,
        "notes": [{
            "type": KINDS.get(e["type"], 1), "startTime": _bt(_beat(e["t_us"] / 1e6, bpm)),
            "endTime": _bt(_beat(e["end_us"] / 1e6, bpm)), "positionX": round(float(e["x"]), 1),
            "above": 1, "isFake": 0, "speed": 1.0, "size": 1.0,
            "yOffset": 0.0, "visibleTime": 999999.0, "alpha": 255,
        } for e in events],
        "posControl": [{"easing": 1, "pos": 1.0, "x": 0.0}],
        "sizeControl": [{"easing": 1, "size": 1.0, "x": 0.0}],
        "skewControl": [{"easing": 1, "skew": 0.0, "x": 0.0}],
        "yControl": [{"easing": 1, "x": 0.0, "y": 1.0}],
        "zOrder": 0,
    }]

    # support lines: fade in only for sections that ask for them, never carry notes
    for idx, (sy, sang, salpha) in enumerate(SUPPORT_SLOTS[:max_support]):
        pts = [(0.0, 0, 4)]
        for frac, _y, _tilt, count in SECTIONS:
            t = frac * dur
            pts.append((max(t - 0.001, 0.0), pts[-1][1], 4))
            pts.append((t + 0.5, salpha if idx < count else 0, 4))
        pts.append((dur, pts[-1][1], 1))
        pts = _dedupe(pts)
        lines.append({
            "Group": 0, "Name": f"Support {idx + 1}", "Texture": "line.png",
            "alphaControl": [{"alpha": 1.0, "easing": 1, "x": 0.0}], "bpmfactor": 1.0,
            "eventLayers": [{
                "moveXEvents": [_ev(0, 0, 0, _beat(dur, bpm), 1)],
                "moveYEvents": [_ev(sy, sy, 0, _beat(dur, bpm), 1)],
                "rotateEvents": [_ev(sang, sang, 0, _beat(dur, bpm), 1)],
                "alphaEvents": _alpha_chain(pts, bpm),
                "speedEvents": [_ev(9, 9, 0, _beat(dur, bpm), 1)],
            }],
            "father": -1, "group": 0, "isCover": 0, "notes": [],
            "posControl": [{"easing": 1, "pos": 1.0, "x": 0.0}],
            "sizeControl": [{"easing": 1, "size": 1.0, "x": 0.0}],
            "skewControl": [{"easing": 1, "skew": 0.0, "x": 0.0}],
            "yControl": [{"easing": 1, "x": 0.0, "y": 1.0}],
            "zOrder": idx + 1,
        })
    return lines


def visible_probe(lines, dur, bpm=60.0, step=0.5):
    """Rough check: how many lines are lit over time (alpha > 20)."""
    out = []
    t = 0.0
    while t <= dur:
        n = 0
        for ln in lines:
            for e in ln["eventLayers"][0]["alphaEvents"]:
                b0 = e["startTime"][0] + e["startTime"][1] / e["startTime"][2]
                b1 = e["endTime"][0] + e["endTime"][1] / e["endTime"][2]
                t0, t1 = b0 * 60.0 / bpm, b1 * 60.0 / bpm
                if t0 <= t <= t1:
                    v = e["start"] if t1 - t0 <= 0 else e["start"] + (e["end"] - e["start"]) * (t - t0) / (t1 - t0)
                    if v > 20:
                        n += 1
                    break
        out.append((round(t, 1), n))
        t += step
    return out
