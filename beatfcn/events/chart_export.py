"""Helpers for consuming charter-supplied timing.

``timing_points_from_map`` turns a canonical TimingMap into ``[(time_ms, bpm)]``
on the audio clock, for callers that keep the charter's own timing points.

Chart writing does not live here: osu!mania export is
``beatfcn.events.osu_mania.export_osu`` and Phira preview output is
``beatfcn.events.rhythm_preview.write_phira_preview``.
"""
from __future__ import annotations


def timing_points_from_map(timing):
    """Canonical TimingMap -> [(time_ms, bpm)] on the audio clock (for charter-supplied timing)."""
    return [(float((s.start_sec+timing.offset_sec)*1000), float(s.bpm)) for s in timing.segments]
