"""Phase 5.3a selection-policy study; never a production representative selector.

Rebuild the seven real lifecycle-derived Phase 4 candidate timelines using the
shared evaluator, materialize unchanged requests with Phase 5.1, and obtain every
pixel comparison from Phase 5.2. No production model, detector, threshold, sampling
rule, ROI, or quality definition is modified. All policies, role strings and trial
collapse criteria live in this development script only. There is no OCR, VLM,
embedding, loading classifier, semantic duplicate classifier or production Enum.

Candidate indexes are video-wide indexes into the original flattened materialized
sequence; local indexes and Investigation ownership are also exported. Active
scroll means the preceding physical pair's existing positive evidence inside the
core. Padded dense reasons are reported separately and must not be mistaken for
current motion. Direct map/route reasons come from the existing planner via its
evaluation helper, without reproducing its trigger rules. Consecutive positive
pairs define diagnostic episodes, not new production interaction events.

Transition context means the existing window's requested padding around a real
OPEN/SWITCH/CLOSE boundary. It is NOT a loading label. The few explicitly named
manual transition controls are audit landmarks only, never policy inputs. The
same physical boundary can belong to two candidate windows and depict the next
ticket; exact anchor coverage is therefore not proof of outgoing-ticket coverage.

Study A retains one nearest-confirmation candidate per stable view; B adds exact
core anchors; C adds non-video boundary context and one last direct candidate per
map/route trigger kind; C_scroll adds one middle moving candidate per episode.
D takes the same per-Investigation count as C by raw sharpness alone. E attempts
conservative adjacent-view collapse; E_loose tests an intentionally riskier margin.
Neither E rule is a calibrated semantic classifier. All policies select references
for this experiment only and leave the complete original timeline intact.

Compare seven within-span choices: earliest, middle, latest, sharpness, contrast,
nearest confirmation (ties favor after), and quality among up to three nearest
post-confirmation candidates. The last is deliberately bounded to the same view.
These are probes, not a learned score. Pair work is limited to stable neighbors,
endpoints, probe choices, adjacent stable views, boundary controls, scroll brackets
and selected temporal neighbors. Scalar memoization avoids repeated measurements;
there is no all-pairs expansion or image cache beyond the existing bounded resolver.

Export measured candidate/policy/strategy/pair tables and bounded contact sheets.
Contact sheets show each policy-selected reference once per Investigation, label
all policies/reasons, and add bounded strategy/episode/manual probes. Full-frame
thumbnails show map context; only ticket_scroll_body enters similarity. Enlarged
body/transition audit panels support manual inspection without machine reading.
Coverage is structural metadata coverage, not ground-truth semantic completeness.
Raw-MAD thresholds in redundancy summaries are explicitly study-only pixel proxies.
No downstream readability accuracy, semantic new-text area or selection correctness
can be measured without labels/OCR. Distinguish measured facts from human judgment.
"""

import csv
import hashlib
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from textwrap import wrap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.adaptive_sampling import SamplingSignalEntry
from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.layout import map_roi_to_frame
from events_diary.ticket_view_similarity import compare_ticket_view_pairs
from scripts.evaluation.evaluate_adaptive_sampling_plan import _pair_reasons
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/selection_policy_calibration"
SOURCES = ("combined_interaction", "ticket_scroll", "investigation_switch",
           "investigation_started_active", "investigation_close_reopen", "map_pan", "map_zoom")
FOCUS = {"combined_interaction", "ticket_scroll", "investigation_switch", "map_pan", "map_zoom"}
MAP_REASONS = ("route_visual_change", "map_translation", "map_scale_change")
ANCHORS = ("window_start", "investigation_start", "investigation_end", "window_end")
POLICIES = ("A_stable", "B_anchors", "C_roles", "C_scroll", "D_quality", "E_diversity", "E_loose")
# Exploratory, not production cutoffs: (minimum overlap, maximum |dy|, maximum aligned MAD).
COLLAPSE = {"E_diversity": (.98, 2., 1.), "E_loose": (.89, 64., 1.)}
STRATEGIES = ("earliest", "middle", "latest", "sharpest", "contrast", "confirmed", "local_quality")


def _close(a, b):
    """Compare timestamp representation without merging distinct candidate requests."""
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def _stats(values):
    """Describe finite raw measurements, retaining unavailable values as unavailable."""
    numbers = [v for v in values if v is not None and math.isfinite(v)]
    if not numbers:
        return "unavailable"
    return f"n={len(numbers)} min={min(numbers):.6g} median={np.median(numbers):.6g} max={max(numbers):.6g}"


def _write_csv(name, rows):
    """Save scalar audit rows; no production records or images are serialized."""
    if not rows:
        return
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episodes(timeline, signals, window, reasons, wanted):
    """Group consecutive positive core pairs using existing reasons, without padding."""
    groups = []
    for entry in timeline:
        if not (window.investigation_start_time_seconds <= entry.time_before_seconds
                and entry.time_after_seconds <= window.investigation_end_time_seconds):
            continue
        positive = (signals[entry.pair_index].ticket_scroll_detected if wanted == "scroll"
                    else any(reason in reasons[entry.pair_index] for reason in MAP_REASONS))
        if not positive:
            continue
        if groups and entry.pair_index == groups[-1][-1].pair_index+1:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


def _strategy_choices(group, view, rows):
    """Choose deterministic within-view probes; quality never competes across views."""
    def time(i):
        """Read the original request time without replacing it with physical time."""
        return rows[i]["requested_timestamp"]
    confirmation = view.stable_confirmed_time_seconds
    after = [i for i in group if time(i) >= confirmation or _close(time(i), confirmation)]
    near = sorted(after, key=lambda i: (abs(time(i)-confirmation), time(i), i))[:3]
    return {
        "earliest": group[0], "middle": group[len(group)//2], "latest": group[-1],
        "sharpest": max(group, key=lambda i: (rows[i]["ticket_sharpness"], -time(i), -i)),
        "contrast": max(group, key=lambda i: (rows[i]["ticket_contrast"], -time(i), -i)),
        "confirmed": min(group, key=lambda i: (abs(time(i)-confirmation), time(i) < confirmation, time(i), i)),
        "local_quality": max(near, key=lambda i: (rows[i]["ticket_sharpness"], rows[i]["ticket_contrast"], -time(i), -i)) if near else group[-1],
    }


def _candidate_rows(name, metadata, contexts, phase3, materialized, direct):
    """Export unchanged metadata and clearly separated calibration-only context tags."""
    windows = {w.investigation_index: w for w, _, _, _ in contexts}
    intervals = {i.investigation_index: i for i in phase3["intervals"]}
    local = Counter()
    rows = []
    for index, item in enumerate(materialized):
        c = item.evidence_candidate
        inv = c.investigation_index
        window, interval = windows[inv], intervals[inv]
        time, physical = c.timestamp_seconds, item.frame_index
        if physical is None or item.resolution_status == "unresolved":
            raise RuntimeError(f"Unresolved actual candidate {name} C{index}")
        previous = phase3["timeline"][physical-1] if physical else None
        pair_owned = (previous is not None and previous.time_before_seconds >= interval.start_time_seconds
                      and previous.time_after_seconds <= interval.end_time_seconds)
        active = bool(pair_owned and phase3["signals"][physical-1].ticket_scroll_detected)
        actual_reasons = direct[physical-1] if pair_owned else ()
        transition = ((interval.start_reason.value != "video_started_active"
                       and window.window_start_time_seconds <= time <= interval.start_time_seconds+window.context_after_seconds)
                      or (interval.end_reason.value != "video_end"
                          and interval.end_time_seconds-window.context_before_seconds <= time <= window.window_end_time_seconds))
        roles = [label for label in ("investigation_start", "investigation_end") if label in c.sources]
        if c.stable_view_index is not None:
            roles.append("stable_ticket_view")
        if c.stable_view_follows_scroll:
            roles.append("post_scroll_stable_view")
        if active:
            roles.append("active_scroll_evidence")
        if any(r in c.dense_reasons for r in MAP_REASONS):
            roles.append("route_or_map_activity")
        if transition:
            roles.append("transition_context")
        if "dense_sampling" in c.sources and not roles:
            roles.append("generic_dense_activity")
        # Explicit previously reviewed controls; these do not influence any policy.
        control = ((name == "combined_interaction" and physical == 146)
                   or (name == "investigation_switch" and physical in (15, 16, 48)))
        rows.append({"video_name": metadata.path.name, "investigation_index": inv,
            "candidate_index": index, "local_candidate_index": local[inv],
            "requested_timestamp": time, "actual_timestamp": item.actual_timestamp_seconds,
            "frame_index": physical, "sources": ";".join(c.sources), "dense_reasons": ";".join(c.dense_reasons),
            "stable_view_index": c.stable_view_index, "stable_view_follows_scroll": c.stable_view_follows_scroll,
            "resolution_status": item.resolution_status, "ticket_sharpness": item.ticket_sharpness_laplacian_variance,
            "ticket_brightness": item.ticket_brightness_mean, "ticket_contrast": item.ticket_contrast_std,
            "roles": ";".join(roles), "active_scroll": active, "padded_scroll_context": "ticket_scroll" in c.dense_reasons,
            "direct_reasons": ";".join(actual_reasons), "transition_context": transition,
            "manual_transition_control": control, "distance_from_core_start": time-interval.start_time_seconds,
            "distance_from_core_end": interval.end_time_seconds-time,
            "inside_core": interval.start_time_seconds <= time <= interval.end_time_seconds})
        local[inv] += 1
    return rows


def _policies(owned, views, choices, episodes, rows, interval, measure):
    """Run only trial policies; preserve the complete original candidate sequence."""
    picks = {name: defaultdict(list) for name in POLICIES}

    def add(policy, index, reason):
        """Aggregate study explanations for a reference already chosen by that policy."""
        picks[policy][index].append(reason)

    for vi in range(len(views)):
        index = choices[vi]["confirmed"]
        for policy in ("A_stable", "B_anchors", "C_roles", "C_scroll"):
            add(policy, index, f"stable V{vi}: nearest confirmation")
    for i in owned:
        sources = rows[i]["sources"].split(";")
        for anchor in ("investigation_start", "investigation_end"):
            if anchor in sources:
                add("B_anchors", i, anchor)
                non_video = (interval.start_reason.value != "video_started_active" if anchor == "investigation_start"
                             else interval.end_reason.value != "video_end")
                if non_video:
                    for policy in ("C_roles", "C_scroll"):
                        add(policy, i, anchor+" interaction context (not primary ticket)")
    for reason in MAP_REASONS:
        group = [i for i in owned if reason in rows[i]["direct_reasons"].split(";")]
        if group:
            for policy in ("C_roles", "C_scroll"):
                add(policy, group[-1], "last direct "+reason+" (map context only)")
    for episode in episodes:
        active = [i for i in owned if rows[i]["active_scroll"]
                  and episode[0].time_before_seconds < rows[i]["actual_timestamp"] <= episode[-1].time_after_seconds]
        if active:
            add("C_scroll", active[len(active)//2], "optional active-scroll provenance")
    k = len(picks["C_roles"])
    for i in sorted(owned, key=lambda i: (-rows[i]["ticket_sharpness"], rows[i]["requested_timestamp"], i))[:k]:
        add("D_quality", i, f"raw sharpness top-{k}; no semantic filter")
    for policy, (overlap, shift, mad) in COLLAPSE.items():
        previous = None
        for vi in range(len(views)):
            index = choices[vi]["confirmed"]
            comparison = measure(previous, index, "consecutive_view_collapse") if previous is not None else None
            collapse = (comparison is not None and comparison["alignment_valid"]
                        and comparison["overlap_fraction"] >= overlap
                        and abs(comparison["dy_pixels"]) <= shift and comparison["aligned_mad"] <= mad)
            if not collapse:
                add(policy, index, f"stable V{vi}; not collapsed by study-only {overlap}/{shift}px/{mad}MAD")
            # Compare consecutive input views, not an expanding representative search.
            previous = index
    return picks


def _policy_summary(name, inv, policy, picks, owned, groups, episodes, map_episodes, rows, measure):
    """Measure structural coverage and bounded pixel redundancy, not semantic correctness."""
    selected = sorted(picks)
    covered = {rows[i]["stable_view_index"] for i in selected if rows[i]["stable_view_index"] is not None}
    anchors = [i for i in owned if any(a in rows[i]["sources"].split(";") for a in ANCHORS)]
    core_anchors = [i for i in anchors if "investigation_start" in rows[i]["sources"] or "investigation_end" in rows[i]["sources"]]
    selected_pairs = [measure(a, b, "selected_neighbors") for a, b in pairwise(selected)]
    # Study-only proxy, never called a semantic duplicate detector.
    low = sum(p["raw_mad"] <= 1 and p["raw_cpr"] <= .01 for p in selected_pairs)
    return {"video_name": name, "investigation_index": inv, "policy_name": policy,
        "original_count": len(owned), "selected_count": len(selected),
        "reduction_percent": 100*(1-len(selected)/len(owned)),
        "stable_views_available": len(groups), "stable_views_covered": len(covered),
        "post_scroll_views_available": sum(rows[g[0]]["stable_view_follows_scroll"] for g in groups.values()),
        "post_scroll_views_covered": sum(any(rows[i]["stable_view_index"] == vi and rows[i]["stable_view_follows_scroll"] for i in selected) for vi in groups),
        "anchors_available": len(anchors), "anchors_covered": len(set(anchors)&set(selected)),
        "core_anchors_available": len(core_anchors), "core_anchors_covered": len(set(core_anchors)&set(selected)),
        "scroll_episodes_available": len(episodes),
        "scroll_episodes_represented": sum(any(rows[i]["active_scroll"] and ep[0].time_before_seconds < rows[i]["actual_timestamp"] <= ep[-1].time_after_seconds for i in selected) for ep in episodes),
        "map_episodes_available": len(map_episodes),
        "map_episodes_represented": sum(any(rows[i]["direct_reasons"] and any(r in rows[i]["direct_reasons"].split(";") for r in MAP_REASONS)
            and ep[0].time_before_seconds < rows[i]["actual_timestamp"] <= ep[-1].time_after_seconds for i in selected) for ep in map_episodes),
        "map_direct_candidates_selected": sum(any(r in rows[i]["direct_reasons"].split(";") for r in MAP_REASONS) for i in selected),
        "selected_neighbor_pairs": len(selected_pairs), "low_raw_neighbor_pairs_study_proxy": low,
        "repeated_stable_view_selections": sum(max(0, sum(rows[i]["stable_view_index"] == vi for i in selected)-1) for vi in groups),
        "active_scroll_selected": sum(rows[i]["active_scroll"] for i in selected),
        "transition_context_selected": sum(rows[i]["transition_context"] for i in selected),
        "manual_transition_controls_selected": sum(rows[i]["manual_transition_control"] for i in selected),
        "outside_core_selected": sum(not rows[i]["inside_core"] for i in selected),
        "unique_physical_frames_selected": len({rows[i]["frame_index"] for i in selected}),
        "ticket_sharpness_min": min((rows[i]["ticket_sharpness"] for i in selected), default=None),
        "ticket_sharpness_median": float(np.median([rows[i]["ticket_sharpness"] for i in selected])) if selected else None,
        "ticket_sharpness_max": max((rows[i]["ticket_sharpness"] for i in selected), default=None),
        "ticket_brightness_median": float(np.median([rows[i]["ticket_brightness"] for i in selected])) if selected else None,
        "ticket_contrast_median": float(np.median([rows[i]["ticket_contrast"] for i in selected])) if selected else None}


def _caption(image, text, y, *, width=72):
    """Draw bounded English debug captions without changing measured image pixels."""
    for line in wrap(text, width=width):
        cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, .35, (20, 20, 20), 1, cv2.LINE_AA)
        y += 14
    return y


def _contact_sheets(name, inv, indexes, labels, rows, materialized, resolver, config):
    """Save small paginated visual probes, each physical request once per context."""
    paths = []
    for page in range(0, len(indexes), 8):
        subset = indexes[page:page+8]
        tiles = []
        for i in subset:
            handle = resolver.resolve(materialized[i].requested_timestamp_seconds)
            if handle.image is None:
                raise RuntimeError("Contact-sheet decode failed")
            frame = handle.image
            body = map_roi_to_frame(config.ticket_scroll_body, config, frame.shape[1], frame.shape[0]).crop(frame)
            tile = np.full((978, 384, 3), 245, np.uint8)
            y = _caption(tile, f"{name} I{inv} C{i} t={rows[i]['requested_timestamp']:.2f}s frame={rows[i]['frame_index']}", 14, width=56)
            y = _caption(tile, f"V{rows[i]['stable_view_index']} sharp={rows[i]['ticket_sharpness']:.1f} {rows[i]['roles']}", y, width=56)
            policy_labels = [p for p in POLICIES if any(label.startswith(p+":") for label in labels[i])]
            y = _caption(tile, "Policies: "+", ".join(policy_labels), y, width=56)
            reasons = []
            for token, short in (("nearest confirmation", "confirmed stable"), ("interaction context", "boundary context"),
                                 ("map context", "direct map/route"), ("provenance", "scroll provenance"),
                                 ("raw sharpness", "quality top-K"), ("not collapsed", "diversity trial")):
                if any(token in label for label in labels[i]):
                    reasons.append(short)
            y = _caption(tile, "Selection reasons: "+", ".join(reasons), y, width=56)
            probes = [label for label in labels[i] if not any(label.startswith(p+":") for p in POLICIES)]
            _caption(tile, "Probes: "+" | ".join(probes), y, width=56)
            # Complete source labels/reasons remain in the CSV even on dense headers.
            tile[196:412] = cv2.resize(frame, (384, 216))
            tile[412:978, 24:360] = body
            tiles.append(tile)
        sheet = np.full((978*((len(tiles)+3)//4), 384*4, 3), 245, np.uint8)
        for j, tile in enumerate(tiles):
            sheet[(j//4)*978:(j//4+1)*978, (j%4)*384:(j%4+1)*384] = tile
        path = OUTPUT / f"{name}_i{inv}_policies_{page//8+1:02d}.png"
        if not cv2.imwrite(str(path), sheet):
            raise OSError(f"Cannot save {path}")
        paths.append(path.name)
    return paths


def _evaluate(name):
    """Run real upstream outputs once, then all study probes with a bounded resolver."""
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
    if anomalies:
        raise RuntimeError(f"Unexpected lifecycle anomalies: {anomalies}")
    config = UIConfig()
    candidates = tuple(c for window, plan, views, _ in contexts
                       for c in build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views))
    snapshot = tuple(asdict(c) for c in candidates)
    direct = tuple(_pair_reasons(SamplingSignalEntry(e.pair_index, e.time_before_seconds, e.time_after_seconds, phase3["signals"][e.pair_index]), metadata.fps)
                   for e in phase3["timeline"])
    summaries, results, strategies, pair_rows, lines, images = [], [], [], [], [], []
    with VideoFrameResolver(metadata.path, metadata=metadata, max_frames=16) as resolver:
        materialized = materialize_evidence_candidates(video_path=metadata.path, candidates=candidates, ui_config=config, resolver=resolver)
        rows = _candidate_rows(name, metadata, contexts, phase3, materialized, direct)
        memo = {}

        def measure(a, b, purpose):
            """Memoize scalar production comparisons only, for explicitly requested pairs."""
            key = (a, b)
            if key not in memo:
                comparison = compare_ticket_view_pairs(video_path=metadata.path, materialized_candidates=materialized,
                    pairs=[key], ui_config=config, resolver=resolver)[0]
                row = {"video_name": metadata.path.name, "investigation_index": rows[a]["investigation_index"],
                    "timestamp_a": rows[a]["requested_timestamp"], "timestamp_b": rows[b]["requested_timestamp"],
                    **asdict(comparison), "alignment_valid": comparison.alignment_valid,
                    "geometric_new_fraction_proxy": 1-comparison.overlap_fraction if comparison.overlap_fraction is not None else None,
                    "purposes": purpose}
                memo[key] = row
            elif purpose not in memo[key]["purposes"].split(";"):
                memo[key]["purposes"] += ";"+purpose
            return memo[key]

        for window, plan, views, observed in contexts:
            inv = window.investigation_index
            interval = next(i for i in phase3["intervals"] if i.investigation_index == inv)
            owned = [i for i, row in enumerate(rows) if row["investigation_index"] == inv]
            groups = {vi: [i for i in owned if rows[i]["stable_view_index"] == vi] for vi in range(len(views))}
            choices = {vi: _strategy_choices(group, views[vi], rows) for vi, group in groups.items()}
            episodes = _episodes(phase3["timeline"], phase3["signals"], window, direct, "scroll")
            map_episodes = _episodes(phase3["timeline"], phase3["signals"], window, direct, "map")
            lines.append(f"\n{name} I{inv}: core={interval.start_time_seconds:g}..{interval.end_time_seconds:g}s "
                f"({interval.start_reason.value}/{interval.end_reason.value}); candidates={len(owned)} "
                f"unique_physical={len({rows[i]['frame_index'] for i in owned})}; stable={sum(len(g) for g in groups.values())}; "
                f"views={len(views)}; active_scroll={sum(rows[i]['active_scroll'] for i in owned)}; "
                f"anchors={sum(any(a in rows[i]['sources'].split(';') for a in ANCHORS) for i in owned)}; "
                f"padded_map={sum(any(r in rows[i]['dense_reasons'].split(';') for r in MAP_REASONS) for i in owned)}; "
                f"direct_map={sum(any(r in rows[i]['direct_reasons'].split(';') for r in MAP_REASONS) for i in owned)}")
            labels = defaultdict(list)
            for vi, group in groups.items():
                view = views[vi]
                neighbors = [measure(a, b, "stable_neighbors") for a, b in pairwise(group)]
                endpoints = measure(group[0], group[-1], "stable_endpoints")
                end = view.stable_end_time_seconds if view.stable_end_time_seconds is not None else observed
                lines.append(f"  V{vi}: n={len(group)} start={view.stable_start_time_seconds:g} confirmed={view.stable_confirmed_time_seconds:g} "
                    f"observed_end={end} censored={view.stable_end_time_seconds is None} duration_lower_bound={end-view.stable_start_time_seconds:g} "
                    f"post_scroll={view.follows_scroll}; sharpness {_stats([rows[i]['ticket_sharpness'] for i in group])}; "
                    f"neighbor rawMAD {_stats([p['raw_mad'] for p in neighbors])}; endpoint rawMAD={endpoints['raw_mad']:.6g}")
                for strategy, i in choices[vi].items():
                    ref = choices[vi]["confirmed"]
                    pixel = measure(i, ref, "strategy_vs_confirmation")
                    strategies.append({"video_name": metadata.path.name, "investigation_index": inv, "stable_view_index": vi,
                        "strategy": strategy, "candidate_index": i, "requested_timestamp": rows[i]["requested_timestamp"],
                        "stable_start": view.stable_start_time_seconds, "stable_confirmed": view.stable_confirmed_time_seconds,
                        "observed_end": end, "candidate_count_in_view": len(group), "follows_scroll": view.follows_scroll,
                        "before_confirmation": rows[i]["requested_timestamp"] < view.stable_confirmed_time_seconds-1e-9,
                        "active_scroll": rows[i]["active_scroll"], "transition_context": rows[i]["transition_context"],
                        "manual_transition_control": rows[i]["manual_transition_control"],
                        "ticket_sharpness": rows[i]["ticket_sharpness"], "ticket_brightness": rows[i]["ticket_brightness"],
                        "ticket_contrast": rows[i]["ticket_contrast"], "raw_mad_vs_confirmed": pixel["raw_mad"]})
                    labels[i].append(f"V{vi}:{strategy}")
                lines.append("    strategies="+str({s: (i, rows[i]["requested_timestamp"]) for s, i in choices[vi].items()}))
            for left, right in pairwise(range(len(views))):
                p = measure(choices[left]["confirmed"], choices[right]["confirmed"], "consecutive_stable_views")
                temporal_gap = views[right].stable_start_time_seconds-(views[left].stable_end_time_seconds or interval.end_time_seconds)
                lines.append(f"  Consecutive V{left}->V{right} gap={temporal_gap:g}s: "+str(p))
            if len(views) > 2:
                p = measure(choices[0]["confirmed"], choices[len(views)-1]["confirmed"], "stable_return_control")
                lines.append("  Bounded first/last stable-position return control: "+str(p))
            for ei, episode in enumerate(episodes):
                start, end = episode[0].time_before_seconds, episode[-1].time_after_seconds
                active = [i for i in owned if rows[i]["active_scroll"] and start < rows[i]["actual_timestamp"] <= end]
                before = [vi for vi, v in enumerate(views) if v.stable_start_time_seconds < start and v.stable_confirmed_time_seconds <= start+1e-9]
                after = [vi for vi, v in enumerate(views) if v.stable_start_time_seconds >= end-1e-9]
                brackets = ([choices[before[-1]]["confirmed"]] if before else []) + ([choices[after[0]]["confirmed"]] if after else [])
                probes = [measure(i, stable, "scroll_vs_bracketing_stable") for i in active for stable in brackets]
                for i in ([active[0], active[len(active)//2], active[-1]] if active else []):
                    labels[i].append(f"scroll episode {ei} probe")
                lines.append(f"  Scroll episode {ei}: {start:g}..{end:g}s; active candidates={active}; beforeV={before[-1] if before else None}; "
                    f"afterV={after[0] if after else None}; bracket fits valid={sum(p['alignment_valid'] for p in probes)}/{len(probes)}; "
                    f"active sharpness {_stats([rows[i]['ticket_sharpness'] for i in active])}. "
                    "Unmatched area is not semantic new text; rejected distant fits leave coverage unknown.")
            for ei, episode in enumerate(map_episodes):
                pool = [i for i in owned if episode[0].time_before_seconds < rows[i]["actual_timestamp"] <= episode[-1].time_after_seconds]
                lines.append(f"  Map/route direct episode {ei}: {episode[0].time_before_seconds:g}..{episode[-1].time_after_seconds:g}s; candidates={pool}")
                if pool:
                    for i in (pool[0], pool[len(pool)//2], pool[-1]):
                        labels[i].append(f"map episode {ei}: first/middle/last")
            for i in owned:
                if any(a in rows[i]["sources"].split(";") for a in ANCHORS):
                    labels[i].append("anchor "+"/".join(a for a in ANCHORS if a in rows[i]["sources"].split(";")))
                    stable_pool = [choices[vi]["confirmed"] for vi in choices]
                    if stable_pool:
                        nearest = min(stable_pool, key=lambda j: abs(rows[i]["requested_timestamp"]-rows[j]["requested_timestamp"]))
                        p = measure(i, nearest, "anchor_vs_nearest_stable")
                        lines.append(f"  Anchor C{i} {rows[i]['sources']} t={rows[i]['requested_timestamp']:g}: nearest stable C{nearest}, "
                            f"rawMAD={p['raw_mad']:.6g}; owner_in_core={rows[i]['inside_core']}; boundary may depict incoming/absent ticket.")
                if rows[i]["manual_transition_control"]:
                    labels[i].append("manual transition audit ONLY")
                    lines.append("  Named manual control: "+str(rows[i]))
            picks = _policies(owned, views, choices, episodes, rows, interval, measure)
            for policy in POLICIES:
                summaries.append(_policy_summary(metadata.path.name, inv, policy, picks[policy], owned, groups, episodes, map_episodes, rows, measure))
                lines.append("  POLICY "+str(summaries[-1]))
                for i in owned:
                    selected = i in picks[policy]
                    reason = "; ".join(picks[policy][i]) if selected else "not selected by this study policy"
                    results.append({**rows[i], "policy_name": policy, "selected": selected, "selection_reason": reason})
                    if selected:
                        labels[i].append(policy+": "+reason)
            # A compact union avoids rendering the same policy/strategy reference repeatedly.
            if name in FOCUS:
                images.extend(_contact_sheets(name, inv, sorted(labels), labels, rows, materialized, resolver, config))
        pair_rows.extend(memo.values())
        lines.append(f"{name}: scalar comparison cache={len(memo)}; resolver decodes={resolver.decode_attempts}, "
                     f"hits={resolver.cache_hits}, retained={resolver.cache_size}/16")
    if snapshot != tuple(asdict(c) for c in candidates):
        raise AssertionError("Input candidates mutated")
    print(f"{name}: candidates={len(rows)}, Investigations={len(contexts)}, bounded comparisons={len(pair_rows)}, sheets={len(images)}", flush=True)
    return rows, results, summaries, strategies, pair_rows, lines, images


def _aggregate(candidates, summaries, strategies, comparisons, lines):
    """Summarize measured policy differences and explicit limits of structural proxies."""
    lines.append("\nMEASURED aggregate policy comparison (original request denominator, overlaps retained):")
    for policy in POLICIES:
        group = [r for r in summaries if r["policy_name"] == policy]
        selected, original = sum(r["selected_count"] for r in group), sum(r["original_count"] for r in group)
        lines.append(f"{policy}: selected={selected}/{original}, reduction={100*(1-selected/original):.3f}%; "
            f"per Investigation={[r['selected_count'] for r in group]}; "
            f"stable coverage={sum(r['stable_views_covered'] for r in group)}/{sum(r['stable_views_available'] for r in group)}; "
            f"post-scroll={sum(r['post_scroll_views_covered'] for r in group)}/{sum(r['post_scroll_views_available'] for r in group)}; "
            f"core anchors={sum(r['core_anchors_covered'] for r in group)}/{sum(r['core_anchors_available'] for r in group)}; "
            f"active-scroll selected={sum(r['active_scroll_selected'] for r in group)}; "
            f"scroll episodes={sum(r['scroll_episodes_represented'] for r in group)}/{sum(r['scroll_episodes_available'] for r in group)}; "
            f"map episodes={sum(r['map_episodes_represented'] for r in group)}/{sum(r['map_episodes_available'] for r in group)}; "
            f"manual transition selected={sum(r['manual_transition_controls_selected'] for r in group)}; "
            f"transition context selected={sum(r['transition_context_selected'] for r in group)}; "
            f"repeated same-view picks={sum(r['repeated_stable_view_selections'] for r in group)}; "
            f"low-raw selected-neighbor proxy={sum(r['low_raw_neighbor_pairs_study_proxy'] for r in group)}/{sum(r['selected_neighbor_pairs'] for r in group)}")
    lines.append("\nMEASURED within-view strategy comparison (content/roles fixed before quality):")
    for strategy in STRATEGIES:
        group = [r for r in strategies if r["strategy"] == strategy]
        lines.append(f"{strategy}: probes={len(group)} before_confirmation={sum(r['before_confirmation'] for r in group)} "
            f"active_scroll={sum(r['active_scroll'] for r in group)} transition_context={sum(r['transition_context'] for r in group)} "
            f"manual_controls={sum(r['manual_transition_control'] for r in group)}; "
            f"sharpness {_stats([r['ticket_sharpness'] for r in group])}; rawMAD vs confirmed {_stats([r['raw_mad_vs_confirmed'] for r in group])}")
    neighbors = [p for p in comparisons if "stable_neighbors" in p["purposes"].split(";")]
    lines.append("Stable temporal-neighbor rawMAD "+_stats([p["raw_mad"] for p in neighbors]))
    lines.append(f"All actual candidates={len(candidates)}; unique (video,physical)={len({(r['video_name'],r['frame_index']) for r in candidates})}; "
        f"stable-associated={sum(r['stable_view_index'] is not None for r in candidates)}; "
        f"stable views={len(strategies)//len(STRATEGIES)}; comparisons={len(comparisons)} (not exhaustive).")
    lines.extend(["\nStudy criteria and interpretation boundaries:",
        "E_diversity: accepted fit, overlap>=.98, |dy|<=2px, alignedMAD<=1. E_loose: overlap>=.89, |dy|<=64px, alignedMAD<=1. Neither criterion is production-approved.",
        "Low-raw selected-neighbor proxy: MAD<=1 AND CPR<=.01. This is pixel redundancy, not semantic identity; candidate/physical duplicates and same-view repeats are separately reported.",
        "C chooses the last actual direct candidate per map trigger kind, not every motion episode: missed map episodes remain visible in coverage counts.",
        "C boundary slots are interaction context; exact core-end anchors can show an incoming/absent ticket. They must not automatically become primary outgoing-ticket OCR evidence.",
        "Transition context is lifecycle padding, not loading detection; some genuinely settled incoming stable frames remain in that context.",
        "A stable association at an inclusive endpoint may coexist with preceding active motion. No policy score resolves that ambiguity.",
        "Geometric new fraction=1-overlap only for accepted direct fits. It is not new semantic text; unavailable fits are not assigned zero new content.",
        "Manual contact-sheet findings and the answers/recommendation are recorded separately after inspection. No labeled selection benchmark exists.",
        "No production selector, Enum, OCR, VLM, semantic classifier, Phase5.3b code or production/test/documentation modification."])


def main():
    """Write reproducible calibration evidence and fail visibly without production repair."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 5.3a Selection Policy Calibration - measurement tables and trial policies",
        "MEASURED means structural counts/pixel metrics; JUDGMENT is explicitly manual interpretation, never labeled ground truth.",
        "All seven sources use actual lifecycle Investigations, unchanged Phase4 candidates and production Phase5.1/5.2 APIs.",
        "Indexes are video-wide; actual frame time and requested time remain distinct. No candidate is removed from the original tables.",
        "Role priority hypothesis: coverage, then diversity, then quality within a role/content group. No combined quality/semantic score.",
        "Study policies and thresholds are confined to this script. Existing production defaults remain unchanged."]
    tables = [[] for _ in range(5)]
    images = []
    try:
        for name in SOURCES:
            *measured, section, sheets = _evaluate(name)
            for table, rows in zip(tables, measured, strict=True):
                table.extend(rows)
            lines.extend(section)
            images.extend(sheets)
        candidates, _results, summaries, strategies, comparisons = tables
        _aggregate(candidates, summaries, strategies, comparisons, lines)
        for filename, rows in zip(("selection_policy_candidates.csv", "selection_policy_results.csv", "selection_policy_summary.csv",
                                   "selection_policy_view_strategies.csv", "selection_policy_comparisons.csv"), tables, strict=True):
            _write_csv(filename, rows)
        lines.append("\nContact-sheet files (all selected policy references + bounded strategy/context probes): "+str(images))
        fingerprint = hashlib.sha256((OUTPUT / "selection_policy_results.csv").read_bytes()).hexdigest()
        review = OUTPUT / "selection_policy_manual_review.txt"
        if review.exists() and review.read_text(encoding="utf-8").startswith("results_sha256: "+fingerprint):
            lines.append("\nMANUAL REVIEW of this exact policy-results snapshot:\n"+review.read_text(encoding="utf-8"))
        else:
            lines.append("\nManual review pending for results_sha256: "+fingerprint+
                         ". Do not treat structural measurements as semantic selection ground truth.")
    except (ValueError, TypeError, OSError, RuntimeError, AssertionError) as error:
        lines.append(f"STOP: {type(error).__name__}: {error}; do not repair production or invent missing semantics")
        (OUTPUT / "selection_policy_calibration_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(lines[-1], flush=True)
        return 1
    report = OUTPUT / "selection_policy_calibration_report.txt"
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Report: {report}", flush=True)
    print(f"Candidates={len(candidates)} Investigations={len(summaries)//len(POLICIES)} policies={len(POLICIES)} comparisons={len(comparisons)} contact sheets={len(images)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
