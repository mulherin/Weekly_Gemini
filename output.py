# output.py
import os
import json
import re
from typing import Dict, Any, List
from datetime import datetime

from core import (
    ensure_dir, save_json, save_text, clean_bullets_list,
    story_sign, make_pos_neg_neutral, parse_date
)

# --- Obsidian Markdown helpers (tabs for nested bullets; underscores italic) ---
def _md_bullet(level: int, text: str) -> str:
    return "\t"*max(0, level) + "- " + (text or "")

def _md_italicize(text: str) -> str:
    s = (text or "").replace("_", "\\_")
    return "_" + s + "_"

# --- Local helpers for provenance tags ---------------------------------------
def _bar_from_tilt(tilt_map: Dict[str, Dict[str, Any]], tkr: str) -> str:
    val = (tilt_map.get(tkr, {}) or {}).get("tilt", "Neutral")
    return "High" if val == "High" else ("Low" if val == "Low" else "Mid")


def _short_title(title: str, n_words: int = 5) -> str:
    """First n words of title, with ellipsis if trimmed."""
    if not title:
        return ""
    words = re.split(r"\s+", str(title).strip())
    return " ".join(words[:n_words]) + ("..." if len(words) > n_words else "")

def _latest_title_date_tag_for_ticker(
    t: str,
    stock_digest: Dict[str, Any],
    flow_digest: Dict[str, Any],
) -> str:
    """Pick freshest stock item (fallback: flow) and return [TitleFirst5, YYYY-MM-DD]."""
    def pick(items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        # keep only items with a date; sort newest first
        dated = [it for it in items if (it.get("as_of") or "").strip()]
        if not dated:
            return ""
        dated.sort(key=lambda it: parse_date(it.get("as_of") or ""), reverse=True)
        it0 = dated[0]
        ts = _short_title(it0.get("title") or "")
        dt = (it0.get("as_of") or "").strip()
        if ts and dt:
            return f"[{ts}, {dt}]"
        if ts:
            return f"[{ts}]"
        if dt:
            return f"[{dt}]"
        return ""

    tag = pick((stock_digest.get(t, {}) or {}).get("items", []) or [])
    if not tag:
        tag = pick((flow_digest.get(t, {}) or {}).get("items", []) or [])
    return tag

def fallback_stock_paragraph(t: str, universe_names: Dict[str, str], flow_digest: Dict[str, Any], stock_digest: Dict[str, Any]) -> str:
    q = stock_digest.get(t, {}).get("signal_quality", {}) or {}
    u = q.get("unique_dates", 0)
    fct = len(flow_digest.get(t, {}).get("items", []))
    name = universe_names.get(t, t)
    if u == 0 and fct == 0:
        return f"No fresh stock-window evidence for {name}. Keeping context minimal until more data is ingested."
    if u == 0:
        return f"Thin stock-window evidence this week; near-term read relies on incremental flow datapoints."
    return f"Limited stock-window depth; use flow and recent signals to gauge near-term cadence."

def write_conflicts_markdown(output_dir: str, run_dt: datetime,
                             soft_compute: Dict[str, Any],
                             flow_by_tkr: Dict[str, List[Dict[str, Any]]]) -> str:
    name = f"Conflicts_{run_dt.strftime('%m_%d_%y')}.md"
    path = os.path.join(output_dir, name)
    lines = []
    lines.append(f"# Conflicts - {run_dt.strftime('%B %d, %Y')}")
    lines.append("")
    any_conf = False

    for t in sorted(soft_compute.keys()):
        if not soft_compute[t].get("conflict"):
            continue
        any_conf = True
        lines.append(f"**{t}**")
        pos = []
        neg = []
        for it in flow_by_tkr.get(t, []):
            from core import mechanism_impact_sign  # local to avoid cycles
            sign = mechanism_impact_sign(it.get("signal_type",""), it.get("direction",""))
            tag = f"[{it.get('title','Unknown')}, {it.get('as_of','')}]"
            ev = (it.get("evidence") or "").strip()
            if sign > 0:
                pos.append(f"- {it.get('signal_type', '')} {it.get('direction', '')}: {ev} {tag}")
            elif sign < 0:
                neg.append(f"- {it.get('signal_type', '')} {it.get('direction', '')}: {ev} {tag}")
        if pos:
            lines.append("- Positive signals")
            lines.extend([f"  {x}" for x in pos[:6]])
        if neg:
            lines.append("- Negative signals")
            lines.extend([f"  {x}" for x in neg[:6]])
        lines.append("")
    if not any_conf:
        lines.append("No conflicts flagged.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def write_markdown(output_dir: str, run_dt: datetime,
                   exec_items: List[Dict[str, Any]],
                   endmkt_stock: Dict[str, Dict[str, Any]],
                   endmkt_flow: Dict[str, List[str]],
                   ticker_groups: Dict[str, List[str]],
                   stock_paragraphs: Dict[str, str],
                   flow_bullets_by_tkr: Dict[str, List[str]],
                   story_scores_int: Dict[str, int],
                   softdata_scores: Dict[str, int],
                   score_reasons_story: Dict[str, str],      # NEW
                   score_reasons_soft: Dict[str, str],       # NEW
                   tilt_map: Dict[str, Dict[str, Any]],      # NEW
                   universe_names: Dict[str, str],
                   flow_digest: Dict[str, Any],
                   stock_digest: Dict[str, Any]) -> str:
    md_name = f"WeeklyRecap_{run_dt.strftime('%m_%d_%y')}.md"
    md_path = os.path.join(output_dir, md_name)
    lines = []
    lines.append(f"## Weekly Recap - {run_dt.strftime('%B %d, %Y')}")
    lines.append("")
    lines.append("## Executive Summary")
    if exec_items:
        for item in exec_items:
            lines.append(_md_bullet(0, item['main_point']))
            for sp in item.get("supporting_points", []):
                lines.append(_md_bullet(1, sp))
            lines.append(_md_bullet(1, "Incremental Flow Datapoints:"))
            flowb = clean_bullets_list(item.get("flow", []))
            if flowb:
                for b in flowb:
                    lines.append(_md_bullet(2, _md_italicize(b)))
            else:
                lines.append(_md_bullet(2, _md_italicize("No incremental data")))
    else:
        lines.append(_md_bullet(0, "No top items flagged."))

    lines.append("")
    lines.append("## End Market Summaries")

    end_parents = set([lab.split("|")[0].strip() for lab in list(endmkt_stock.keys()) + list(endmkt_flow.keys())])
    tick_parents = set(ticker_groups.keys())
    all_parents = sorted(end_parents.union(tick_parents))
    if "" in all_parents:
        all_parents.remove("")
    all_parents = ["UNGROUPED"] + all_parents

    for parent in all_parents:
        lines.append("")
        lines.append(f"## {parent}")

        if parent != "UNGROUPED":
            pst = endmkt_stock.get(parent, {"main_point": "", "supporting_points": []})
            if pst.get("main_point"):
                lines.append(_md_bullet(0, pst['main_point']))
                for sp in pst.get("supporting_points", []):
                    lines.append(_md_bullet(1, sp))

            pfl = endmkt_flow.get(parent, [])
            lines.append(_md_bullet(0, "Incremental Flow Datapoints:"))
            if pfl:
                for b in clean_bullets_list(pfl):
                    lines.append(_md_bullet(1, _md_italicize(b)))
            else:
                lines.append(_md_bullet(1, _md_italicize("No incremental data")))

            children = sorted(set(
                [lab for lab in endmkt_stock.keys() if " | " in lab and lab.split("|")[0].strip() == parent] +
                [lab for lab in endmkt_flow.keys() if " | " in lab and lab.split("|")[0].strip() == parent]
            ))
            for child in children:
                child_name = child.split("|", 1)[1].strip()
                lines.append(f"### {child_name}")
                cst = endmkt_stock.get(child, {"main_point": "", "supporting_points": []})
                if cst.get("main_point"):
                    lines.append(_md_bullet(0, cst['main_point']))
                    for sp in cst.get("supporting_points", []):
                        lines.append(_md_bullet(1, sp))
                cfl = endmkt_flow.get(child, [])
                lines.append(_md_bullet(0, "Incremental Flow Datapoints:"))
                if cfl:
                    for b in clean_bullets_list(cfl):
                        lines.append(_md_bullet(1, _md_italicize(b)))
                else:
                    lines.append(_md_bullet(1, _md_italicize("No incremental data")))

        group_key = "" if parent == "UNGROUPED" else parent
        tkrs = sorted(ticker_groups.get(group_key, []))
        if tkrs:
            lines.append("### Ticker Summaries")
            for t in tkrs:
                # Ticker header
                lines.append(_md_bullet(0, f"**{t}**"))

                # Stock paragraph
                sp = (stock_paragraphs.get(t, "") or "").strip()
                if not sp:
                    sp = fallback_stock_paragraph(t, universe_names, flow_digest, stock_digest)
                sp = __import__("re").sub(r"\bInversion\b:?", "", sp).strip()
                lines.append(sp)

                # Flow bullets
                lines.append(_md_bullet(1, "Incremental Flow Datapoints:"))
                fb = flow_bullets_by_tkr.get(t, [])
                if fb:
                    for b in clean_bullets_list(fb):
                        lines.append(_md_bullet(2, _md_italicize(b)))
                else:
                    lines.append(_md_bullet(2, _md_italicize("No incremental data")))

                # Simple metrics
                lines.append(f"    StoryScore: {story_scores_int.get(t, 0)} - {(score_reasons_story.get(t) or '').strip()}")
                lines.append(f"    SoftData: {softdata_scores.get(t, 0)} - {(score_reasons_soft.get(t) or '').strip()}")
                lines.append(f"    Bar: {_bar_from_tilt(tilt_map, t)}")
                lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path
