# pipeline.py
import os
import re
import math
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from dateutil import parser as dtparser
from collections import defaultdict

from core import (
    load_config, ensure_dir, load_configs, parse_jsonl_dir, build_views, build_digests,
    openai_client_from_cfg, quantize_with_ranks_debug, rebalance_softdata,
    compute_deltas, latest_prior_scores, select_exec_flags, story_sign,
    clean_bullets_list, parse_date, sanitize_text_for_excel
)

from llm import (
    storyscore_compute_pass, storyscore_write_pass,
    softdata_compute_pass, softdata_write_pass,
    endmarket_stock_pass, endmarket_flow_pass,
    exec_summary_pass
)
from output import write_markdown, write_conflicts_markdown, fallback_stock_paragraph


def run_pipeline(config_path: str):
    # ---- Load config / dirs -------------------------------------------------
    cfg = load_config(config_path)
    run_dt = datetime.today() if not cfg.get("run_date") else dtparser.parse(cfg["run_date"]).replace(tzinfo=None)

    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)
    # NEW: separate destination for Markdown (falls back to output_dir if not set)
    md_output_dir = cfg.get("markdown_output_dir", output_dir)
    ensure_dir(md_output_dir)
    debug_dir = cfg.get("debug_dir") or os.path.join(output_dir, f"debug_{run_dt.date().isoformat()}")
    if cfg.get("debug_enabled"):
        ensure_dir(debug_dir)

    print("[0%] Starting WeeklyRecap run.")
    trace_tickers = set([str(t).upper() for t in (cfg.get("debug_trace_tickers") or [])])

    # ---- Canonical configs & (optional) valuation ---------------------------
    cfgs = load_configs(cfg["configs_dir"])

    valuation_df = None
    valuation_tickers = []
    if os.path.exists(cfg["valuation_file"]):
        valuation_df = pd.read_excel(cfg["valuation_file"])
        tcol = next((c for c in valuation_df.columns if c.strip().lower() in ["ticker", "tickers"]), None)
        if tcol is None:
            raise ValueError("Valuation file must contain a 'Ticker' column")
        valuation_df[tcol] = valuation_df[tcol].astype(str).str.upper()
        valuation_tickers = sorted(set(valuation_df[tcol].tolist()))

    # ---- Universe (wide) ----------------------------------------------------
    from core import build_universe
    universe_names = build_universe(cfgs, valuation_tickers)

    # ---- Hard gate: ticker_master + optional debug limiter ------------------
    try:
        master_path = (
            cfg.get("ticker_master_file")
            or cfg.get("ticker_master")
            or cfg.get("ticker_master_path")
            or ""
        )
        restrict_master = bool(cfg.get("restrict_universe_to_ticker_master", True))
        abort_on_leak = bool(cfg.get("abort_on_universe_leak", False))

        if restrict_master:
            if not master_path:
                msg = (
                    "restrict_universe_to_ticker_master=True but no ticker master path set. "
                    "Configure one of: 'ticker_master_file', 'ticker_master', or 'ticker_master_path' in config.yaml"
                )
                if abort_on_leak:
                    raise RuntimeError(msg)
                else:
                    print(f"[WARN] {msg}")
            elif not os.path.exists(master_path):
                msg = f"Ticker master path not found: {master_path}"
                if abort_on_leak:
                    raise FileNotFoundError(msg)
                else:
                    print(f"[WARN] {msg}")
            else:
                import json
                with open(master_path, "r", encoding="utf-8") as _f:
                    _mj = json.load(_f)
                if isinstance(_mj, dict) and "tickers" in _mj:
                    _mt = [str(x).upper() for x in (_mj.get("tickers") or [])]
                elif isinstance(_mj, list):
                    _mt = [str(x).upper() for x in _mj]
                else:
                    _mt = []
                _ms = set([t for t in _mt if t])
                universe_names = {t: n for t, n in universe_names.items() if t in _ms}

        if bool(cfg.get("restrict_universe_to_debug_trace")):
            trace_tickers = set([str(t).upper() for t in (cfg.get("debug_trace_tickers") or [])])
            if trace_tickers:
                universe_names = {t: n for t, n in universe_names.items() if t in trace_tickers}

    except Exception as _e:
        if bool(cfg.get("abort_on_universe_leak", False)):
            raise
        print(f"[WARN] Universe master restriction skipped: {_e}")

    # ---- Progress ticker ----------------------------------------------------
    num_tickers = len(universe_names)
    batches_sw = max(1, math.ceil(num_tickers / int(cfg.get("tickers_per_call_storyscore_write", 40))))
    total_units = 25 + batches_sw
    done_units = 0

    def _progress(label, inc=1):
        nonlocal done_units
        done_units += inc
        pct = min(99, int((done_units / max(1, total_units)) * 100))
        print(f"[{pct}%] {label}")

    _progress(f"Universe built with {num_tickers} tickers")

    # ---- Expectation tilt (valuation) --------------------------------------
    tilt = {}
    if valuation_df is not None:
        if "Cov_SpotPct_EV" not in valuation_df.columns:
            raise ValueError("Valuation file must contain 'Cov_SpotPct_EV' column")
        tcol = next((c for c in valuation_df.columns if c.strip().lower() in ["ticker", "tickers"]))
        vdf = valuation_df[[tcol, "Cov_SpotPct_EV"]].copy()
        vdf[tcol] = vdf[tcol].astype(str).str.upper()
        for t in universe_names.keys():
            row = vdf[vdf[tcol] == t]
            if row.empty:
                tilt[t] = {"cov_spotpct_ev": None, "tilt": "Neutral"}
            else:
                v = float(row["Cov_SpotPct_EV"].iloc[0])
                tilt[t] = {
                    "cov_spotpct_ev": (None if pd.isna(v) else v),
                    "tilt": "High" if (not pd.isna(v) and v >= cfg.get("tilt_high_pct", 80))
                    else ("Low" if (not pd.isna(v) and v <= cfg.get("tilt_low_pct", 20)) else "Mid"),
                }
    else:
        for t in universe_names.keys():
            tilt[t] = {"cov_spotpct_ev": None, "tilt": "Neutral"}

    # ---- Ingest evidence ----------------------------------------------------
    raw = parse_jsonl_dir(cfg["input_jsonl_dir"])
    _progress("Parsed input JSONL")

    # ---- Build views (windows and routing) ----------------------------------
    flow_days  = int(cfg["flow_window_days"])  + int(cfg.get("flow_window_coast_days", 0))
    stock_days = int(cfg["stock_window_days"]) + int(cfg.get("stock_window_coast_days", 0))
    flow_by_tkr, stock_by_tkr, endmkt_flow_by_label, endmkt_stock_by_label = build_views(
        raw, universe_names, cfgs, run_dt, flow_days, stock_days
    )
    _progress("Built views")

    # ---- Digests (per-ticker subsets to pass to LLM) ------------------------
    stock_digest, flow_digest = build_digests(
        flow_by_tkr, stock_by_tkr,
        cfg["max_items_per_ticker_stock_digest"],
        cfg["max_items_per_ticker_flow_digest"],
        cfg, cfgs, run_dt
    )
    _progress("Built digests")

    # ---- OpenAI client ------------------------------------------------------
    client = openai_client_from_cfg(cfg)
    _progress("OpenAI client ready")

    # ---- StoryScore passes --------------------------------------------------
    story_compute, _ = storyscore_compute_pass(client, cfg, stock_digest, tilt, universe_names, run_dt, debug_dir)
    _progress("StoryScore compute pass complete")
    stock_paragraphs = storyscore_write_pass(client, cfg, stock_digest, story_compute, universe_names, run_dt, debug_dir)
    _progress("StoryScore write pass complete", inc=batches_sw)

    # ---- SoftData passes ----------------------------------------------------
    try:
        soft_compute, _ = softdata_compute_pass(client, cfg, flow_digest, tilt, universe_names, run_dt, debug_dir)
        _progress("SoftData compute pass complete")
    except Exception as _e:
        print(f"[WARN] SoftData compute pass failed globally; using deterministic fallback: {_e}")
        soft_compute = {t: {"softdata": 0, "flow_bullets": [], "dominant_mechanisms": [], "confidence": "low", "conflict": False, "score_reason": ""} for t in universe_names}

    try:
        flow_bullets_by_tkr = softdata_write_pass(client, cfg, soft_compute, universe_names, run_dt, debug_dir, flow_digest)
        _progress("SoftData bullets written")
    except Exception as _e:
        print(f"[WARN] SoftData write pass failed; using deterministic fallback bullets: {_e}")
        flow_bullets_by_tkr = {t: [] for t in universe_names}
        _progress("SoftData bullets written")


    # ---- End-market passes --------------------------------------------------

    def _label_to_str(lab) -> str:
        if isinstance(lab, tuple) and len(lab) == 2:
            p, c = lab
            return p if not c else f"{p} | {c}"
        return str(lab)

    def _lean_simple(items: List[dict], max_n: int, chars_key: str) -> List[dict]:
        lim_chars = int(cfg.get(chars_key, 320))
        arr = sorted((items or []), key=lambda x: parse_date(x.get("as_of") or ""), reverse=True)[:max_n]
        out = []
        for it in arr:
            ev = (it.get("evidence") or "")
            ev = re.sub(r"\s*\[\s*Disclosure.*$", "", ev, flags=re.I | re.S).strip()
            if lim_chars > 0 and len(ev) > lim_chars:
                ev = ev[:lim_chars]
            ttl = str(it.get("title") or "")
            ttl_short = " ".join(ttl.split()[:5]) + ("..." if len(ttl.split()) > 5 else "")
            out.append({
                "title_short": ttl_short,
                "as_of": it.get("as_of"),
                "signal_type": it.get("signal_type"),
                "direction": it.get("direction"),
                "evidence": ev
            })
        return out

    def _deterministic_endmarket_stock(em_items: dict) -> dict:
        out = {}
        for lab, items in em_items.items():
            lean = _lean_simple(items, max_n=12, chars_key="evidence_chars_per_item_endmarket")
            mp = (lean[0]["evidence"] if lean else "").strip()
            if len(mp) > 160:
                mp = mp[:160].rstrip() + "..."
            sps = []
            for it in lean[1:4]:
                txt = (it.get("evidence") or "").strip()
                sps.append(txt[:140] + ("..." if len(txt) > 140 else ""))
            out[_label_to_str(lab)] = {"main_point": mp, "supporting_points": sps}
        return out

    def _deterministic_endmarket_flow(em_items: dict) -> dict:
        out = {}
        for lab, items in em_items.items():
            lean = _lean_simple(items, max_n=6, chars_key="evidence_chars_per_item_endmarket")
            bullets = []
            for it in lean[:3]:
                mech = (it.get("signal_type") or "Signal").split()[0].capitalize()
                dirn = (it.get("direction") or "").lower()
                txt = (it.get("evidence") or "").strip()
                tag = f"[{it.get('title_short','')}, {it.get('as_of','')}]"
                bullets.append(f"{mech} {dirn}: {txt} {tag}")
            out[_label_to_str(lab)] = clean_bullets_list(bullets)
        return out

    try:
        if bool(cfg.get("use_llm_endmarket", True)):
            endmkt_stock_struct = endmarket_stock_pass(client, cfg, run_dt, endmkt_stock_by_label, debug_dir)
            _progress("End market stock summaries done")
            endmkt_flow_bullets = endmarket_flow_pass(client, cfg, run_dt, endmkt_flow_by_label, debug_dir)
            _progress("End market flow bullets done")
        else:
            endmkt_stock_struct = _deterministic_endmarket_stock(endmkt_stock_by_label)
            _progress("End market stock summaries done")
            endmkt_flow_bullets = _deterministic_endmarket_flow(endmkt_flow_by_label)
            _progress("End market flow bullets done")
    except Exception as _e:
        # Hard fallback if LLM end-market passes raise
        endmkt_stock_struct = _deterministic_endmarket_stock(endmkt_stock_by_label)
        print(f"[WARN] End-market stock pass failed globally; using deterministic fallback: {_e}")
        endmkt_flow_bullets = _deterministic_endmarket_flow(endmkt_flow_by_label)
        print(f"[WARN] End-market flow pass failed globally; using deterministic fallback.")

    # ---- Prepare outputs ----------------------------------------------------
    rows = []
    deadband = cfg["story_sign_deadband"]

    def primary_label_for_ticker(t: str) -> str:
        exp = cfgs["company_endmarket_exposure"]
        for parent, mp in exp.items():
            if t in mp:
                child_weights = list(mp[t].items())
                child_weights.sort(key=lambda kv: kv[1], reverse=True)
                if child_weights:
                    return child_weights[0][0]
        return ""

    ticker_groups = defaultdict(list)
    raw_story = {}
    for t in sorted(universe_names.keys()):
        parent_label = primary_label_for_ticker(t)
        parent = parent_label.split("|")[0].strip() if parent_label else ""
        ticker_groups[parent].append(t)
        raw_story[t] = float(story_compute.get(t, {}).get("story_score", 0.0))       

    story_scores_int, story_ranks_table = quantize_with_ranks_debug(raw_story)
    softdata_rebalanced = rebalance_softdata(soft_compute, flow_digest, cfg, target_pos_share=0.33, target_neg_share=0.33)

    for t in sorted(universe_names.keys()):
        sc_int = int(story_scores_int.get(t, 0))
        sd = int(softdata_rebalanced.get(t, 0))
        inv_sign = story_sign(raw_story.get(t, 0.0), deadband)

        divergence = 0
        if sd != 0:
            if sd == 1 and inv_sign == -1:
                divergence = +1
            elif sd == -1 and inv_sign == +1:
                divergence = -1

        evid = flow_by_tkr.get(t, [])
        from core import mechanism_impact_sign  # local import to avoid cycles
        rows.append({
            "Ticker": t,
            "StoryScore": sc_int,
            "SoftData": sd,
            "Divergence": divergence,
            "Cov_SpotPct_EV": tilt.get(t, {}).get("cov_spotpct_ev"),
            "Expectation_Tilt": tilt.get(t, {}).get("tilt", "Neutral"),
            "Confidence_Story": story_compute.get(t, {}).get("confidence"),
            "DominantMech_Story": "; ".join(story_compute.get(t, {}).get("dominant_mechanisms", [])),
            "Drivers_Story": "; ".join(story_compute.get(t, {}).get("drivers", [])),
            "Confidence_Flow": soft_compute.get(t, {}).get("confidence"),
            "DominantMech_Flow": "; ".join(soft_compute.get(t, {}).get("dominant_mechanisms", [])),
            "Flow_Conflict": bool(soft_compute.get(t, {}).get("conflict", False)),
            "Evidence_Count": len(evid),
            "Pos_Count": sum(1 for it in evid if mechanism_impact_sign(it.get("signal_type",""), it.get("direction","")) > 0),
            "Neg_Count": sum(1 for it in evid if mechanism_impact_sign(it.get("signal_type",""), it.get("direction","")) < 0),
            "Mixed_Count": sum(1 for it in evid if mechanism_impact_sign(it.get("signal_type",""), it.get("direction","")) == 0),
            "Unique_Dates_Stock": stock_digest.get(t, {}).get("signal_quality", {}).get("unique_dates", 0),
            "Single_Source_Dominance_Stock": bool(stock_digest.get(t, {}).get("signal_quality", {}).get("single_source_dominance", False)),
            # NEW FIELDS:
            "StoryScore_Why": (story_compute.get(t, {}) or {}).get("score_reason", ""),
            "SoftData_Why":  (soft_compute.get(t, {})  or {}).get("score_reason", "")
        })


    df_now = pd.DataFrame(rows)
    prior = latest_prior_scores(cfg["output_dir"])

    col_order = [
        "Ticker","StoryScore","StoryScore_Why","SoftData","SoftData_Why","Divergence",
        "Cov_SpotPct_EV","Expectation_Tilt",
        "Evidence_Count","Pos_Count","Neg_Count","Mixed_Count",
        "Unique_Dates_Stock","Single_Source_Dominance_Stock",
        "Confidence_Story","DominantMech_Story","Drivers_Story",
        "Confidence_Flow","DominantMech_Flow","Flow_Conflict"
    ]
    df_now = df_now[col_order].copy()
    df_now = compute_deltas(df_now, prior)

    selected = select_exec_flags(df_now)

    primary_parent = {}
    for t in sorted(universe_names.keys()):
        pl = primary_label_for_ticker(t)
        primary_parent[t] = pl.split("|")[0].strip() if pl else ""

    # Executive summary with outer guard to make crashes impossible
    def _deterministic_exec(selected_tickers: list) -> list:
        max_chars = int(cfg.get("exec_summary_main_point_chars", 260))

        def _trim_to_limit(s: str, n: int) -> str:
            if not s:
                return ""
            s = s.replace("\u2013", "-").replace("\u2014", "-").strip()
            if len(s) <= n:
                return s
            cut = s[:n]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            return cut.rstrip(",;: ") + "..."

        out_items = []
        for t in selected_tickers[:8]:
            para = (stock_paragraphs.get(t) or "").strip()
            items_full = (flow_digest.get(t, {}) or {}).get("items", []) or []
            lean = sorted(items_full, key=lambda x: parse_date(x.get("as_of") or ""), reverse=True)[:6]

            flow_bullets = []
            for it in lean[:3]:
                mech = (it.get("signal_type") or "Signal").split()[0].capitalize()
                dirn = (it.get("direction") or "").lower()
                import re as _re
                txt = _re.sub(r"\s*\[\s*Disclosure.*$", "", (it.get("evidence") or ""), flags=_re.I | _re.S).strip()
                ttl = " ".join(str(it.get("title") or "").split()[:5]) + ("..." if len(str(it.get("title") or "").split()) > 5 else "")
                tag = f"[{ttl}, {it.get('as_of','')}]"
                flow_bullets.append(f"{mech} {dirn}: {_trim_to_limit(txt, 220)} {tag}")

            pref = f"{t} ({primary_parent.get(t, '')}): " if primary_parent.get(t, "") else f"{t}: "
            out_items.append({
                    "main_point": (pref + (_trim_to_limit(para, max_chars) if para else "No paragraph")).strip(),
                    "supporting_points": [],
                    "flow": clean_bullets_list(flow_bullets)
                })
        return out_items

    try:
        exec_items = exec_summary_pass(client, cfg, run_dt, selected, df_now, stock_paragraphs, flow_digest, primary_parent)
    except Exception as _e:
        print(f"[WARN] Executive summary LLM failed; using deterministic fallback: {_e}")
        exec_items = _deterministic_exec(selected)

    _progress("Executive summary built")

    # ---- Write outputs ------------------------------------------------------
    story_why = {t: (story_compute.get(t, {}) or {}).get("score_reason", "") for t in universe_names}
    soft_why  = {t: (soft_compute.get(t, {}) or {}).get("score_reason", "") for t in universe_names}
    
    md_path = write_markdown(
        md_output_dir, run_dt, exec_items, endmkt_stock_struct, endmkt_flow_bullets,
        ticker_groups, stock_paragraphs, flow_bullets_by_tkr,
        story_scores_int, softdata_rebalanced,
        story_why, soft_why, tilt,
        universe_names, flow_digest, stock_digest
    )

    _progress(f"Markdown written: {md_path}")

    if cfg.get("write_conflicts_page"):
        conf_path = write_conflicts_markdown(md_output_dir, run_dt, soft_compute, flow_by_tkr)
        _progress(f"Conflicts written: {conf_path}")

    xlsx_name = f"WeeklyRecap_Scores_{run_dt.strftime('%m_%d_%y')}.xlsx"
    xlsx_path = os.path.join(output_dir, xlsx_name)
    # Scrub illegal control characters from all object columns for Excel safety
    obj_cols = df_now.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df_now[col] = df_now[col].map(sanitize_text_for_excel)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xlw:
        df_now.to_excel(xlw, index=False, sheet_name="scores")
    _progress(f"Excel written: {xlsx_path}")

    # ---- Optional per-ticker debug bundles ---------------------------------
    if cfg.get("debug_enabled") and cfg.get("debug_save_per_ticker_bundles") and trace_tickers:
        from core import make_pos_neg_neutral
        for t in sorted(universe_names.keys()):
            if t not in trace_tickers:
                continue
            final_para = (stock_paragraphs.get(t) or "").strip() or fallback_stock_paragraph(t, universe_names, flow_digest, stock_digest)
            final_bullets = flow_bullets_by_tkr.get(t, []) or ["No incremental data"]
            tri = make_pos_neg_neutral(flow_digest.get(t, {}).get("items", []))

            tdir = os.path.join(debug_dir, "tickers")
            os.makedirs(tdir, exist_ok=True)
            with open(os.path.join(tdir, f"{t}.md"), "w", encoding="utf-8") as f:
                f.write(f"# {t} - {universe_names.get(t, t)}\n\n")
                f.write("## Stock Paragraph (final)\n" + final_para + "\n\n")
                f.write("## Flow Bullets (final)\n" + "\n".join(f"- {b}" for b in final_bullets) + "\n\n")
            print(f"[TRACE] Saved debug bundle for {t}: {os.path.join(tdir, f'{t}.md')}")
            print(f"[TRACE] {t} flow triage +{len(tri['positive'])}/-{len(tri['negative'])}/0{len(tri['neutral'])}")

    print("[100%] Completed.")
