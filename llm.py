# llm.py — lean, provenance-safe LLM passes
# Replaces prior budget-heavy pack and brittle post-LLM validation.

from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from core import (
    responses_create_json_schema, fit_payload_to_budget, parse_date,
    clean_bullets_list, STYLE_BULLET, STYLE_PARAGRAPH, save_json, ensure_dir
)


from prompts import (
    STORYSCORE_COMPUTE_PROMPT, STORYSCORE_WRITE_PROMPT,
    SOFTDATA_COMPUTE_PROMPT, SOFTDATA_WRITE_PROMPT,
    ENDMARKET_STOCK_PROMPT, ENDMARKET_FLOW_PROMPT, EXEC_SUMMARY_PROMPT,
    SCHEMA_STORYSCORE_COMPUTE, SCHEMA_STORYSCORE_WRITE,
    SCHEMA_SOFTDATA_COMPUTE, SCHEMA_SOFTDATA_WRITE,
    SCHEMA_ENDMARKET_STOCK, SCHEMA_ENDMARKET_FLOW, SCHEMA_EXEC_SUMMARY,
)

# ------------------------- Helpers -------------------------

def _short_title(title: str, n_words: int = 5) -> str:
    """Return the first n words of title + '...'."""
    if not title:
        return ""
    words = re.split(r'\s+', str(title).strip())
    return " ".join(words[:n_words]) + ("..." if len(words) > n_words else "")

def _strip_disclosures(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s*\[\s*Disclosure.*$", "", text, flags=re.I | re.S).strip()

def _lean_pack(items: List[Dict[str, Any]], limit: int, cfg: Dict[str, Any],
               chars_key: str = "evidence_chars_per_item") -> List[Dict[str, Any]]:
    """Minimal, LLM-relevant pack with a configurable evidence char limit."""
    max_chars = int(cfg.get(chars_key, cfg.get("evidence_chars_per_item", 600)))
    out = []
    for it in (items or [])[:max(0, limit)]:
        ev = _strip_disclosures(it.get("evidence") or "")
        if max_chars > 0 and len(ev) > max_chars:
            ev = ev[:max_chars]
        out.append({
            "title_short": _short_title(it.get("title") or ""),
            "as_of": it.get("as_of"),
            "signal_type": it.get("signal_type"),
            "direction": it.get("direction"),
            "evidence": ev,
        })
    return out

def _fit_or_reduce_single(payload_builder, items: List[Dict[str, Any]], cfg: Dict[str, Any],
                          max_tokens_key: str, step: float = 0.85, min_items: int = 10,
                          limit_key: str = None, default_limit: int = 200) -> Tuple[List[Dict[str, Any]], int]:
    """Reduce per-ticker item limit until payload fits token budget."""
    limit = int(cfg.get(limit_key, default_limit)) if limit_key else int(default_limit)
    max_in = int(cfg.get(max_tokens_key, 200000))
    approx = int(cfg.get("approx_chars_per_token", 4))

    def build(lim: int) -> List[Dict[str, Any]]:
        return payload_builder(lim)

    cur = build(limit)
    while not fit_payload_to_budget(cur, max_in, approx) and limit > min_items:
        new_limit = max(min_items, int(limit * step))
        limit = new_limit
        cur = build(limit)
    return cur, limit

def _save_debug(debug_dir: str, name: str, obj: Any):
    try:
        path = f"{debug_dir}/{name}"
        save_json(path, obj)
    except Exception:
        pass

def _ensure_title_asof_tags(flow_bullets: List[str],
                            lean_items: List[Dict[str, Any]]) -> List[str]:
    """
    Normalize provenance tags so EVERY ticker flow bullet ends with either:
      [title_short, YYYY-MM-DD] if we can map the bullet to a specific item in lean_items
      [UNVERIFIED] otherwise

    Rules:
      1) If a bullet already has a [...] tag WITH a date (YYYY-MM-DD):
         - If that date exists in lean_items, canonicalize to [title_short, YYYY-MM-DD]
         - Else, replace the tag with [UNVERIFIED]
      2) If a bullet has a [...] tag WITHOUT a date:
         - If the bracket text matches a known title_short in lean_items, attach its freshest date and canonicalize
         - Else, replace the tag with [UNVERIFIED]
      3) If a bullet has NO [...] tag, append [UNVERIFIED]

    Notes:
      - We never attach a "freshest" tag unless the bullet's tag maps to lean_items
      - We never drop a bullet
    """
    # Build lookup tables from provided lean_items
    items_sorted = sorted((lean_items or []),
                          key=lambda x: parse_date(x.get("as_of") or ""),
                          reverse=True)

    by_date: Dict[str, str] = {}   # as_of -> title_short
    by_title: Dict[str, str] = {}  # title_short -> freshest as_of

    for it in items_sorted:
        dt = (it.get("as_of") or "").strip()
        ts = (it.get("title_short") or "").strip()
        if not dt or not ts:
            continue
        if dt not in by_date:
            by_date[dt] = ts
        # record the freshest date for each title_short (first hit due to sort desc)
        if ts not in by_title:
            by_title[ts] = dt

    UNVERIFIED_TAG = "[UNVERIFIED]"
    # Determine deterministic fallback: freshest item from lean_items
    freshest_tag = ""
    if items_sorted:
        ts0 = (items_sorted[0].get("title_short") or "").strip()
        dt0 = (items_sorted[0].get("as_of") or "").strip()
        if ts0 and dt0:
            freshest_tag = f"[{ts0}, {dt0}]"

    out: List[str] = []
    for b in (flow_bullets or []):
        if not isinstance(b, str):
            continue
        s = b.strip()

        # Find a trailing bracket tag (if present)
        m = re.search(r"\[(.*?)\]\s*$", s)
        if m:
            inside = (m.group(1) or "").strip()
            new_tag = None

            # Respect an existing UNVERIFIED tag
            if inside.upper() == "UNVERIFIED":
                new_tag = UNVERIFIED_TAG
            else:
                # If the tag contains a date, map date -> title_short
                dm = re.search(r"(\d{4}-\d{2}-\d{2})", inside)
                if dm:
                    dt = dm.group(1)
                    ts = by_date.get(dt)
                    new_tag = f"[{ts}, {dt}]" if ts else UNVERIFIED_TAG
                else:
                    # No date: try exact title_short match
                    ts_guess = inside
                    dt_match = by_title.get(ts_guess)
                    new_tag = f"[{ts_guess}, {dt_match}]" if dt_match else UNVERIFIED_TAG

            s = s[:m.start()].rstrip()
            s = f"{s} {new_tag}"
        else:
            # No tag present -> attach freshest deterministic tag if available
            s = f"{s} {freshest_tag or UNVERIFIED_TAG}"

        out.append(s)

    return out

def _label_to_str(lab) -> str:
    if isinstance(lab, tuple) and len(lab) == 2:
        p, c = lab
        return p if not c else f"{p} | {c}"
    return str(lab)

# ------------------------- StoryScore passes -------------------------

def storyscore_compute_pass(client, cfg: Dict[str, Any],
                            stock_digest: Dict[str, Any],
                            tilt_map: Dict[str, Dict[str, Any]],
                            universe_names: Dict[str, str],
                            run_dt: datetime,
                            debug_dir: str) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Process ONE ticker at a time with a lean pack, but do it concurrently across tickers.
    """
    model = cfg["model"]
    out: Dict[str, Dict[str, Any]] = {}
    last_resp_id = ""
    tickers = sorted(universe_names.keys())
    max_workers = int(cfg.get("llm_concurrency", 6))

    def _do_one(t: str) -> Tuple[str, Dict[str, Any], str]:
        d = stock_digest.get(t, {"items": []})
        items_full = d.get("items", []) or []

        def _builder(lim: int):
            return [{
                "ticker": t,
                "company": universe_names.get(t, t),
                "expectation_tilt": tilt_map.get(t, {}).get("tilt", "Neutral"),
                "exposure_weights": stock_digest.get(t, {}).get("exposure_weights", {}),
                "signal_quality": stock_digest.get(t, {}).get("signal_quality", {}),
                "stock_items": _lean_pack(items_full, lim, cfg)
            }]

        payload_list, used_limit = _fit_or_reduce_single(
            _builder, items_full, cfg,
            "max_input_tokens_storyscore", step=0.9, min_items=8,
            limit_key="max_items_per_ticker_stock_digest", default_limit=60
        )
        payload = payload_list  # schema packs under "items"
        user = "Current date: " + run_dt.date().isoformat() + "\n" + \
               "For EACH ticker in the JSON below, compute a relative StoryScore.\n" + \
               json.dumps({"items": payload}, ensure_ascii=True)

        _save_debug(debug_dir, f"storyscore_payload_{t}.json", {"items": payload, "limit_used": used_limit})

        instructions = STORYSCORE_COMPUTE_PROMPT.strip() + "\n" + STYLE_BULLET
        resp, resp_id = responses_create_json_schema(
            client,
            model=model,
            instructions=instructions,
            user_input=user,
            json_schema=SCHEMA_STORYSCORE_COMPUTE,
            max_output_tokens=int(cfg.get("max_output_tokens_storyscore", 6000)),
            reasoning_effort=cfg.get("reasoning_effort", "medium"),
            text_verbosity=cfg.get("text_verbosity", "low"),
        )
        obj = resp if isinstance(resp, dict) else {}
        items = obj.get("items", [])
        rec: Dict[str, Any] = {}
        for it in (items or []):
            if (it.get("ticker") or "").upper() == t:
                rec = {
                    "story_score": it.get("story_score", 0.0),
                    "drivers": it.get("drivers", []),
                    "dominant_mechanisms": it.get("dominant_mechanisms", []),
                    "inversion_flag": it.get("inversion_flag", "none"),
                    "confidence": it.get("confidence", "low"),
                    "throughline": it.get("throughline", ""),
                    "why_now": it.get("why_now", ""),
                    "watch_next": it.get("watch_next", ""),
                    "score_reason": it.get("score_reason", "")
                }
                break
        _save_debug(debug_dir, f"storyscore_response_{t}.json", obj)
        return t, rec, (resp_id or "")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, rec, rid = fut.result()
            if rec:
                out[t] = rec
            if rid:
                last_resp_id = rid

    return out, last_resp_id


def storyscore_write_pass(client, cfg: Dict[str, Any],
                          stock_digest: Dict[str, Any],
                          story_compute: Dict[str, Dict[str, Any]],
                          universe_names: Dict[str, str],
                          run_dt: datetime,
                          debug_dir: str) -> Dict[str, str]:
    """Write 1 paragraph per ticker. Run tickers concurrently."""
    model = cfg["model"]
    out: Dict[str, str] = {}
    tickers = sorted(universe_names.keys())
    max_workers = int(cfg.get("llm_concurrency", 6))

    def _do_one(t: str) -> Tuple[str, str]:
        d = stock_digest.get(t, {"items": []})
        items_full = d.get("items", []) or []

        def _builder(lim: int):
            return [{
                "ticker": t,
                "company": universe_names.get(t, t),
                "story_score": float(story_compute.get(t, {}).get("story_score", 0.0)),
                "drivers": story_compute.get(t, {}).get("drivers", []),
                "throughline": story_compute.get(t, {}).get("throughline", ""),
                "why_now": story_compute.get(t, {}).get("why_now", ""),
                "watch_next": story_compute.get(t, {}).get("watch_next", ""),
                "exposure_weights": stock_digest.get(t, {}).get("exposure_weights", {}),
                "stock_items": _lean_pack(items_full, lim, cfg)
            }]

        payload_list, used_limit = _fit_or_reduce_single(
            _builder, items_full, cfg,
            "max_input_tokens_storyscore_write", step=0.9, min_items=8,
            limit_key="max_items_per_ticker_stock_digest", default_limit=60
        )
        payload = payload_list

        instructions = STORYSCORE_WRITE_PROMPT.strip() + "\n" + STYLE_PARAGRAPH
        user = "Current date: " + run_dt.date().isoformat() + "\n" + json.dumps({"items": payload}, ensure_ascii=True)
        resp, _ = responses_create_json_schema(
            client,
            model=model,
            instructions=instructions,
            user_input=user,
            json_schema=SCHEMA_STORYSCORE_WRITE,
            max_output_tokens=int(cfg.get("max_output_tokens_storyscore_write", 5000)),
            reasoning_effort=cfg.get("reasoning_effort", "low"),
            text_verbosity=cfg.get("text_verbosity", "low"),
        )
        obj = resp if isinstance(resp, dict) else {}
        para = ""
        for it in (obj.get("items") or []):
            if (it.get("ticker") or "").upper() == t:
                para = (it.get("stock_paragraph") or "").strip()
                break
        _save_debug(debug_dir, f"storyscore_write_response_{t}.json", obj)
        return t, para

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, para = fut.result()
            out[t] = para

    return out

# ------------------------- SoftData passes -------------------------

def softdata_compute_pass(client, cfg: Dict[str, Any],
                          flow_digest: Dict[str, Any],
                          tilt_map: Dict[str, Dict[str, Any]],
                          universe_names: Dict[str, str], run_dt: datetime,
                          debug_dir: str) -> Tuple[Dict[str, Any], str]:
    """
    Compute SoftData and pick incremental datapoints per ticker with full coverage,
    but run tickers concurrently. Fail-open on any LLM error.
    """
    model = cfg["model"]
    out: Dict[str, Any] = {}
    last_resp_id = ""
    tickers = sorted(universe_names.keys())
    max_workers = int(cfg.get("llm_concurrency", 6))

    def _do_one(t: str) -> Tuple[str, Dict[str, Any], str]:
        d = flow_digest.get(t, {"items": []})
        items_full = d.get("items", []) or []

        def _builder(lim: int):
            return [{
                "ticker": t,
                "company": universe_names.get(t, t),
                "expectation_tilt": tilt_map.get(t, {}).get("tilt", "Neutral"),
                "exposure_weights": flow_digest.get(t, {}).get("exposure_weights", {}),
                "flow_items": _lean_pack(items_full, lim, cfg)
            }]

        payload_list, used_limit = _fit_or_reduce_single(
            _builder, items_full, cfg,
            "max_input_tokens_softdata", step=0.9, min_items=int(cfg.get("min_items_per_ticker_flow_digest", 60)),
            limit_key="max_items_per_ticker_flow_digest", default_limit=120
        )
        payload = payload_list

        user = "Current date: " + run_dt.date().isoformat() + "\n" + \
               json.dumps({"items": payload}, ensure_ascii=True)

        _save_debug(debug_dir, f"softdata_payload_{t}.json", {"items": payload, "limit_used": used_limit})

        instructions = SOFTDATA_COMPUTE_PROMPT.strip() + "\n" + STYLE_BULLET

        try:
            resp, resp_id = responses_create_json_schema(
                client,
                model=model,
                instructions=instructions,
                user_input=user,
                json_schema=SCHEMA_SOFTDATA_COMPUTE,
                max_output_tokens=int(cfg.get("max_output_tokens_softdata", 6000)),
                reasoning_effort=cfg.get("reasoning_effort", "medium"),
                text_verbosity=cfg.get("text_verbosity", "low"),
            )
        except Exception as e:
            # Fail-open: keep run alive with deterministic neutral record
            _save_debug(debug_dir, f"softdata_error_{t}.json", {"error": str(e)})
            rec_out = {
                "softdata": 0,
                "flow_bullets": [],
                "dominant_mechanisms": [],
                "confidence": "low",
                "conflict": False,
                "score_reason": "",
                "_lean_items": _lean_pack(items_full, used_limit, cfg)
            }
            return t, rec_out, ""

        obj = resp if isinstance(resp, dict) else {}
        _save_debug(debug_dir, f"softdata_response_{t}.json", obj)

        it = None
        for rec in (obj.get("items") or []):
            if (rec.get("ticker") or "").upper() == t:
                it = rec
                break
        if not it:
            return t, {
                "softdata": 0,
                "flow_bullets": [],
                "dominant_mechanisms": [],
                "confidence": "low",
                "conflict": False,
                "score_reason": "",
                "_lean_items": _lean_pack(items_full, used_limit, cfg)
            }, (resp_id or "")

        rec_out = {
            "softdata": int(it.get("softdata", 0)),
            "flow_bullets": [x for x in (it.get("flow_bullets") or []) if isinstance(x, str)],
            "dominant_mechanisms": it.get("dominant_mechanisms", []),
            "confidence": it.get("confidence", "low"),
            "conflict": bool(it.get("conflict", False)),
            "score_reason": it.get("score_reason", ""),
            "_lean_items": _lean_pack(items_full, used_limit, cfg)
        }

        return t, rec_out, (resp_id or "")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, rec, rid = fut.result()
            out[t] = rec
            if rid:
                last_resp_id = rid

    return out, last_resp_id


    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, rec, rid = fut.result()
            out[t] = rec
            if rid:
                last_resp_id = rid

    return out, last_resp_id


def softdata_write_pass(client, cfg: Dict[str, Any],
                        soft_compute: Dict[str, Any],
                        universe_names: Dict[str, str], run_dt: datetime,
                        debug_dir: str,
                        flow_digest: Dict[str, Any]) -> Dict[str, List[str]]:
    """Normalize tags to [TitleShort, YYYY-MM-DD] and never drop a bullet. Run concurrently."""
    model = cfg["model"]
    out: Dict[str, List[str]] = {}
    tickers = sorted(universe_names.keys())
    max_workers = int(cfg.get("llm_concurrency", 6))

    def _do_one(t: str) -> Tuple[str, List[str]]:
        rec = soft_compute.get(t, {}) or {}
        bullets_raw = [x for x in rec.get("flow_bullets") or [] if isinstance(x, str)]
        lean_items = rec.get("_lean_items") or _lean_pack((flow_digest.get(t, {}) or {}).get("items", []), int(cfg.get("max_items_per_ticker_flow_digest", 120)), cfg)
        bullets = _ensure_title_asof_tags(bullets_raw, lean_items)

        final_bullets = bullets
        if bool(cfg.get("softdata_write_improve_wording", False)) and bullets:
            payload = [{"ticker": t, "flow": bullets}]
            user = "Current date: " + run_dt.date().isoformat() + "\n" + \
                   "Polish wording and merge near-duplicates. Keep EXACT provenance tags unchanged.\n" + \
                   json.dumps({"items": payload}, ensure_ascii=True)
            instructions = SOFTDATA_WRITE_PROMPT.strip() + "\n" + STYLE_BULLET
            try:
                resp, _ = responses_create_json_schema(
                    client,
                    model=model,
                    instructions=instructions,
                    user_input=user,
                    json_schema=SCHEMA_SOFTDATA_WRITE,
                    max_output_tokens=int(cfg.get("max_output_tokens_softdata_write", 3000)),
                    reasoning_effort=cfg.get("reasoning_effort", "medium"),
                    text_verbosity=cfg.get("text_verbosity", "low"),
                )
                obj = resp if isinstance(resp, dict) else {}
                _save_debug(debug_dir, f"softdata_write_response_{t}.json", obj)
                items = obj.get("items") or []
                if items and isinstance(items[0], dict):
                    final_bullets = [x for x in (items[0].get("flow") or []) if isinstance(x, str)]
            except Exception as e:
                _save_debug(debug_dir, f"softdata_write_error_{t}.json", {"error": str(e)})
                final_bullets = bullets  # keep originals on failure

        return t, clean_bullets_list(final_bullets)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, bullets = fut.result()
            out[t] = bullets

    return out

# ------------------------- End-market passes -------------------------

def endmarket_stock_pass(client, cfg: Dict[str, Any], run_dt: datetime,
                         endmarket_stock_items: Dict[Any, List[Dict[str, Any]]],
                         debug_dir: str) -> Dict[str, Dict[str, Any]]:
    """Summarize stock setup per end market label using lean pack. Run concurrently. Do not require label echo."""
    model = cfg["model"]
    out: Dict[str, Dict[str, Any]] = {}
    labels = list(endmarket_stock_items.keys())
    max_workers = int(cfg.get("llm_concurrency", 6))

    def _do_one(lab) -> Tuple[str, Dict[str, Any]]:
        lab_str = _label_to_str(lab)
        items = (endmarket_stock_items.get(lab) or [])
        if bool(cfg.get("endmarket_sort_by_date_desc", True)):
            items = sorted(items, key=lambda x: parse_date(x.get("as_of") or ""), reverse=True)


        def _builder(lim: int):
            # Provide the label only as context; model does not need to echo it.
            return [{"label_context": lab_str, "items": _lean_pack(items, lim, cfg, chars_key="evidence_chars_per_item_endmarket")}]

        payload_list, used_limit = _fit_or_reduce_single(
            _builder, items, cfg,
            "max_input_tokens_endmarket_stock", step=0.85, min_items=8,
            limit_key="max_items_per_endmarket_stock_digest", default_limit=80
        )
        user = "Current date: " + run_dt.date().isoformat() + "\n" + json.dumps({"items": payload_list}, ensure_ascii=True)
        instructions = ENDMARKET_STOCK_PROMPT.strip() + "\n" + STYLE_PARAGRAPH

        try:
            resp, _ = responses_create_json_schema(
                client,
                model=model,
                instructions=instructions,
                user_input=user,
                json_schema=SCHEMA_ENDMARKET_STOCK,
                max_output_tokens=int(cfg.get("max_output_tokens_endmarket_stock", 3000)),
                reasoning_effort=cfg.get("reasoning_effort", "medium"),
                text_verbosity=cfg.get("text_verbosity", "low"),
            )
            obj = resp if isinstance(resp, dict) else {}
            items_obj = (obj.get("items") or [])
            rec: Dict[str, Any] = {"main_point": "", "supporting_points": []}
            if items_obj and isinstance(items_obj[0], dict):
                it0 = items_obj[0]
                rec = {
                    "main_point": (it0.get("main_point") or "").strip(),
                    "supporting_points": [x for x in (it0.get("supporting_points") or []) if isinstance(x, str)]
                }
            # Fallback if empty: use deterministic snippets
            if not rec.get("main_point"):
                lean = _lean_pack(items, min(12, used_limit), cfg, chars_key="evidence_chars_per_item_endmarket")
                mp = (lean[0]["evidence"] if lean else "").strip()
                if len(mp) > 160:
                    mp = mp[:160].rstrip() + "..."
                sps = []
                for it in lean[1:4]:
                    txt = (it.get("evidence") or "").strip()
                    sps.append(txt[:140] + ("..." if len(txt) > 140 else ""))
                rec = {"main_point": mp, "supporting_points": sps}
            return lab_str, rec
        except Exception:
            # Deterministic fallback
            lean = _lean_pack(items, min(12, used_limit), cfg, chars_key="evidence_chars_per_item_endmarket")
            mp = (lean[0]["evidence"] if lean else "").strip()
            if len(mp) > 160:
                mp = mp[:160].rstrip() + "..."
            sps = []
            for it in lean[1:4]:
                txt = (it.get("evidence") or "").strip()
                sps.append(txt[:140] + ("..." if len(txt) > 140 else ""))
            return lab_str, {"main_point": mp, "supporting_points": sps}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, lab) for lab in labels]
        for fut in as_completed(futures):
            lab_str, rec = fut.result()
            out[lab_str] = rec

    return out


def endmarket_flow_pass(client, cfg: Dict[str, Any], run_dt: datetime,
                        endmarket_flow_items: Dict[Any, List[Dict[str, Any]]],
                        debug_dir: str) -> Dict[str, List[str]]:
    """Select 3–5 flow bullets per end market. Require provenance tags. Do not require label echo. Run concurrently."""
    model = cfg["model"]
    out: Dict[str, List[str]] = {}
    labels = list(endmarket_flow_items.keys())
    max_workers = int(cfg.get("llm_concurrency", 6))

    def _do_one(lab) -> Tuple[str, List[str]]:
        lab_str = _label_to_str(lab)
        items = (endmarket_flow_items.get(lab) or [])
        if bool(cfg.get("endmarket_sort_by_date_desc", True)):
            items = sorted(items, key=lambda x: parse_date(x.get("as_of") or ""), reverse=True)

        def _builder(lim: int):
            return [{"label_context": lab_str, "items": _lean_pack(items, lim, cfg, chars_key="evidence_chars_per_item_endmarket")}]

        payload_list, used_limit = _fit_or_reduce_single(
            _builder, items, cfg,
            "max_input_tokens_endmarket_flow", step=0.85, min_items=8,
            limit_key="max_items_per_endmarket_flow_digest", default_limit=80
        )
        guardrail = (
            "RULE: Every bullet MUST end with provenance tag in square brackets: "
            "[title_short, YYYY-MM-DD]. Use provided fields verbatim."
        )
        user = "Current date: " + run_dt.date().isoformat() + "\n" + guardrail + "\n" + json.dumps({"items": payload_list}, ensure_ascii=True)
        instructions = ENDMARKET_FLOW_PROMPT.strip() + "\n" + STYLE_BULLET

        try:
            resp, _ = responses_create_json_schema(
                client,
                model=model,
                instructions=instructions,
                user_input=user,
                json_schema=SCHEMA_ENDMARKET_FLOW,
                max_output_tokens=int(cfg.get("max_output_tokens_endmarket_flow", 3000)),
                reasoning_effort=cfg.get("reasoning_effort", "medium"),
                text_verbosity=cfg.get("text_verbosity", "low"),
            )
            obj = resp if isinstance(resp, dict) else {}
            raw = []
            items_arr = obj.get("items") or []
            if items_arr and isinstance(items_arr[0], dict):
                raw = [x for x in (items_arr[0].get("bullets") or []) if isinstance(x, str)]
            lean = _lean_pack(items, used_limit, cfg, chars_key="evidence_chars_per_item_endmarket")
            final_bullets = clean_bullets_list(_ensure_title_asof_tags(raw, lean))
            return lab_str, final_bullets
        except Exception:
            # Deterministic fallback: synthesize 1-3 bullets from recent items with tags
            lean = _lean_pack(items, min(6, used_limit), cfg, chars_key="evidence_chars_per_item_endmarket")
            bullets = []
            for it in lean[:3]:
                mech = (it.get("signal_type") or "Signal").split()[0].capitalize()
                dirn = (it.get("direction") or "").lower()
                txt = (it.get("evidence") or "").strip()
                tag = f"[{it.get('title_short','')}, {it.get('as_of','')}]"
                piece = f"{mech} {dirn}: {txt} {tag}".strip()
                bullets.append(piece)
            return lab_str, clean_bullets_list(bullets)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, lab) for lab in labels]
        for fut in as_completed(futures):
            lab_str, bullets = fut.result()
            out[lab_str] = bullets

    return out

# ------------------------- Executive summary -------------------------

def exec_summary_pass(client, cfg: Dict[str, Any], run_dt: datetime,
                      selected_tickers: List[str], df_now,  # DataFrame
                      stock_paragraphs: Dict[str, str],
                      flow_digest: Dict[str, Any],
                      primary_parent: Dict[str, str]) -> List[Dict[str, Any]]:
    model = cfg["model"]
    items = []
    for t in selected_tickers:
        items_full = (flow_digest.get(t, {}) or {}).get("items", []) or []
        items.append({
            "ticker": t,
            "parent": primary_parent.get(t, ""),
            "stock_paragraph": (stock_paragraphs.get(t) or "").strip(),
            "flow_items": _lean_pack(items_full, int(cfg.get("max_items_per_ticker_flow_digest", 120)), cfg)[:20]
        })
    payload = {"items": items}

    # Guardrail: force subject identity
    guardrail_id = (
        "IDENTITY RULE: Start main_point with the subject. "
        "Prefix EXACTLY with the ticker and, if provided, the parent end-market in parentheses, "
        "e.g., 'HUBB (Electrical): ...'. Always include the ticker."
    )

    user = "Current date: " + run_dt.date().isoformat() + "\n" + guardrail_id + "\n" + json.dumps(payload, ensure_ascii=True)
    instructions = EXEC_SUMMARY_PROMPT.strip() + "\n" + STYLE_PARAGRAPH

    try:
        resp, _ = responses_create_json_schema(
            client,
            model=model,
            instructions=instructions,
            user_input=user,
            json_schema=SCHEMA_EXEC_SUMMARY,
            max_output_tokens=int(cfg.get("max_output_tokens_exec_summary", 4500)),
            reasoning_effort=cfg.get("reasoning_effort", "low"),
            text_verbosity=cfg.get("text_verbosity", "low"),
        )
        items_obj = (resp if isinstance(resp, dict) else {}).get("items", []) if 'resp' in locals() else (obj.get("items", []) if isinstance(obj, dict) else [])

        out: List[Dict[str, Any]] = []
        for idx, it in enumerate(items_obj or []):
            mp = (it.get("main_point") or "").strip()
            sps = [x for x in (it.get("supporting_points") or []) if isinstance(x, str)]
            flw = [x for x in (it.get("flow") or []) if isinstance(x, str)]

            # Map back to selected ticker by index, then enforce subject prefix
            tkr = selected_tickers[idx] if idx < len(selected_tickers) else ""
            parent = primary_parent.get(tkr, "")
            if tkr and not mp.upper().startswith(tkr.upper()):
                prefix = f"{tkr} ({parent}): " if parent else f"{tkr}: "
                mp = prefix + mp

            # Normalize provenance tags against THIS ticker’s lean items
            lean = _lean_pack((flow_digest.get(tkr, {}) or {}).get("items", []) or [], int(cfg.get("max_items_per_ticker_flow_digest", 120)), cfg)
            flw = _ensure_title_asof_tags(flw, lean)

            out.append({"main_point": mp, "supporting_points": sps, "flow": clean_bullets_list(flw)})
        return out

    except Exception:
        if not bool(cfg.get("fail_open_exec_summary", True)):
            raise

        def _trim_to_limit(s: str, n: int) -> str:
            if not s:
                return ""
            # Normalize dashes for ASCII hygiene
            s = s.replace("\u2013", "-").replace("\u2014", "-").strip()
            if len(s) <= n:
                return s
            cut = s[:n]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            return cut.rstrip(",;: ") + "..."

        max_chars = int(cfg.get("exec_summary_main_point_chars", 260))

        out: List[Dict[str, Any]] = []
        for t in selected_tickers[:8]:
            para = (stock_paragraphs.get(t) or "").strip()
            items_full = (flow_digest.get(t, {}) or {}).get("items", []) or []
            lean = _lean_pack(items_full, 6, cfg)

            flow_bullets = []
            for it in lean[:3]:
                mech = (it.get("signal_type") or "Signal").split()[0].capitalize()
                dirn = (it.get("direction") or "").lower()
                txt = (it.get("evidence") or "").strip()
                tag = f"[{it.get('title_short','')}, {it.get('as_of','')}]"
                flow_bullets.append(f"{mech} {dirn}: {txt} {tag}")

            prefix = f"{t} ({primary_parent.get(t, '')}): " if primary_parent.get(t, "") else f"{t}: "
            main_point = prefix + (_trim_to_limit(para, max_chars) if para else "No paragraph")
            out.append({"main_point": main_point, "supporting_points": [], "flow": clean_bullets_list(flow_bullets)})
        return out
