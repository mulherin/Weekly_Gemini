# core.py
import os
import re
import json
import glob
import math
import yaml
import hashlib
import pandas as pd
from datetime import datetime
from dateutil import parser as dtparser
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import time


# ---------- Style guidelines ----------
STYLE_BULLET = """
STYLE GUIDELINES:
- Write in clear, direct English. Avoid fluff.
- Each bullet is 1 sentence (<= 28 words) unless unavoidable.
- Focus on rate of change (improving, worsening, stabilizing, tracking ahead/behind plan).
- Numbers are optional; include only if incremental and company‑specific.
- Tie every bullet explicitly to a forward impact (orders, revenue, margin, EPS, EBITDA, etc.).
- No macro wallpaper or long‑dated context unless it credibly ties to a potential future earnings revision.
- ASCII only; semicolons allowed; no headings.
"""

STYLE_PARAGRAPH = """
STYLE GUIDELINES:
- 3–5 sentences, single paragraph, thesis‑first.
- First sentence: the 6-12 month thesis anchor; then add one 1-2 quarter supporting mechanism.
- Build a coherent throughline; no bullet-y lists or enumerations.
- No valuation talk. Avoid macro wallpaper unless it credibly shifts the next quarter (e.g., SAAR revision that lifts OEM volumes).
- ASCII only.
"""
# Backward compatibility for any legacy call sites
STYLE_SUFFIX = STYLE_BULLET

# ------------------------- Config & client helpers -------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("model", "gpt-5")
    cfg.setdefault("max_output_tokens", 60000)
    cfg.setdefault("reasoning_effort", "medium")
    cfg.setdefault("text_verbosity", "medium")
    cfg.setdefault("story_sign_deadband", 0.5)
    cfg.setdefault("exec_summary_max_words", 500)
    cfg.setdefault("softdata_write_improve_wording", True)

    # Budgets
    cfg.setdefault("max_input_tokens_storyscore", 260000)
    cfg.setdefault("max_input_tokens_softdata",   260000)
    cfg.setdefault("approx_chars_per_token", 4)

    # Digest caps
    cfg.setdefault("max_items_per_ticker_stock_digest", 300)
    cfg.setdefault("max_items_per_ticker_flow_digest",  180)
    cfg.setdefault("max_items_per_endmarket_stock_digest", 120)
    cfg.setdefault("max_items_per_endmarket_flow_digest",  120)
    cfg.setdefault("tickers_per_call_storyscore_write", 40)
    cfg.setdefault("tickers_per_call_softdata_compute", 30)
    cfg.setdefault("min_items_per_ticker_flow_digest", 60)
    cfg.setdefault("max_input_tokens_endmarket_stock", 48000)
    cfg.setdefault("max_input_tokens_endmarket_flow",  42000)
    cfg.setdefault("evidence_chars_per_item_endmarket", 320)
    cfg.setdefault("fail_open_exec_summary", True)
    cfg.setdefault("use_llm_endmarket", True)

    # Debug toggles
    cfg.setdefault("debug_enabled", False)
    cfg.setdefault("debug_save_payloads", False)
    cfg.setdefault("debug_save_per_ticker_bundles", False)
    cfg.setdefault("debug_save_tables_csv", False)
    cfg.setdefault("debug_trace_tickers", [])

    # Tilt thresholds
    cfg.setdefault("tilt_high_pct", 80)
    cfg.setdefault("tilt_low_pct", 20)

    def _deep_merge(base: dict, override: dict) -> dict:
        out = dict(base or {})
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    # Importance weights — deep‑merge defaults with YAML
    _defaults_iw = {
        "recency_halflife_days": 28,
        "near_term_timeframe_boost": 1.10,
        "source_type": {"sell-side": 1.00, "company": 0.95, "channel": 1.10},
        "kind": {"channel_signal": 1.20, "endmarket_signal": 0.80, "key_comment": 0.95},
        "core_mechanism_boost": 1.12,
        "negative_admission_boost": 1.12,
        "alignment_gain": 0.20,
        "minor_exposure_floor_pct": 15,
        "max_per_source_share": 0.35,
        "multi_ticker_penalty_mode": "sqrt",
        "large_multi_ticker_penalty_N": 6,
        "large_multi_ticker_penalty_factor": 0.75,
        "title_anchor_penalty_factor": 0.60
    }
    cfg["importance_weights"] = _deep_merge(_defaults_iw, cfg.get("importance_weights", {}))


    # Per-note cap for final digests
    cfg.setdefault("max_items_per_note", 3)
    return cfg


def openai_client_from_cfg(cfg: Dict[str, Any]):
    # Local import to avoid making core depend on OpenAI at import time
    from openai import OpenAI
    api_key = (cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("No OpenAI API key found. Set config.api_key or OPENAI_API_KEY env var.")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

def _extract_first_json_object(s: str) -> str:
    """Return the first balanced {...} substring from s (best-effort)."""
    if not s:
        return s
    start = s.find("{")
    if start == -1:
        return s.strip()
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1].strip()
    return s[start:].strip()

def responses_create_json_schema(
    client,
    *,
    model: str,
    instructions: str,
    user_input: str,
    json_schema: Dict[str, Any],
    max_output_tokens: int,
    reasoning_effort: str,
    text_verbosity: str
) -> Tuple[Any, str]:
    """Schema-constrained call with robust retries and JSON-salvage."""
    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "verbosity": text_verbosity,
            "format": {"type": "json_schema", "name": "Schema", "strict": True, "schema": json_schema},
        },
    }

    def _try_extract_parsed(resp_obj) -> Any:
        for attr in ("output_parsed", "parsed"):
            val = getattr(resp_obj, attr, None)
            if isinstance(val, (dict, list)):
                return val
        out = getattr(resp_obj, "output", None) or []
        for item in out:
            content = getattr(item, "content", None) or []
            for c in content:
                for a in ("parsed", "json", "object", "arguments"):
                    v = getattr(c, a, None)
                    if isinstance(v, (dict, list)):
                        return v
                    if isinstance(v, str):
                        try:
                            return json.loads(v)
                        except Exception:
                            pass
        return None

    def _try_extract_text(resp_obj) -> str:
        text = getattr(resp_obj, "output_text", None)
        if text and text.strip():
            return text.strip()
        out = getattr(resp_obj, "output", None) or []
        for item in out:
            content = getattr(item, "content", None) or []
            for c in content:
                if getattr(c, "type", "") == "output_text" and getattr(c, "text", ""):
                    return c.text.strip()
        parts = []
        for item in out:
            content = getattr(item, "content", None) or []
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t:
                    parts.append(t)
        return "".join(parts).strip() if parts else ""

    def _extract_first_json_object(s: str) -> str:
        if not s:
            return s
        start = s.find("{")
        if start == -1:
            return s.strip()
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(s[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1].strip()
        return s[start:].strip()

    def _json_hard_trim(s: str) -> str:
        j = s.rfind("}")
        return s[:j+1] if j != -1 else s

    last_exc: Exception = None
    for attempt in range(3):
        try:
            resp = client.responses.create(**kwargs)
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))
                continue
            raise

        rid = getattr(resp, "id", None)

        parsed = _try_extract_parsed(resp)
        if parsed is not None:
            return parsed, rid

        text = _try_extract_text(resp)
        if text:
            for cand in (text, _extract_first_json_object(text), _json_hard_trim(text)):
                if not cand:
                    continue
                try:
                    return json.loads(cand), rid
                except json.JSONDecodeError as e:
                    last_exc = e
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))
                continue

        last_exc = RuntimeError("Empty response from Responses API (no parsed JSON or text).")
        if attempt < 2:
            time.sleep(0.7 * (attempt + 1))
            continue

    raise last_exc if last_exc else RuntimeError("Empty response from Responses API (no parsed JSON or text).")

# ------------------------- IO & misc helpers -------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)

def save_text(path: str, txt: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def save_df_csv(path: str, df: pd.DataFrame):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def parse_jsonl_dir(jsonl_dir: str) -> List[Dict[str, Any]]:
    items = []
    for path in glob.glob(os.path.join(jsonl_dir, "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except json.JSONDecodeError:
                    continue
    return items

def parse_date(s: str) -> datetime:
    try:
        return dtparser.parse(s).replace(tzinfo=None)
    except Exception:
        return datetime.min

def sha1_of(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=True, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

# ------------------------- Labeling, routing, filters -------------------------

def normalize_label(s: str) -> str:
    if not s:
        return ""
    trans = {
        0x00A0: 0x0020,  # NBSP -> space
        0x2010: 0x002D,  # hyphen -> '-'
        0x2011: 0x002D,  # non-breaking hyphen -> '-'
        0x2012: 0x002D,  # figure dash -> '-'
        0x2013: 0x002D,  # en dash -> '-'
        0x2014: 0x002D,  # em dash -> '-'
        0x2212: 0x002D,  # minus -> '-'
        0x005F: 0x002D,  # underscore -> '-'
    }
    s2 = s.translate(trans)
    s2 = "".join(ch for ch in s2 if ord(ch) >= 32)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def _is_nullish_submarket(s: str) -> bool:
    h = re.sub(r"[^A-Z0-9]+", "", (s or "").upper())
    return h in {"NA", "NAN", "NONE", "NULL"} or h == ""

def apply_content_filters(evidence: str, content_filters: Dict[str, List[str]]) -> bool:
    ev = (evidence or "")
    ev_low = ev.lower()
    # Allow-list: any number+unit signal should pass
    if re.search(r"\b\d+(\.\d+)?\s*(%|bps|m\/m|q\/q|y\/y|t|mt|mw|gw|k|m|bn|b)\b", ev_low):
        return False
    for _, kws in (content_filters or {}).items():
        for kw in (kws or []):
            if kw and re.search(rf"\b{re.escape(str(kw).lower())}\b", ev_low):
                return True
    return False

def canonical_endmarkets_from_text(text: str, aliases: Dict[str, List[str]]) -> List[str]:
    if not text:
        return []
    t = text.lower()
    hits = []
    for label, words in aliases.items():
        if any(w.lower() in t for w in words):
            hits.append(normalize_label(label))
    return sorted(set(hits))

def labels_for_item(it: Dict[str, Any], aliases: Dict[str, List[str]]) -> List[str]:
    """
    Single source of truth: use structured end_market and sub_market.
    Do not add alias-based labels when these fields exist.
    """
    labs = []
    em_raw = (it.get("end_market") or "")
    sm_raw = (it.get("sub_market") or "")
    em = normalize_label(em_raw.strip())
    sm = normalize_label(sm_raw.strip())
    if em and sm and not _is_nullish_submarket(sm_raw):
        labs.append(f"{em} | {sm}")
    elif em:
        labs.append(em)
    # If neither structured field is present, fall back to alias detection from evidence.
    if not labs:
        labs.extend(canonical_endmarkets_from_text(it.get("evidence") or "", aliases))
    return [normalize_label(x) for x in labs]

def is_valid_ticker(x: Any) -> bool:
    t = str(x).strip().upper() if x is not None else ""
    if not t or t in {"N/A", "NA", "NONE", "TICKER", "NAN"} or len(t) > 12:
        return False
    return bool(re.match(r'^[A-Z0-9]+([.-][A-Z0-9]+)?$', t))

def build_universe(cfgs: Dict[str, Any], valuation_tickers: List[str]) -> Dict[str, str]:
    desc = cfgs.get("company_descriptions", {}) or {}
    short = cfgs.get("common_names", {}) or {}
    cexp  = cfgs.get("company_endmarket_exposure", {}) or {}
    ind2t = cfgs.get("industry_to_tickers", {}) or {}

    universe_set = set()
    universe_set.update(str(t).upper() for t in (desc.keys() or []))
    universe_set.update(str(t).upper() for t in (short.keys() or []))
    universe_set.update(t for t in (valuation_tickers or []) if is_valid_ticker(t))

    for _, mp in (cexp or {}).items():
        for t in (mp or {}).keys():
            universe_set.add(str(t).upper())

    def _add_ticker(x):
        if is_valid_ticker(x):
            universe_set.add(str(x).upper())

    if isinstance(ind2t, dict):
        for v in ind2t.values():
            if isinstance(v, list):
                for t in v:
                    _add_ticker(t)
        nested = ind2t.get("industry_to_end_markets")
        if isinstance(nested, dict):
            for submap in nested.values():
                if isinstance(submap, dict):
                    for arr in submap.values():
                        if isinstance(arr, list):
                            for rec in arr:
                                if isinstance(rec, dict) and "ticker" in rec:
                                    _add_ticker(rec["ticker"])

    out = {}
    for t in sorted(universe_set):
        if t in short and short[t]:
            out[t] = str(short[t])
        elif t in desc and desc[t]:
            s = str(desc[t]).strip()
            dot = s.find(".")
            out[t] = s[:dot+1] if dot != -1 else (s[:120] if s else t)
        else:
            out[t] = t
    return out

# ------------------------- Windows & views -------------------------

def route_to_tickers(item: Dict[str, Any], approved: Dict[str, str], aliases: Dict[str, List[str]], ind2tkr: Dict[str, Any]):
    tkrs = item.get("tickers") or []
    tkrs = [t.upper() for t in tkrs if isinstance(t, str)]
    tkrs = [t for t in tkrs if t in approved]
    if tkrs:
        return sorted(set(tkrs)), "company"
    return [], "group"

def in_window(as_of: datetime, run_date: datetime, days: int) -> bool:
    if as_of == datetime.min:
        return False
    return (run_date - as_of).days <= days and as_of <= run_date

def split_flow_stock(items: List[Dict[str, Any]], run_date: datetime, flow_days: int, stock_days: int):
    flow, stock = [], []
    for it in items:
        d = parse_date(it.get("as_of") or it.get("date") or "")
        it["_as_of_dt"] = d
        if in_window(d, run_date, flow_days):
            flow.append(it)
        if in_window(d, run_date, stock_days):
            stock.append(it)
    return flow, stock

def build_views(all_items: List[Dict[str, Any]],
                universe: Dict[str, str],
                cfgs: Dict[str, Any],
                run_date: datetime,
                flow_days: int,
                stock_days: int):
    end_alias = cfgs["end_market_aliases"]
    content_filters = cfgs["content_filters"]

    filtered, routed = [], []
    for it in all_items:
        evidence = it.get("evidence", "")
        if apply_content_filters(evidence, content_filters):
            continue
        it["_labels"] = labels_for_item(it, end_alias)
        filtered.append(it)
        tkrs, spec = route_to_tickers(it, approved=universe, aliases=end_alias, ind2tkr=cfgs["industry_to_tickers"])
        if spec == "company" and tkrs:
            it["_tickers"] = sorted(set(tkrs))
            it["_specificity"] = spec
            routed.append(it)

    flow_all_em, stock_all_em = split_flow_stock(filtered, run_date, flow_days, stock_days)
    flow_all_tkr, stock_all_tkr = split_flow_stock(routed, run_date, flow_days, stock_days)

    flow_by_tkr = defaultdict(list)
    stock_by_tkr = defaultdict(list)
    # Use tuple keys (parent, child_or_None) and roll children up to parent.
    endmkt_flow_by_label: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    endmkt_stock_by_label: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for it in flow_all_tkr:
        orig_len = len(it.get("_tickers", [])) or 1
        for t in it.get("_tickers", []):
            c = dict(it)
            c["_tickers"] = [t]
            c["_orig_num_tickers"] = orig_len
            flow_by_tkr[t].append(c)

    for it in stock_all_tkr:
        orig_len = len(it.get("_tickers", [])) or 1
        for t in it.get("_tickers", []):
            c = dict(it)
            c["_tickers"] = [t]
            c["_orig_num_tickers"] = orig_len
            stock_by_tkr[t].append(c)

    def _add_endmarket(it: Dict[str, Any], into: Dict[Tuple[str, str], List[Dict[str, Any]]]):
        em_raw = (it.get("end_market") or "")
        sm_raw = (it.get("sub_market") or "")
        parent = normalize_label(em_raw.strip())
        child = None if _is_nullish_submarket(sm_raw) else normalize_label(sm_raw.strip()) or None
        if not parent:
            return
        into[(parent, None)].append(it)
        if child:
            into[(parent, child)].append(it)

    for it in flow_all_em:
        _add_endmarket(it, endmkt_flow_by_label)

    for it in stock_all_em:
        _add_endmarket(it, endmkt_stock_by_label)

    return flow_by_tkr, stock_by_tkr, endmkt_flow_by_label, endmkt_stock_by_label

# ------------------------- Importance & mechanism helpers -------------------------

def _mechanism_kind(signal_type: str) -> str:
    st = (signal_type or "").lower()
    if any(k in st for k in ["volume","order","booking","sell","traffic","comp","pos"]):
        return "demand"
    if any(k in st for k in ["lead","backlog","lead_time"]):
        return "lead_backlog"
    if "pricing" in st or "price" in st:
        return "pricing"
    if "inventory" in st:
        return "inventory"
    if any(k in st for k in ["cost","margin","cogs"]):
        return "cost"
    if any(k in st for k in ["capacity","utilization","capex"]):
        return "capacity"
    if any(k in st for k in ["logistics","freight","shipping"]):
        return "logistics"
    return "qualitative"

def _mechanism_tags_from_item(it: Dict[str, Any]) -> List[str]:
    st = (it.get("signal_type") or "").lower()
    ev = (it.get("evidence") or "").lower()
    tags = set()
    if any(k in st for k in ["volume","order","booking","sell-through","pos","comps","traffic"]) or any(k in ev for k in ["order","booking","demand","shipment","utiliz","volume","sell-through","traffic","comps"]):
        tags.add("orders")
    if "backlog" in st or "backlog" in ev:
        tags.add("backlog")
    if "lead" in st or "lead time" in ev or "lead-time" in ev or "leadtime" in ev:
        tags.add("lead_times")
    if "price" in st or "pricing" in st or "price" in ev or "realization" in ev or "list price" in ev:
        tags.add("pricing")
    if "mix" in st or "mix" in ev:
        tags.add("mix")
    if "inventory" in st or "inventory" in ev or "destock" in ev or "restock" in ev:
        tags.add("inventory")
    if "capacity" in st or "utilization" in st or "supply" in st or "capacity" in ev or "production cut" in ev or "production cuts" in ev:
        tags.add("capacity")
    if "cost" in st or "margin" in st or "cost" in ev or "margin" in ev or "raw material" in ev:
        tags.add("cost")
    if "logistics" in ev or "freight" in ev or "shipping" in ev:
        tags.add("logistics")
    return sorted(list(tags))[:3]

def _source_title_key(it: Dict[str, Any]) -> str:
    s = normalize_label(str(it.get("source", "") or ""))
    t = normalize_label(str(it.get("title", "") or ""))
    if s or t:
        return f"{s}::{t}".strip(":")
    return sha1_of({"evidence": (it.get("evidence") or "")[:160]})

def _note_title_tickers(title: str) -> List[str]:
    if not title:
        return []
    m = re.search(r"\(([A-Z0-9 ,.-]+)\)", title)
    if not m:
        return []
    raw = [x.strip().upper() for x in m.group(1).split(",")]
    return [x for x in raw if re.fullmatch(r"[A-Z0-9.-]+", x)]

def _near_term_flag(it: Dict[str, Any], run_date: datetime) -> bool:
    tfs = [str(x).upper() for x in (it.get("timeframe") or [])]
    if not tfs:
        return False
    q = (run_date.month - 1) // 3 + 1
    nxt = 1 if q == 4 else q + 1
    near = {f"Q{q}", f"Q{nxt}", "QTD", "MTD", "H1" if q <= 2 else "H2", "1H" if q <= 2 else "2H"}
    return any(tf in near for tf in tfs)

def importance_weight_item(it: Dict[str, Any]) -> float:
    kind = (it.get("kind") or "").lower()
    st = (it.get("signal_type") or "").lower()
    d  = (it.get("direction") or "").lower()
    nt = len(it.get("tickers") or [])

    base = 1.0
    if kind == "channel_signal":
        base += 1.5
    elif kind == "key_comment":
        base += 0.75
    elif kind == "endmarket_signal":
        base += 0.25

    mech = _mechanism_kind(st)
    if mech in ("demand","lead_backlog"):
        base += 1.25
    elif mech == "pricing":
        base += 0.5
    elif mech in ("inventory","capacity","logistics"):
        base += 0.25
    elif mech == "cost":
        base += 0.5
    else:
        base -= 0.25

    if d in ("improving","worsening"):
        base += 0.5
    elif d in ("unknown","flat","mixed",""):
        base -= 0.25

    if nt == 0:
        base -= 0.5
    elif nt == 1:
        base += 1.0
    elif 2 <= nt <= 4:
        base -= 0.5
    elif nt >= 5:
        base -= 1.0
    return max(0.1, round(base, 2))

def mechanism_impact_sign(signal_type: str, direction: str) -> int:
    st = (signal_type or "").lower()
    d  = (direction or "").lower()
    if not d or d in {"mixed", "unknown", "flat"}:
        return 0
    if d == "improving":
        return +1
    if d == "worsening":
        return -1
    is_cost      = ("cost" in st) or ("margin" in st)
    is_inventory = "inventory" in st
    is_pricing   = "price" in st or "pricing" in st
    is_volume    = ("volume" in st) or ("order" in st) or ("booking" in st) or ("sell-through" in st) or ("pos" in st) or ("traffic" in st) or ("comps" in st)
    is_backlogLT = ("backlog" in st) or ("lead" in st)
    if is_cost or is_inventory:
        return +1 if d == "down" else (-1 if d == "up" else 0)
    if is_backlogLT:
        return +1 if d == "down" else (-1 if d == "up" else 0)
    if is_pricing or is_volume:
        return +1 if d == "up" else (-1 if d == "down" else 0)
    return +1 if d == "up" else (-1 if d == "down" else 0)

def compute_item_importance(it: Dict[str, Any],
                            ticker: str,
                            run_cfg: Dict[str, Any],
                            canonical_cfgs: Dict[str, Any],
                            run_date: datetime) -> float:
    iw = run_cfg.get("importance_weights", {})

    # Company specificity penalty
    n_in_note = int(it.get("_orig_num_tickers", 1) or 1)
    mode = iw.get("multi_ticker_penalty_mode", "sqrt")
    if mode == "linear":
        specificity = 1.0 / float(n_in_note)
    elif mode == "log":
        specificity = 1.0 / max(1.0, math.log2(n_in_note + 1.0))
    else:
        specificity = 1.0 / math.sqrt(float(n_in_note))
    if n_in_note == 1:
        specificity = 1.0
    if n_in_note > int(iw.get("large_multi_ticker_penalty_N", 6)):
        specificity *= float(iw.get("large_multi_ticker_penalty_factor", 0.75))

    # Source & kind
    src_type = str(it.get("source_type", "") or "").lower()
    kind = str(it.get("kind", "") or "").lower()
    w_src = float(iw.get("source_type", {}).get(src_type, 1.0))
    w_kind = float(iw.get("kind", {}).get(kind, 1.0))

    # Mechanism boost
    mech_tags = _mechanism_tags_from_item(it)
    core_mech = {"orders", "backlog", "lead_times", "pricing", "mix", "inventory", "capacity", "cost", "logistics"}
    w_mech = float(iw.get("core_mechanism_boost", 1.0)) if any(m in core_mech for m in mech_tags) else 1.0

    # Negative admissions boost
    dirn = str(it.get("direction", "") or "").lower()
    st = str(it.get("signal_type", "") or "").lower()
    neg_like_family = any(k in st for k in ["order", "volume", "booking", "shipment", "backlog", "lead"])
    neg_state = dirn in {"down", "lower", "shorter", "decline", "worsening"}
    w_neg = float(iw.get("negative_admission_boost", 1.0)) if (neg_like_family and neg_state) else 1.0

    # Alignment to exposures
    def exposure_weights_for_ticker(canonical_cfgs: Dict[str, Any], ticker: str) -> Dict[str, float]:
        cexp = canonical_cfgs.get("company_endmarket_exposure", {}) or {}
        t = (ticker or "").upper()
        if not isinstance(cexp, dict):
            return {}
        maybe = cexp.get(t)
        if isinstance(maybe, dict):
            return {normalize_label(k): float(v) for k, v in maybe.items() if isinstance(v, (int, float))}
        for _, mp in cexp.items():
            if isinstance(mp, dict) and t in mp and isinstance(mp[t], dict):
                return {normalize_label(k): float(v) for k, v in mp[t].items() if isinstance(v, (int, float))}
        return {}

    exposure = exposure_weights_for_ticker(canonical_cfgs, ticker)
    labels = [normalize_label(x) for x in (it.get("_labels") or [])]
    align_score = 0.0
    for lab, wt in exposure.items():
        if normalize_label(lab) in labels:
            align_score = max(align_score, float(wt))
    minor_floor = float(iw.get("minor_exposure_floor_pct", 0.0))
    if align_score:
        scaled = max(align_score, minor_floor) / 100.0
        w_align = 1.0 + float(iw.get("alignment_gain", 0.0)) * scaled
    else:
        w_align = 1.0

    # Near-term timeframe boost & recency decay
    w_tf = float(iw.get("near_term_timeframe_boost", 1.0)) if _near_term_flag(it, run_date) else 1.0
    as_of = it.get("_as_of_dt") or datetime.min
    age_days = (run_date - as_of).days if as_of != datetime.min else 999
    half_life = max(1, int(iw.get("recency_halflife_days", 28)))
    recency_min = float(iw.get("recency_min_weight", 0.0))
    w_rec = 0.5 ** (age_days / float(half_life))

    # Title anchor penalty if note headline names other tickers but not this one
    ttl_tkrs = set(_note_title_tickers(str(it.get("title") or "")))
    w_anchor = float(iw.get("title_anchor_penalty_factor", 0.60)) if (ttl_tkrs and ticker.upper() not in ttl_tkrs) else 1.0

    base = specificity * w_src * w_kind * w_mech * w_neg * w_align * w_tf * w_anchor
    return base * w_rec

# ------------------------- Summaries, quality, bullets -------------------------

def summarize_items(items: List[Dict[str, Any]], limit: int, run_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    lim = int((run_cfg or {}).get("evidence_chars_per_item", 1200))
    for it in items[:max(0, limit)]:
        ev = (it.get("evidence") or "")
        ev = re.sub(r"\s*\[\s*Disclosure.*$", "", ev, flags=re.I | re.S)
        tail = ev[lim:lim*2] if len(ev) > lim else ""
        out.append({
            "title": it.get("title"),
            "as_of": it.get("as_of"),
            "source": it.get("source"),
            "source_type": it.get("source_type"),
            "kind": it.get("kind"),
            "signal_type": it.get("signal_type"),
            "direction": it.get("direction"),
            "region": it.get("region"),
            "timeframe": it.get("timeframe"),
            "labels": [normalize_label(x) for x in (it.get("_labels") or [])],
            "specificity": it.get("_specificity"),
            "num_tickers_in_note": len(it.get("tickers") or []),
            "importance_weight": importance_weight_item(it),
            "evidence": ev[:lim],
            "evidence_tail": tail
        })
    return out

def quality_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    spec = Counter([it.get("_specificity", "company") for it in items])
    dates = [it.get("_as_of_dt") for it in items if it.get("_as_of_dt")]
    date_counts = Counter([d.date().isoformat() for d in dates])
    top_date_share = (max(date_counts.values()) / max(1, sum(date_counts.values()))) if date_counts else 0.0
    title_counts = Counter([(it.get("title") or "").strip() for it in items if it.get("title")])
    source_counts = Counter([(it.get("source") or "").strip() for it in items if it.get("source")])
    total = float(len(items) or 1)
    top_title_share = (max(title_counts.values()) / total) if title_counts else 0.0
    top_source_share = (max(source_counts.values()) / total) if source_counts else 0.0
    nt_list = [len(it.get("tickers") or []) for it in items if isinstance(it.get("tickers"), list)]
    single_ticker = sum(1 for n in nt_list if n == 1)
    with_tickers = sum(1 for n in nt_list if n > 0)
    company_specific_share = (single_ticker / float(with_tickers or 1))
    avg_tickers_per_note = (sum(nt_list) / float(len(nt_list) or 1))
    return {
        "counts": dict(spec),
        "unique_dates": len(set(date_counts.keys())),
        "single_source_dominance": bool(max(top_date_share, top_title_share, top_source_share) >= 0.6),
        "top_title_share": round(top_title_share, 3),
        "top_source_share": round(top_source_share, 3),
        "company_specific_share": round(company_specific_share, 3),
        "avg_tickers_per_note": round(avg_tickers_per_note, 2)
    }

def clean_flow_bullet(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = re.sub(r"(?i)\bflow[_\s-]*payload\b", "", s).strip()
    s = re.sub(r"(?i)\bflow[_\s-]*context\b", "", s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    m = re.match(r"(?i)^\s*(costs?|pricing|prices?|inventory|backlog|lead[\s-]?times?|orders?|order|volume|mix|capacity|logistics|freight|shipping)\s*:\s*(.*)$", s)
    if m:
        label = m.group(1).lower().strip()
        rest = m.group(2).strip()
        if label.startswith("lead"):
            label = "lead times"
        subj_map = {
            "cost": "Costs", "costs": "Costs", "pricing": "Pricing",
            "price": "Prices", "prices": "Prices", "inventory": "Inventory",
            "orders": "Orders", "order": "Orders", "volume": "Volume",
            "backlog": "Backlog", "lead times": "Lead times",
            "capacity": "Capacity", "logistics": "Logistics",
            "freight": "Freight", "shipping": "Shipping", "mix": "Product mix",
        }
        subj = subj_map.get(label, label.capitalize())
        if re.match(r"(?i)^(up|down|flat|higher|lower|improving|worsening|shorter|longer)\b", rest):
            verb = "are" if subj.lower() in {"costs", "prices", "orders", "lead times"} else "is"
            s = f"{subj} {verb} {rest}"
        else:
            s = f"{subj} {rest}"
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s

def clean_bullets_list(arr: List[str]) -> List[str]:
    out = []
    for x in (arr or []):
        y = clean_flow_bullet(x)
        if y:
            out.append(y)
    if not out:
        out = ["No incremental data"]
    return out

# Excel-safe text sanitizer (remove illegal control chars and zero-widths)
_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def sanitize_text_for_excel(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    s = val.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # normalize dashes to ASCII
    return _ILLEGAL_XLSX_RE.sub("", s)

def fit_payload_to_budget(payload_obj: Any, max_input_tokens: int, approx_chars_per_token: int) -> bool:
    s = json.dumps(payload_obj, ensure_ascii=True)
    est_tokens = len(s) // max(1, approx_chars_per_token)
    return est_tokens <= max_input_tokens

# ------------------------- Digests -------------------------

def build_digests(flow_items_by_tkr: Dict[str, List[Dict[str, Any]]],
                  stock_items_by_tkr: Dict[str, List[Dict[str, Any]]],
                  max_stock_items: int,
                  max_flow_items: int,
                  run_cfg: Dict[str, Any],
                  canonical_cfgs: Dict[str, Any],
                  run_date: datetime):
    stock_digest, flow_digest = {}, {}
    all_tickers = sorted(set(list(stock_items_by_tkr.keys()) + list(flow_items_by_tkr.keys())))
    max_share = float(run_cfg.get("importance_weights", {}).get("max_per_source_share", 0.80))
    hard_cap_per_note = int(run_cfg.get("max_items_per_note", 999))

    # --- Local merge helper to override importance weights per digest ---
    import copy as _copy
    def _merge_iw_cfg(base_cfg: Dict[str, Any], iw_override: Dict[str, Any]) -> Dict[str, Any]:
        out = _copy.deepcopy(base_cfg or {})
        iw_base = dict(out.get("importance_weights", {}))
        for k, v in (iw_override or {}).items():
            if isinstance(v, dict) and isinstance(iw_base.get(k), dict):
                iw_base[k] = {**iw_base.get(k, {}), **v}
            else:
                iw_base[k] = v
        out["importance_weights"] = iw_base
        return out

    # Per-digest configs: stock (StoryScore) vs flow (SoftData)
    run_cfg_stock = _merge_iw_cfg(run_cfg, run_cfg.get("importance_weights_stock", {}))
    run_cfg_flow  = _merge_iw_cfg(run_cfg, run_cfg.get("importance_weights_flow", {}))

    def _select(items: List[Dict[str, Any]], limit: int, tkr: str, cfg_local: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not items:
            return []
        for it in items:
            it["_importance"] = compute_item_importance(it, tkr, cfg_local, canonical_cfgs, run_date)
        use_date_tie = bool(cfg_local.get("select_use_date_tiebreak", True))
        if use_date_tie:
            cands = sorted(items, key=lambda x: (float(x.get("_importance", 0.0)), x.get("_as_of_dt")), reverse=True)
        else:
            cands = sorted(items, key=lambda x: float(x.get("_importance", 0.0)), reverse=True)
        distinct = list({(normalize_label(it.get("source") or ""), normalize_label(it.get("title") or "")) for it in cands})
        distinct_sources = len(distinct)
        inv = int(round(1.0 / max(1e-9, max_share)))
        cap = limit if distinct_sources <= inv else max(1, int(math.ceil(max_share * max(1, limit))))
        keep, counts, per_note_counts = [], defaultdict(int), defaultdict(int)
        for it in cands:
            key = _source_title_key(it)
            if counts[key] >= cap:
                if len(keep) < int(limit * 0.85):
                    pass
                else:
                    continue
            if per_note_counts[key] >= hard_cap_per_note:
                if len(keep) < int(limit * 0.85):
                    pass
                else:
                    continue
            keep.append(it)
            counts[key] += 1
            per_note_counts[key] += 1
            if len(keep) >= limit:
                break
        return keep

    def _exposure_weights_for_ticker(canonical_cfgs: Dict[str, Any], ticker: str) -> Dict[str, float]:
        cexp = canonical_cfgs.get("company_endmarket_exposure", {}) or {}
        t = (ticker or "").upper()
        if isinstance(cexp, dict):
            if t in cexp and isinstance(cexp[t], dict):
                return {normalize_label(k): float(v) for k, v in cexp[t].items() if isinstance(v, (int, float))}
            for _, mp in cexp.items():
                if isinstance(mp, dict) and t in mp and isinstance(mp[t], dict):
                    return {normalize_label(k): float(v) for k, v in mp[t].items() if isinstance(v, (int, float))}
        return {}


    for tkr in all_tickers:
        sitems = _select(stock_items_by_tkr.get(tkr, []), max_stock_items, tkr, run_cfg_stock)
        fitems = _select(flow_items_by_tkr.get(tkr, []),  max_flow_items,  tkr, run_cfg_flow)
        expw = _exposure_weights_for_ticker(canonical_cfgs, tkr)

        stock_digest[tkr] = {
            "items": summarize_items(sitems, max_stock_items, run_cfg),
            "signal_quality": quality_summary(sitems),
            "exposure_weights": expw
        }
        flow_digest[tkr]  = {
            "items": summarize_items(fitems, max_flow_items, run_cfg),
            "signal_quality": quality_summary(fitems),
            "exposure_weights": expw
        }
        
    return stock_digest, flow_digest

# ------------------------- Scores, deltas, selection -------------------------

def story_sign(x: float, deadband: float) -> int:
    if x > deadband:
        return 1
    if x < -deadband:
        return -1
    return 0

def quantize_with_ranks_debug(raw_scores: Dict[str, float]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    ranked = sorted(raw_scores.items(), key=lambda kv: kv[1])
    n = len(ranked)
    buckets, table = {}, []
    for i, (t, val) in enumerate(ranked):
        bin_idx = int((i * 11) / max(1, n))
        if bin_idx > 10:
            bin_idx = 10
        bucket = bin_idx - 5
        buckets[t] = bucket
        table.append({"Ticker": t, "RawStoryScore": val, "RankIndex": i, "Bucket": bucket})
    return buckets, table

def rebalance_softdata(soft_compute: Dict[str, Any],
                       flow_digest: Dict[str, Any],
                       run_cfg: Dict[str, Any],
                       target_pos_share: float,
                       target_neg_share: float) -> Dict[str, int]:
    tkrs = sorted(soft_compute.keys())
    n = len(tkrs)
    target_pos = int(round(n * target_pos_share))
    target_neg = int(round(n * target_neg_share))
    def latest_dt(t: str) -> datetime:
        arr = flow_digest.get(t, {}).get("items", [])
        if not arr:
            return datetime.min
        return parse_date(arr[0].get("as_of") or "")
    conf_rank = {"high": 2, "medium": 1, "low": 0}
    pos_cand = [t for t in tkrs if soft_compute.get(t, {}).get("softdata") == 1 and not soft_compute.get(t, {}).get("conflict")]
    neg_cand = [t for t in tkrs if soft_compute.get(t, {}).get("softdata") == -1 and not soft_compute.get(t, {}).get("conflict")]
    use_latest_bias = bool(run_cfg.get("softdata_rebalance_latest_first", True))
    def _key(t):
        conf = conf_rank.get(soft_compute.get(t, {}).get("confidence", "medium"), 1)
        dt_key = latest_dt(t) if use_latest_bias else datetime.min
        return (conf, dt_key)

    pos_cand.sort(key=_key, reverse=True)
    neg_cand.sort(key=_key, reverse=True)
    final = {t: 0 for t in tkrs}
    for t in pos_cand[:target_pos]:
        final[t] = 1
    for t in neg_cand[:target_neg]:
        final[t] = -1
    for t in tkrs:
        if soft_compute.get(t, {}).get("conflict"):
            final[t] = 0
    return final

def select_exec_flags(df_now: pd.DataFrame) -> List[str]:
    if df_now.empty:
        return []
    df = df_now.copy()
    df["abs_div"] = df["Divergence"].abs()
    df["abs_ds"] = df["StoryScore_Change"].abs()
    df["abs_df"] = df["SoftData_Change"].abs()
    df["conflict_weight"] = df["Flow_Conflict"].astype(int)
    df["rank_score"] = (df["abs_div"] * 3.0) + (df["abs_df"] * 2.0) + (df["abs_ds"] * 1.0) + (df["conflict_weight"] * 2.0)
    df = df.sort_values(["rank_score", "abs_div", "abs_df"], ascending=False)
    return list(df["Ticker"].head(12))

def latest_prior_scores(output_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(output_dir, "WeeklyRecap_Scores_*.xlsx")))
    return files[-1] if files else ""

def compute_deltas(current_df: pd.DataFrame, prior_path: str) -> pd.DataFrame:
    if not prior_path or not os.path.exists(prior_path):
        current_df["StoryScore_Change"] = 0.0
        current_df["SoftData_Change"] = 0
        return current_df
    prev = pd.read_excel(prior_path)
    for c in ["Ticker", "StoryScore", "SoftData"]:
        if c not in prev.columns:
            prev[c] = None
    merged = current_df.merge(prev[["Ticker", "StoryScore", "SoftData"]].rename(
        columns={"StoryScore": "StoryScore_prev", "SoftData": "SoftData_prev"}), on="Ticker", how="left")
    merged["StoryScore_Change"] = merged["StoryScore"] - merged["StoryScore_prev"].fillna(0.0)
    merged["SoftData_Change"] = merged["SoftData"] - merged["SoftData_prev"].fillna(0)
    return merged.drop(columns=["StoryScore_prev", "SoftData_prev"])

# ------------------------- Debugging helpers -------------------------

def make_pos_neg_neutral(flow_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pos, neg, neu = [], [], []
    for it in flow_items:
        s = mechanism_impact_sign(it.get("signal_type",""), it.get("direction",""))
        rec = {
            "title": it.get("title"),
            "as_of": it.get("as_of"),
            "signal_type": it.get("signal_type"),
            "direction": it.get("direction"),
            "evidence": (it.get("evidence") or "")[:600]
        }
        if s > 0:
            pos.append(rec)
        elif s < 0:
            neg.append(rec)
        else:
            neu.append(rec)
    return {"positive": pos, "negative": neg, "neutral": neu}

# ------------------------- Config file loaders -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_configs(cfg_dir: str) -> Dict[str, Any]:
    files = {
        "common_names": "common_names.json",
        "company_descriptions": "company_descriptions.json",
        "company_endmarket_exposure": "company_endmarket_exposure.json",
        "content_filters": "content_filters.json",
        "end_market_aliases": "end_market_aliases.json",
        "industry_to_tickers": "industry_to_tickers.json"
    }
    cfgs = {}
    for k, fname in files.items():
        full = os.path.join(cfg_dir, fname)
        cfgs[k] = load_json(full)

    ind = cfgs.get("industry_to_tickers") or {}
    if isinstance(ind, dict) and "industry_to_end_markets" in ind:
        nested = ind.get("industry_to_end_markets", {}) or {}
        flat = {}
        for parent, submap in nested.items():
            tickers = []
            for arr in (submap or {}).values():
                for rec in (arr or []):
                    t = str(rec.get("ticker", "")).upper()
                    if t:
                        tickers.append(t)
            flat[parent] = sorted(set(tickers))
        cfgs["industry_to_tickers"] = flat
        cfgs["industry_to_end_markets"] = nested
    return cfgs
