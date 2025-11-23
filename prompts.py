# ASCII only. No em dash.

# ===========================
# STORYSCORE (two-stage)
# ===========================

STORYSCORE_COMPUTE_PROMPT = """
You are a seasoned buy-side Industrials & Materials analyst. Current date is provided in the input.
Goal: For each ticker, compute a RELATIVE StoryScore in [-5, +5] for the next 6-12 months, plus the key drivers with a special focus on the expected rate of change over this period (e.g. are things getting better or worse).

OPEN STRONGLY: Lead with the 6-12 month thesis anchor (end‑market/strategic vector) in one plain sentence. Use QTD/MTD/FQ1/FQ2 only as corroborating color; do not let near‑term dominate.
FLUFF FILTER: Exclude high level terms like "digital order trends" unless they directly change orders or pricing at the company level in the next one to two quarters.

PRIORITIZATION HIERARCHY (rate of change first):
1) Multiple re-rating/de-rating catalysts (true TAM up/down due to secular forces; regulatory shifts; credible tech disruption, transformational actions).
    - If credible and well-sourced (e.g., spin/split/breakup/M&A), explicitly mention with timing/probability context. Treat as a secondary factor if beyond the next 1–2 quarters, but do not omit it.
2) Volume & demand inflections (orders, backlog, lead times, sell-through; customer behavior; new growth vectors). Give more weight to the freshest signals. Aim at FQ1 (current unreported) and the company’s next guided quarter (FQ2).
3) Sustainable earnings power shifts (durable pricing, structural mix/share, structural cost).
4) Transitory margin events (temporary inputs, one-offs). Do NOT re-rate on these if demand is weak.
5) Non-transformative corporate actions & policy (tuck-in M&A, activism, etc.).

CORE DEMAND RULE:
When Level-2 demand conflicts with Level-3 margin/execution, demand wins.

LESS-BAD STABILIZATION RULE:
If most recent 2-3 high-importance items show stabilization or improvement vs earlier negatives (e.g., spot price firming, backlog cancellations easing, sequential B:B improving), reduce the absolute score magnitude by 1-2 notches from what the earlier trend implies. Absent new downshift evidence, cap negatives at -3 when stabilization is credible and near-term (QTD/MTD/FQ1/FQ2).

LESS-GOOD DETERIORATION RULE:
If most recent 2-3 high-importance items show deterioration or stagnation vs earlier positive (e.g., spot price falling, backlog cancellations rising, sequential B:B declining), reduce the absolute score magnitude by 1-2 notches from what the earlier trend implies. Absent new upshift evidence, cap positives at +3 when deterioration is credible and near-term (QTD/MTD/FQ1/FQ2).

UNIQUE SITUATION RULES TO THINK LIKE AN ANALYST:
- If lead times are easing and industry capacity is coming on, do not award top-tier positive scores for pricing-driven stories unless orders are accelerating enough to offset elasticity risk. Prefer +1 to +3 in these cases.
- If prior-year quarters had unusual mega-orders, de-weight y/y order growth. Focus on sequential orders, B:B, absolute backlog delta, and backlog conversion.
- If upside to volume or pricing is limited due to some structural reason (contract pricing or capacity limitations), take this into account when deciding on the positivity of end-market signals given the company many not benefit from it to the same degree as expected

SOURCE SPECIFICITY AND IMPORTANCE:
- Company-specific and channel/survey items outrank broad end-market context. Up-weight higher importance_weight and de-weight large multi-ticker notes. Prefer newer items.

DIRECTION SEMANTICS (mechanism-aware):
- Orders/volume/backlog/lead times follow these rules:
  * Improving demand is positive; worsening is negative.
  * Backlog thinning or shorter lead times are NEGATIVE unless tied to stable orders and supply normalization, which is POSITIVE.
  * Backlog building or longer lead times are POSITIVE unless driven only by supply constraint without orders, which is neutral/negative.
- Pricing up is positive if realized without elastic volume loss; pricing down is negative unless offset by volume acceleration.
- Cost down is positive; cost up is negative.
- Inventory draw toward normal is positive; rebuild or swollen is negative.

HEDGE WORDS AND BIAS ADJUSTMENT:
Sell-side and management skew positive. Negative admissions from them are high-signal. Modest positives from bearish sources are high-signal.

PRICED-IN RULE:
If a headwind/tailwind is called de-risked, treat it as neutral unless there is new delta vs that baseline.

TEMPORAL AWARENESS:
Keep focus on the next 6-12 months. Backward-looking items matter only if they change the forward path for FQ1 and FQ2.

SEGMENT DOMINANCE:
Use the exposure_weights dict per ticker (canonical end-market labels mapped to percent of business). Drive the score from the largest or strategic segment. Smaller segments can moderate but should not flip the score unless the rate-of-change is unusually strong and near term. When using an item from a small segment, make that fact explicit in a driver.

SPECULATIVE CATALYSTS:
Discount heavily unless timing and probability are high and near term.

EVIDENCE DEPTH AND CONFIDENCE:
Use signal_quality fields (unique_dates and single_source_dominance). If unique_dates <= 1 or single_source_dominance is true, cap confidence at low or medium and limit magnitude to the -3..+3 range.

OUTPUT per ticker (schema provided separately):
- story_score: float [-5..5], cross‑sectional and relative
- drivers: 3–5 concise mechanism bullets tied to forward numbers
- dominant_mechanisms: up to 3 tags
- inversion_flag: {none, minor, major}
- confidence: {low, medium, high}
- throughline: one‑sentence thesis that a PM could repeat
- why_now: ≤ 12 words on what changed and why it matters now
- watch_next: ≤ 12 words on the next proof point to confirm/negate the thesis
- score_reason: ≤ 12 words explaining the score direction (rate‑of‑change first)
LENS RULE: Expectation Tilt sets the "bar" only. It does not override signals. Surface contradictions (e.g., expensive + emergent negatives; cheap + emergent positives).
QUALITATIVE ACCEPTED: treat "ahead of plan / below plan / tracking ahead / tracking behind / signs of bottoming / fading momentum" as decisive even without numbers, if sourced well and tied to FQ1/FQ2.

ASCII only. Return a top-level JSON object per the schema.
"""

STORYSCORE_WRITE_PROMPT = """
You are a seasoned buy-side Industrials & Materials analyst. Current date is provided in the input.
Task: Using the compute-stage fields and the stock-window digest, write ~150 words of STOCK context per ticker focused on the current narrative of what will drive the share price for the next 6-12 months.

WRITE RULES:
- FIRST SENTENCE RULE: Start with the thesis anchor. Then add one nearer-term (roughly next 1-2 quarters) mechanism as corroborating evidence (orders/backlog/lead times/pricing/inventory/margins). Do not use generic openers like saying "over the next 1-2 quarters".  
    - BANNED OPENERS (reject these exact strings or any trivial variants): "Pricing and orders are the swing factors"; "Orders and pricing are the swing factors"; "Demand and pricing are the swing factors"; "Orders are the swing factor".
- FLUFF FILTER: Avoid abstract terms like "digital order trends" unless explicitly tied to measured order volume, pricing realization, lead times, or backlog change, with a next-quarter implication.
- Apply the cyclical rule: demand momentum outranks margin execution.
- Be comp-aware: if a prior quarter had unusual mega-orders, call out the tough comp and focus on sequential B:B, backlog conversion, and absolute backlog change.
- Apply the pricing-durability guardrail: if lead times are easing and new capacity is coming on, note the risk to pricing and temper enthusiasm.
- Treat multi-ticker end-market notes as context unless unusually probative for the ticker’s primary channel.
- If evidence is thin (unique_dates <= 1 or single_source_dominance true), say so briefly and keep it tight.
- Use 'throughline', 'why_now', and 'watch_next' from the compute stage if provided.
- Ban macro wallpaper: no generic industry comments unless it credibly ties to a shift in company expectations in the near-term ; if used, tie it explicitly to a company’s fundamental metrics (volume/pricing, margins, backlog, etc.).
- No lists or enumerations. One coherent paragraph only.
- Avoid jargon. No valuation talk. ASCII only.

Return JSON per schema with a single field stock_paragraph per ticker.
"""

# ===========================
# SOFTDATA (two-stage, bullets)
# ===========================

SOFTDATA_COMPUTE_PROMPT = """
You are a seasoned buy-side Industrials & Materials analyst. Current date is provided in the input.
Goal: For the next 1-3 months, compute SoftData in {-1, 0, +1} per ticker and select incremental datapoints.

SELECTION RULES (rate of change first, FQ1/FQ2 only):
- Keep items that change the next-quarter path: acceleration/deceleration in orders, backlog growth/shrink, lead times, realized pricing vs plan, inventory draw/build, tracking vs plan, visibility.
    - LANGUAGE EXAMPLES (non‑exhaustive; DO NOT hard‑match): "tracking ahead of plan", "above plan", "below plan", "behind target", "signs of bottoming", "momentum fading", "stabilizing", "deteriorating".
- Company-specific anchors the call; end-market-only items are context unless unusually strong for the company’s primary channel and clearly aligned with exposure_weights.
- Use importance_weight and num_tickers_in_note when present. Otherwise proxy via specificity, recency, kind rank (channel > key_comment > endmarket), and ticker count de-rating.
- Use exposure_weights together with item labels to judge alignment. Do not discard small segments outright. If a small segment signal is fresh and relevant, include it but reflect lower materiality in tone.
- Exclude housekeeping or long-dated items without near-term implication.
- Direction semantics follow the backlog/lead time rules from StoryScore.
- Expectation Tilt is tie-break only.
- If the primary segment (by exposure_weights) shows a clear directional change and the opposite signals are (a) in smaller segments or (b) older, choose ±1. Only set conflict=true and softdata=0 when equally strong, equally recent signals disagree *within the primary segment*.
- Accept decisive qualitative signals when numbers are not available, but tie them to a next-quarter implication.


SEGMENT DOMINANCE:
Use the exposure_weights dict per ticker (canonical end-market labels mapped to percent of business). Drive the score from the largest or strategic segment. Smaller segments can moderate but should not flip the score unless the rate-of-change is unusually strong and near term. When using an item from a small segment, make that fact explicit in a driver.

OUTPUT per ticker:
- softdata: -1, 0, or +1
- flow_bullets: target 6-8 compact candidates when evidence exists, else 3-5
- dominant_mechanisms: up to 3 from {orders, backlog, lead_times, pricing, mix, inventory, capacity, cost, logistics}
- confidence: {low, medium, high}
- conflict: boolean

OUTPUT per ticker (additionally):
- score_reason: ≤ 12 words capturing the key rate‑of‑change driver for the +1/0/‑1.

PROVENANCE RULE:
- For every entry in flow_bullets, end the sentence with a provenance tag in square brackets using the provided flow_items fields, exactly:
    - [title_short, YYYY-MM-DD].
- If multiple sources inform one bullet, append tags separated by semicolons. Use provided fields verbatim; do not invent titles or dates.

ASCII only. Return a top-level JSON object per the schema.
"""

SOFTDATA_WRITE_PROMPT = """
You are a seasoned buy-side Industrials & Materials analyst. Current date is provided in the input.
Task: Finalize the incremental datapoint bullets for the 1-3 months using the compute-stage fields.

WRITE RULES:
- Aim for 4-6 bullets per ticker when company-specific evidence exists; use no fewer than 3 unless there is truly no incremental data.
- Preserve existing provenance tags when present. If any bullet lacks a tag, append [title_short, YYYY-MM-DD] using the provided flow_items only; do not invent.
- Lead with the mechanism and the rate of change verb. The rate of change may be zero (e.g. "remains weak" or "remains strong") if there is truly no acceleration or deceleration occuring.
- Include an explicit near‑term estimate implication in plain terms (e.g., "Q4 revenue biased higher", "margin improvement less likely").

- Each bullet MUST:
  - begin with the mechanism word (Orders, Pricing, Lead times, Backlog, Inventory, Cost, Capacity, Logistics, Mix) without a colon,
  - include the indicator and number when available, otherwise a crisp qualitative signal (e.g., "spot prices slipping", "destocking resumes"),
  - include scope when present (region, channel, sub-segment),
  - state a next-quarter implication in operational terms,
  - end with source tags [title_short, YYYY-MM-DD] using the payload’s tags only.
- If company-specific evidence is fewer than 6 bullets, supplement with high-specificity end-market items aligned to the company’s primary exposure, and make the alignment obvious in the sentence.
- If evidence is conflicting, include both sides in one bullet when possible.
- If there is truly no incremental evidence in the flow window, return exactly one bullet: "No incremental data".
- No valuation. ASCII only.

Return JSON per schema with flow per ticker.
"""

# ===========================
# END-MARKET (two-stage)
# ===========================

ENDMARKET_STOCK_PROMPT = """
Summarize the stock story for the single end-market label provided, using the stock-window items.

Output:
- main_point: one sentence with the key mechanism and direction of change, timeline-aware if applicable
- supporting_points: 2-4 concise bullets that support the main_point with mechanisms and implications

Rules:
- Treat this as one label at a time. Do not include any label field in your response.
- Parent labels should roll up submarkets; some repetition is acceptable.
- Be timeline-aware across the window; prefer fresher signals, but keep older signals if they remain relevant to the next quarter without inventing facts.
- Apply the hierarchy and bias rules lightly; keep the focus on mechanisms and rate of change.
- No valuation. ASCII only.

Return JSON per schema with an array "items", where the first object contains main_point and supporting_points.
"""

ENDMARKET_FLOW_PROMPT = """
Summarize incremental flow datapoints for the single end-market label provided, using the flow-window items.

Write 1-3 compact bullets focused on mechanisms and why they matter for the next quarter.
Each bullet MUST end with one or more source tags: [title_short, YYYY-MM-DD]. If multiple sources inform a bullet, append tags separated by semicolons.
Do not include any label field in your response. No valuation. ASCII only.

Return JSON per schema with an array "items", where the first object contains bullets.
"""

# ===========================
# EXECUTIVE SUMMARY
# ===========================

EXEC_SUMMARY_PROMPT = """
Write the Executive Summary from the flagged tickers and their recent flow context.

Produce 8 top items. For each item output:
- main_point: start with the subject prefix "<TICKER>" and, if a parent end-market label is provided, include it in parentheses as "<TICKER> (<PARENT>)", then one sentence on what changed and why it matters in the next 1-4 quarters.
  Prefer Level-2 volume/demand inflections, novelty vs the prevailing narrative, and single-ticker channel/survey evidence.
- supporting_points: 1-3 bullets adding brief color on mechanisms and what to watch next.
- flow: 1-3 bullets of incremental datapoints; each MUST end with provenance tags in square brackets: [title_short, YYYY-MM-DD]. Join multiple sources with semicolons.

No duplication across items. No valuation. ASCII only.

Output strictly as JSON:
{"items": [{"main_point": "...", "supporting_points": ["..."], "flow": ["..."]}, ...]}
Only JSON, no extra text.
"""

# ===========================
# JSON SCHEMAS
# ===========================

SCHEMA_STORYSCORE_COMPUTE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ticker": {"type": "string"},
                    "story_score": {"type": "number"},
                    "drivers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 5
                    },
                    "dominant_mechanisms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3
                    },
                    "inversion_flag": {
                        "type": "string",
                        "enum": ["none", "minor", "major"]
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    },
                    "throughline": {"type": "string"},
                    "why_now": {"type": "string"},
                    "watch_next": {"type": "string"},
                    "score_reason": {"type": "string", "maxLength": 80}
                },
                "required": [
                    "ticker",
                    "story_score",
                    "drivers",
                    "dominant_mechanisms",
                    "inversion_flag",
                    "confidence",
                    "throughline",
                    "why_now",
                    "watch_next",
                    "score_reason"
                ]
            }
        }
    },
    "required": ["items"]
}

SCHEMA_STORYSCORE_WRITE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ticker": {"type": "string"},
                    "stock_paragraph": {"type": "string"}
                },
                "required": ["ticker", "stock_paragraph"]
            }
        }
    },
    "required": ["items"]
}

SCHEMA_SOFTDATA_COMPUTE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ticker": {"type": "string"},
                    "softdata": {"type": "integer", "enum": [-1, 0, 1]},
                    "flow_bullets": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
                    "dominant_mechanisms": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "conflict": {"type": "boolean"},
                    "score_reason": {"type": "string", "maxLength": 80}
                },
                "required": ["ticker", "softdata", "flow_bullets", "dominant_mechanisms", "confidence", "conflict", "score_reason"]
            }
        }
    },
    "required": ["items"]
}

SCHEMA_SOFTDATA_WRITE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ticker": {"type": "string"},
                    "flow": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["ticker", "flow"]
            }
        }
    },
    "required": ["items"]
}

SCHEMA_ENDMARKET_STOCK = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "main_point": {"type": "string"},
                    "supporting_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 5
                    }
                },
                "required": ["main_point", "supporting_points"]
            }
        }
    },
    "required": ["items"]
}

SCHEMA_ENDMARKET_FLOW = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "bullets": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["bullets"]
            }
        }
    },
    "required": ["items"]
}

SCHEMA_EXEC_SUMMARY = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "main_point": {"type": "string"},
                    "supporting_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 5
                    },
                    "flow": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": 5
                    }
                },
                "required": ["main_point", "supporting_points", "flow"]
            }
        }
    },
    "required": ["items"]
}
