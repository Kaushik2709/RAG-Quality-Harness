from typing import Dict, Tuple

# Pricing per 1,000,000 tokens: (input_price_usd, output_price_usd)
PRICING_TABLE_PER_1M: Dict[str, Tuple[float, float]] = {
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
}

def compute_cost(model_name: str, tokens_in: int, tokens_out: int) -> float:
    key = model_name.lower()
    pricing = None
    for model_key, p in PRICING_TABLE_PER_1M.items():
        if model_key in key:
            pricing = p
            break

    if pricing is None:
        return 0.0

    in_cost = (tokens_in / 1_000_000.0) * pricing[0]
    out_cost = (tokens_out / 1_000_000.0) * pricing[1]
    return float(in_cost + out_cost)
