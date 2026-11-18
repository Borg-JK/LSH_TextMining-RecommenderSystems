"""Loading the TV data set and cleaning product titles."""
import json
import re
from dataclasses import dataclass


@dataclass
class Product:
    idx: int
    shop: str
    title: str
    features: dict
    model_id: str      # gold label: only for training and evaluation


def load_products(path: str) -> list[Product]:
    """All listings as one list sorted by shop, title and URL (the grouping by
    modelID is discarded). The URL is not kept, as it contains the model ID."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = [p for group in raw.values() for p in group]
    items.sort(key=lambda p: (p["shop"], p["title"], p["url"]))
    return [Product(i, p["shop"], p["title"], p["featuresMap"], p["modelID"])
            for i, p in enumerate(items)]


_INCH = re.compile(r"(\d)\s*-?\s*(?:inches|inch|\"|”|'')")
_HZ = re.compile(r"(\d)\s*-?\s*(?:hertz|hz)")
_NOT_TOKEN = re.compile(r"[^a-z0-9.]+")
_STRAY_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")


def clean_title(title: str, shop: str) -> list[str]:
    """Title tokens (Sect. 3.1): lowercase, inch and hertz units unified,
    punctuation removed except decimal points, tokens of >= 3 characters that
    are part of the shop's name dropped. Title order, without repeats."""
    t = title.lower()
    t = _INCH.sub(r"\1inch ", t)       # 46"  46-Inch  46 inches -> 46inch
    t = _HZ.sub(r"\1hz ", t)           # 120 Hz  120hertz        -> 120hz
    t = _NOT_TOKEN.sub(" ", t)
    t = _STRAY_DOT.sub(" ", t)         # keep dots only inside numbers (45.9)
    shop_name = re.sub(r"[^a-z0-9]", "", shop.lower())   # bestbuy.com -> bestbuycom
    seen, out = set(), []
    for tok in t.split():
        if len(tok) >= 3 and tok in shop_name:           # 'best', 'buy', 'newegg', 'com'
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out
