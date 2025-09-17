import argparse
import ast
import json
import os
import re
from typing import List, Tuple, Optional

import pandas as pd


# -----------------------
# Utils
# -----------------------
def parse_listlike(x) -> List[str]:
    """
    Parse RecipeNLG list-like fields safely.
    Accepts real Python lists, JSON strings, or Python-literal strings.
    Fallback: split on commas/semicolons.
    """
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]

    s = str(x).strip()
    if not s:
        return []

    # Python literal (common in RecipeNLG dumps)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(t).strip() for t in obj if str(t).strip()]
    except Exception:
        pass

    # JSON list
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(t).strip() for t in obj if str(t).strip()]
    except Exception:
        pass

    # Fallback
    parts = re.split(r"[;,]\s*|\s{2,}", s)
    return [p.strip() for p in parts if p.strip()]


def normalize_ingredients(ings: List[str]) -> List[str]:
    out = []
    for it in ings:
        t = re.sub(r"\s+", " ", str(it)).strip()
        # drop leading bullets/numbers like "1.", "•", "-" etc.
        t = re.sub(r"^\s*(?:[-•*]|\d+[\).])\s*", "", t)
        if t:
            out.append(t)
    return out


def normalize_steps(steps: List[str], max_steps: Optional[int] = None) -> List[str]:
    out = []
    for s in steps:
        t = re.sub(r"\s+", " ", str(s)).strip()
        # remove leading numbering if present
        t = re.sub(r"^\s*(?:[-•*]|\d+[\).])\s*", "", t)
        if t:
            out.append(t)
    if max_steps is not None and max_steps > 0:
        out = out[:max_steps]
    return out


def format_ingredients_target(ings: List[str]) -> str:
    """
    Keep the list-string style your trainer expects, e.g.:
    "['milk', 'eggs', 'butter']"
    """
    return "[" + ", ".join(f"'{i}'" for i in ings) + "]"


def format_steps_target(steps: List[str]) -> str:
    """
    Human-readable numbered lines (seq2seq-friendly).
    1) Step one
    2) Step two
    """
    lines = [f"{i+1}) {s}" for i, s in enumerate(steps)]
    return "\n".join(lines)


def build_rows_recipenlg(
    df: pd.DataFrame,
    mode: str,
    max_steps: Optional[int],
    use_task_tags: bool,
    prefer_ner: bool,
) -> List[Tuple[str, str]]:
    """
    Build (input, target) rows from RecipeNLG-style dataframe:
      title, ingredients, directions, (optional) NER

    mode: 'ingredients' | 'steps' | 'both'
    """
    rows: List[Tuple[str, str]] = []
    have_ner = "NER" in df.columns

    for _, r in df.iterrows():
        title = str(r.get("title", "")).strip()
        if not title:
            continue

        # Choose ingredients: NER (cleaner) optionally preferred, else ingredients list
        ings_raw = []
        if prefer_ner and have_ner and pd.notna(r["NER"]):
            ings_raw = parse_listlike(r["NER"])
        elif "ingredients" in df.columns and pd.notna(r.get("ingredients", "")):
            ings_raw = parse_listlike(r["ingredients"])
        ings = normalize_ingredients(ings_raw)

        # Steps/directions
        steps = []
        if "directions" in df.columns and pd.notna(r.get("directions", "")):
            steps = normalize_steps(parse_listlike(r["directions"]), max_steps=max_steps)

        if mode in ("ingredients", "both") and ings:
            inp = f"<INGR> {title}" if use_task_tags else title
            tgt = format_ingredients_target(ings)
            rows.append((inp, tgt))

        if mode in ("steps", "both") and steps:
            inp = f"<STEPS> {title}" if use_task_tags else title
            tgt = format_steps_target(steps)
            rows.append((inp, tgt))

    return rows


# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True, help="Path to RecipeNLG CSV (title, ingredients, directions[, NER])")
    ap.add_argument("--out_dir", required=True, help="Where to write processed CSV(s)")
    ap.add_argument("--mode", choices=["ingredients", "steps", "both"], default="both",
                    help="Which targets to create")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = unlimited; else cap number of steps per recipe")
    ap.add_argument("--use_task_tags", action="store_true",
                    help="Prefix titles with <INGR>/<STEPS> so one model can learn both")
    ap.add_argument("--prefer_ner", action="store_true",
                    help="If present, use NER column for ingredients (often cleaner)")
    ap.add_argument("--outfile", default="", help="Optional custom filename; otherwise auto-named")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.raw_csv)

    max_steps = None if args.max_steps <= 0 else args.max_steps
    rows = build_rows_recipenlg(
        df=df,
        mode=args.mode,
        max_steps=max_steps,
        use_task_tags=args.use_task_tags,
        prefer_ner=args.prefer_ner,
    )

    out_df = pd.DataFrame(rows, columns=["input", "target"])

    # name output
    if args.outfile:
        out_path = os.path.join(args.out_dir, args.outfile)
    else:
        tag = "multitask" if args.use_task_tags and args.mode == "both" else args.mode
        ner_tag = "_ner" if args.prefer_ner and args.mode in ("ingredients", "both") else ""
        out_path = os.path.join(args.out_dir, f"processed_recipes_{tag}{ner_tag}.csv")

    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df):,} rows → {out_path}")


if __name__ == "__main__":
    main()
