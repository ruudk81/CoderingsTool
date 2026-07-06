"""Concatenate a set of open-answer variables in an SPSS (.sav) file into one
new text variable, separator ", " by default.

Typical use: a "multiple response" open question stores answers across numbered
slots (xQd1_1 .. xQd1_10). This merges the filled slots per respondent into a
single variable (e.g. Qd1), skipping empty slots so there are no stray commas.

The original file is never modified: a new .sav is written and its result is
verified against a fresh recomputation before the script reports success.

Examples
--------
Select by prefix (grabs xQd1_1 .. xQd1_10, numeric order):
    python concatenate/concat_open_ends.py \
        --infile "data/M000000 Associatiemonitor Merk X tabellenbestand vergelijkend.sav" \
        --prefix xQd1_ --newvar Qd1

Select an explicit list of variables:
    python concatenate/concat_open_ends.py \
        --infile "data/some file.sav" \
        --vars xQd1_1,xQd1_2,xQd1_3 --newvar Qd1

Custom separator and output path:
    python concatenate/concat_open_ends.py --infile "..." --prefix xQd1_ \
        --newvar Qd1 --sep " | " --outfile "data/out.sav"
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pyreadstat


def select_columns(all_cols, prefix=None, vars_list=None):
    """Return the source columns, in the intended order.

    - prefix: matches columns '<prefix><number>' and sorts them numerically,
      so xQd1_2 comes before xQd1_10 (string sort would get this wrong).
    - vars_list: an explicit, order-preserving list.
    """
    if vars_list:
        missing = [v for v in vars_list if v not in all_cols]
        if missing:
            sys.exit(f"ERROR: variables not found in file: {missing}")
        return vars_list

    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matched = [(int(m.group(1)), c) for c in all_cols if (m := pat.match(c))]
    if not matched:
        sys.exit(f"ERROR: no columns match prefix '{prefix}' followed by a number.")
    matched.sort(key=lambda t: t[0])
    return [c for _, c in matched]


def combine_row(values, sep):
    parts = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != ""]
    return sep.join(parts) if parts else pd.NA


def default_outfile(infile, newvar):
    p = Path(infile)
    return str(p.with_name(f"{p.stem} met {newvar}{p.suffix}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infile", required=True, help="Path to the source .sav file")
    ap.add_argument("--newvar", required=True, help="Name of the new variable, e.g. Qd1")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prefix", help="Select all columns '<prefix><number>', e.g. xQd1_")
    grp.add_argument("--vars", help="Explicit comma-separated list of source variables")
    ap.add_argument("--sep", default=", ", help="Separator between answers (default: ', ')")
    ap.add_argument("--outfile", help="Output .sav path (default: '<infile> met <newvar>.sav')")
    ap.add_argument("--label", help="Variable label for the new variable")
    args = ap.parse_args()

    vars_list = [v.strip() for v in args.vars.split(",")] if args.vars else None
    outfile = args.outfile or default_outfile(args.infile, args.newvar)

    df, meta = pyreadstat.read_sav(args.infile)
    if args.newvar in meta.column_names:
        sys.exit(f"ERROR: variable '{args.newvar}' already exists in the file.")

    cols = select_columns(meta.column_names, prefix=args.prefix, vars_list=vars_list)
    print(f"Source variables ({len(cols)}): {cols}")

    df[args.newvar] = df[cols].apply(lambda r: combine_row(r, args.sep), axis=1)

    label = args.label or f"{args.newvar} (samengevoegd uit {cols[0]} t/m {cols[-1]})"
    column_labels = [meta.column_names_to_labels.get(c, "") or "" for c in meta.column_names]
    column_labels.append(label)

    pyreadstat.write_sav(
        df, outfile,
        column_labels=column_labels,
        variable_value_labels=meta.variable_value_labels,
        variable_measure=meta.variable_measure,
    )

    # --- Verify round-trip against a fresh recomputation ---
    chk, cmeta = pyreadstat.read_sav(outfile)
    expected = df[cols].apply(lambda r: combine_row(r, args.sep), axis=1).fillna("").astype(str)
    got = chk[args.newvar].fillna("").astype(str)
    if not (expected.values == got.values).all():
        sys.exit("ERROR: verification failed — written values differ from expected.")

    n_filled = int((got != "").sum())
    print(f"OK  Written: {outfile}")
    print(f"    Rows: {len(chk)} | '{args.newvar}' filled: {n_filled} | blank: {len(chk) - n_filled}")
    print(f"    Label: {cmeta.column_names_to_labels.get(args.newvar)}")
    print("    Examples:")
    for v in got[got.str.contains(re.escape(args.sep))].head(2):
        print(f"      {v!r}")


if __name__ == "__main__":
    main()