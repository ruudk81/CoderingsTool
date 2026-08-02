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
        --infile "data/M000000 Merkonderzoek tabellenbestand.sav" \
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
            raise ValueError(f"variables not found in file: {missing}")
        return vars_list

    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matched = [(int(m.group(1)), c) for c in all_cols if (m := pat.match(c))]
    if not matched:
        raise ValueError(f"no columns match prefix '{prefix}' followed by a number.")
    matched.sort(key=lambda t: t[0])
    return [c for _, c in matched]


def combine_row(values, sep):
    parts = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != ""]
    return sep.join(parts) if parts else pd.NA


def default_outfile(infile, newvar):
    p = Path(infile)
    return str(p.with_name(f"{p.stem} met {newvar}{p.suffix}"))


def concat_variables(infile, newvar, prefix=None, vars_list=None, sep=", ",
                     outfile=None, label=None):
    """Write a new .sav with `newvar` = the concatenated source variables.

    Importable core (the Streamlit app uses this; the CLI below wraps it).
    Raises ValueError on bad input, RuntimeError when the written file does not
    verify against a fresh recomputation. Returns
    {"outfile", "columns", "rows", "filled", "label"}.
    """
    outfile = outfile or default_outfile(infile, newvar)

    df, meta = pyreadstat.read_sav(infile)
    if newvar in meta.column_names:
        raise ValueError(f"variable '{newvar}' already exists in the file.")

    cols = select_columns(meta.column_names, prefix=prefix, vars_list=vars_list)
    df[newvar] = df[cols].apply(lambda r: combine_row(r, sep), axis=1)

    label = label or f"{newvar} (samengevoegd uit {cols[0]} t/m {cols[-1]})"
    column_labels = [meta.column_names_to_labels.get(c, "") or "" for c in meta.column_names]
    column_labels.append(label)

    pyreadstat.write_sav(
        df, outfile,
        column_labels=column_labels,
        variable_value_labels=meta.variable_value_labels,
        variable_measure=meta.variable_measure,
    )

    # --- Verify round-trip against a fresh recomputation ---
    chk, _ = pyreadstat.read_sav(outfile)
    expected = df[cols].apply(lambda r: combine_row(r, sep), axis=1).fillna("").astype(str)
    got = chk[newvar].fillna("").astype(str)
    if not (expected.values == got.values).all():
        raise RuntimeError("verification failed — written values differ from expected.")

    return {"outfile": outfile, "columns": cols, "rows": len(chk),
            "filled": int((got != "").sum()), "label": label,
            "examples": got[got.str.contains(re.escape(sep))].head(2).tolist()}


def find_slot_groups(columns, min_size=2):
    """Detect numbered slot groups among `columns`: {prefix: [cols, numeric order]}.

    'xQd1_1', 'xQd1_2', … 'xQd1_10' → {'xQd1_': ['xQd1_1', …, 'xQd1_10']}.
    """
    pat = re.compile(r"^(.*?)(\d+)$")
    groups = {}
    for c in columns:
        if m := pat.match(c):
            groups.setdefault(m.group(1), []).append((int(m.group(2)), c))
    return {p: [c for _, c in sorted(members)]
            for p, members in groups.items() if len(members) >= min_size}


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
    try:
        res = concat_variables(args.infile, args.newvar, prefix=args.prefix,
                               vars_list=vars_list, sep=args.sep,
                               outfile=args.outfile, label=args.label)
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")

    print(f"Source variables ({len(res['columns'])}): {res['columns']}")
    print(f"OK  Written: {res['outfile']}")
    print(f"    Rows: {res['rows']} | '{args.newvar}' filled: {res['filled']} "
          f"| blank: {res['rows'] - res['filled']}")
    print(f"    Label: {res['label']}")
    print("    Examples:")
    for v in res["examples"]:
        print(f"      {v!r}")


if __name__ == "__main__":
    main()