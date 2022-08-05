#!/usr/bin/env python3

import argparse
import pandas as pd
import difflib
from jiwer import compute_measures


def string_diff(a, b):
    print(b)
    matcher = difflib.SequenceMatcher(None, a, b)

    def process_tag(tag, i1, i2, j1, j2):
        if tag == "replace":
            return "{" + matcher.a[i1:i2] + " -> " + matcher.b[j1:j2] + "}"
        if tag == "delete":
            return "{- " + matcher.a[i1:i2] + "}"
        if tag == "equal":
            return matcher.a[i1:i2]
        if tag == "insert":
            if matcher.b[j1:j2] == ' ':
                return ''
            else:
                return "{+ " + matcher.b[j1:j2] + "}"
        assert False, "Unknown tag %r" % tag

    return "".join(process_tag(*t) for t in matcher.get_opcodes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp")
    parser.add_argument("split_folder")
    parser.add_argument("--output", default=None)
    parser.add_argument("--diff", action="store_true", help="show diff only.")
    parser.add_argument("--verbose", action="store_true", help="show more columns.")
    args = parser.parse_args()

    if "test" in args.hyp:
        df = pd.read_csv(f"{args.split_folder}/test.csv", sep="|")
    elif "dev" in args.hyp:
        df = pd.read_csv(f"{args.split_folder}/val.csv", sep="|")
    else:
        msg = "Please choose a path that contains either test or dev."
        raise NotImplementedError(msg)

    del df["duration"]

    hyp = pd.read_csv(args.hyp, keep_default_na=False, names=["id", "hypothesis"])

    df = pd.merge(df, hyp, on="id", how="left")
    df = df.dropna()

    df["diff"] = df.apply(
        lambda row: string_diff(
            row["glosses"],
            row["hypothesis"],
        ),
        axis=1,
    )

    # filter out those only contain differences
    df = df[df["diff"].str.contains("{")]

    if args.diff:
        print(df["diff"].to_markdown(index=False))
        exit()

    if not args.verbose:
        for key in ["words", "raw_glosses"]:
            del df[key]

    # print(df.to_markdown(index=False))
    # print(df.to_markdown())

    if args.output:
        # df.to_csv(args.output, index=False)
        with open(args.output, 'w') as f:
            f.write('id\nwords\nglosses\nhypothesis\ndiff{-:deletions, +:insertions, ->:substitutions}\n\n')
            for index, row in df.iterrows():
                f.write(row['id']+'\n'+row['words']+'\n'+row['glosses']+'\n'+row['hypothesis']+'\n'+row['diff']+'\n\n')

    def print_dict_as_table(d):
        # print(pd.DataFrame([d]).to_markdown(index=False))
        print(pd.DataFrame([d]).to_markdown())

    measures = compute_measures(df["glosses"].tolist(), df["hypothesis"].tolist())
    print_dict_as_table(measures)
