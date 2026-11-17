# -*- coding: utf-8 -*-
"""Binary protocol metrics (HUMAN vs AI-involved) from exported doc_logits.

Paper protocol (two steps, exactly as described in the paper):
  Step 1  Select the operating threshold on the VALIDATION split:
            python binary_metrics.py --eval <eval_val_dir> --data <val.jsonl.gz> \
                --split-name validation --select-threshold --out threshold_val.json
          -> writes {"threshold": 0.6847, "split": "validation", "criterion": "bacc"}
  Step 2  Apply that frozen threshold to the TEST split:
            python binary_metrics.py --eval <eval_test_dir> --data <test.jsonl.gz> \
                --split-name test --threshold-file threshold_val.json
          -> reproduces AUROC 0.9570 / BACC 0.9113 / F1(AI) 0.9102 / Spec 0.9590
             on the full test split (n=202,762), at the validation-selected
             operating point (threshold 0.6847), CPU-only.

Safety guard: --select-threshold refuses to run when --split-name is "test";
threshold selection on test data is not part of the paper protocol.

Diagnostic mode (optional): omit --threshold-file and omit --select-threshold
to sweep the operating curve of the given split. The argmax point is then
labeled "split-internal-diagnostic" and is NOT a paper number.

Prediction rule: a document is predicted AI-involved iff its score is strictly
greater than the threshold (score = 1 - P(HUMAN), from softmax over the 3
document logits).
"""
import argparse, gzip, json, sys
import numpy as np


def load_scores(eval_dir, data_gz):
    gold = {}
    with gzip.open(data_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold[r["row_id_v2"]] = 0 if r["label"] == "HUMAN" else 1
    scores, labels = [], []
    mms = {r: np.memmap(f"{eval_dir}/rank{r}.f32", dtype="<f4", mode="r") for r in range(4)}
    for r in range(4):
        with open(f"{eval_dir}/rank{r}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sl = rec["arrays"]["doc_logits"]
                z = np.asarray(mms[r][sl["offset"]:sl["offset"] + 3], dtype="<f4")
                p = np.exp(z - z.max()); p = p / p.sum()
                scores.append(1.0 - p[0])          # P(AI-involved)
                labels.append(gold[rec["doc_id"]])
    return np.asarray(scores), np.asarray(labels)


def auroc_mann_whitney(s, y):
    order = np.argsort(s, kind="mergesort")        # tie-averaged ranks
    sr = s[order]; ranks = np.empty(len(s)); i = 0
    P, N = int(y.sum()), len(y) - y.sum()
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def metrics_at_threshold(s, y, t):
    pred = s > t
    TP = int((pred & (y == 1)).sum()); FP = int((pred & (y == 0)).sum())
    FN = int((~pred & (y == 1)).sum()); TN = int((~pred & (y == 0)).sum())
    prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
    return dict(
        BACC=round(float((TP / max(1, TP + FN) + TN / max(1, TN + FP)) / 2), 4),
        F1_AI=round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 4),
        Spec=round(float(TN / max(1, TN + FP)), 4),
        threshold=round(float(t), 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="eval dir with rank{0..3}.jsonl + rank{0..3}.f32")
    ap.add_argument("--data", required=True, help="dataset jsonl.gz for the SAME split")
    ap.add_argument("--split-name", required=True, choices=["validation", "test"],
                    help="which split --eval/--data contain (labeled into all outputs)")
    ap.add_argument("--select-threshold", action="store_true",
                    help="select the BACC-max threshold on THIS split (validation only)")
    ap.add_argument("--out", default="threshold_val.json", help="output json for --select-threshold")
    ap.add_argument("--threshold-file", default=None,
                    help="apply the frozen threshold stored in this json (from step 1)")
    ap.add_argument("--threshold", type=float, default=None, help="apply a numeric threshold directly")
    args = ap.parse_args()

    if args.select_threshold and args.split_name != "validation":
        sys.exit("REFUSED: threshold selection must run on the validation split; "
                 "got --split-name %s. This guard enforces the paper protocol." % args.split_name)

    s, y = load_scores(args.eval, args.data)
    P, N = int(y.sum()), len(y) - y.sum()
    auroc = auroc_mann_whitney(s, y)

    if args.select_threshold:
        order = np.argsort(s, kind="mergesort")[::-1]
        tps = np.cumsum(y[order]); fps = np.cumsum(1 - y[order])
        bacc = tps / P + (1 - fps / N)
        k = int(np.argmax(bacc))
        t = float(s[order[k]])
        json.dump(dict(threshold=round(t, 6), split=args.split_name,
                       criterion="max balanced accuracy"), open(args.out, "w"), indent=2)
        print(json.dumps(dict(threshold_selection=dict(
            threshold=round(t, 6), split=args.split_name, criterion="bacc",
            note="frozen before test evaluation; apply with --threshold-file")),
            indent=2))
        return

    if args.threshold_file or args.threshold is not None:
        if args.threshold is not None:
            t = args.threshold; source = "cli"
        else:
            t = json.load(open(args.threshold_file))["threshold"]; source = args.threshold_file
        out = dict(n=len(y), split=args.split_name, AUROC=round(float(auroc), 4),
                   threshold_source=source,
                   protocol="threshold selected on validation, applied frozen")
        out.update(metrics_at_threshold(s, y, float(t)))
        print(json.dumps(out, indent=2))
        return

    # Diagnostic sweep (no paper number): label the argmax as split-internal.
    order = np.argsort(s, kind="mergesort")[::-1]
    tps = np.cumsum(y[order]); fps = np.cumsum(1 - y[order])
    bacc = tps / P + (1 - fps / N)
    k = int(np.argmax(bacc))
    out = dict(n=len(y), split=args.split_name, AUROC=round(float(auroc), 4),
               threshold_source="split-internal-diagnostic (NOT a paper number)",
               protocol="run --select-threshold on validation, then --threshold-file for paper numbers")
    out.update(metrics_at_threshold(s, y, float(s[order[k]])))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
