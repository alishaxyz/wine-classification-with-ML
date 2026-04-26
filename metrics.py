from collections import Counter


def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true) if len(y_true) else 0.0

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    index = {lbl: i for i, lbl in enumerate(labels)}
    m = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    for yt, yp in zip(y_true, y_pred):
        if yt in index and yp in index:
            m[index[yt]][index[yp]] += 1
    return m, labels

def _safe_div(num, den):
    return num / den if den else 0.0

def precision_recall_f1_support(y_true, y_pred, labels=None):
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    k = len(labels)

    row_sum = [sum(cm[i][j] for j in range(k)) for i in range(k)]  # actual
    col_sum = [sum(cm[i][j] for i in range(k)) for j in range(k)]  # predicted

    per_class = {}
    for i, lbl in enumerate(labels):
        tp = cm[i][i]
        fp = col_sum[i] - tp
        fn = row_sum[i] - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = row_sum[i]
        per_class[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total = len(y_true)
    correct = sum(cm[i][i] for i in range(k))

    # micro averages (equivalent to accuracy for single-label multiclass)
    micro_precision = _safe_div(correct, total)
    micro_recall = _safe_div(correct, total)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    # macro averages
    macro_precision = sum(per_class[l]["precision"] for l in labels) / k if k else 0.0
    macro_recall = sum(per_class[l]["recall"] for l in labels) / k if k else 0.0
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / k if k else 0.0

    # weighted averages
    weighted_precision = _safe_div(sum(per_class[l]["precision"] * per_class[l]["support"] for l in labels), total)
    weighted_recall = _safe_div(sum(per_class[l]["recall"] * per_class[l]["support"] for l in labels), total)
    weighted_f1 = _safe_div(sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels), total)

    out = {
        "labels": labels,
        "per_class": per_class,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1, "support": total},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1, "support": total},
        "weighted": {"precision": weighted_precision, "recall": weighted_recall, "f1": weighted_f1, "support": total},
        "accuracy": _safe_div(correct, total),
    }
    return out

def balanced_accuracy(y_true, y_pred, labels=None):
    stats = precision_recall_f1_support(y_true, y_pred, labels)
    labels = stats["labels"]
    if not labels:
        return 0.0
    recalls = [stats["per_class"][l]["recall"] for l in labels]
    return sum(recalls) / len(recalls)

def classification_report(y_true, y_pred, labels=None, digits=4):
    stats = precision_recall_f1_support(y_true, y_pred, labels)
    labels = stats["labels"]

    def fmt(x):
        return f"{x:.{digits}f}"

    lines = []
    header = f"{'class':>10}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for lbl in labels:
        s = stats["per_class"][lbl]
        lines.append(
            f"{str(lbl):>10}  {fmt(s['precision']):>9}  {fmt(s['recall']):>9}  {fmt(s['f1']):>9}  {s['support']:>7}"
        )

    lines.append("-" * len(header))
    for avg_name in ["micro", "macro", "weighted"]:
        s = stats[avg_name]
        lines.append(
            f"{avg_name:>10}  {fmt(s['precision']):>9}  {fmt(s['recall']):>9}  {fmt(s['f1']):>9}  {s['support']:>7}"
        )
    lines.append(f"{'accuracy':>10}  {fmt(stats['accuracy']):>9}")
    lines.append(f"{'bal_acc':>10}  {fmt(balanced_accuracy(y_true, y_pred, labels)) :>9}")
    return "\n".join(lines)