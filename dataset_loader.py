def load_wintqt_csv(path):
    X = []
    y = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    header = lines[0].split(",")
    quality_idx = header.index("quality")

    for row in lines[1:]:
        parts = row.split(",")

        if len(parts) != len(header):
            continue

        features = []
        skip = False

        for i, val in enumerate(parts):
            col = header[i]
            if col == "quality" or col == "Id":
                continue
            if val.strip() == "":
                skip = True
                break
            features.append(float(val))

        if skip:
            continue

        quality = int(float(parts[quality_idx]))

        X.append(features)
        y.append(quality)

    return X, y


def load_wine_csv(path, has_type=False):
    X = []
    y = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    rows = lines[1:]

    for row in rows:
        parts = row.split(",")

        if has_type:
            if len(parts) < 13:
                continue
            features = parts[1:-1]
            quality_str = parts[-1]
        else:
            if len(parts) < 13:
                continue
            features = parts[:-2]
            quality_str = parts[-2]

        if any(v.strip() == "" for v in features):
            continue

        try:
            features = [float(v) for v in features]
            quality = int(float(quality_str))
        except ValueError:
            continue

        X.append(features)
        y.append(quality)

    return X, y
