import argparse
import csv
import json
import os
import random
import string
from datetime import datetime, timezone


DEFAULT_INPUT_DIR = os.path.join("data", "complete_overall")
DEFAULT_OUTPUT_DIR = os.path.join("data", "public")


SOURCE_FILES = {
    "surveys_public.csv": "2_surveys_results.csv",
    "features_public.csv": "5_overall_features.csv",
    "stroop_trials_public.csv": "4a_stroop_trials.csv",
    "wcst_trials_public.csv": "4b_wcst_trials.csv",
}


REMOVE_TOKENS = ["timestamp"]
ID_COLUMNS = {"participantid", "participant_id"}


def normalize(col: str) -> str:
    return col.lstrip("\ufeff").strip()


def random_id(n=12):
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def build_id_map(paths):
    ids = set()
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            norm_header = [normalize(h) for h in header]
            id_idx = [i for i, h in enumerate(norm_header) if h.lower() in ID_COLUMNS]
            if not id_idx:
                raise ValueError(f"No participant ID column in {path}")
            idx = id_idx[0]
            for row in reader:
                if len(row) <= idx:
                    continue
                ids.add(row[idx])
    id_map = {}
    for pid in sorted(ids):
        rid = random_id()
        while rid in id_map.values():
            rid = random_id()
        id_map[pid] = rid
    return id_map


def filter_header(header):
    norm = [normalize(h) for h in header]
    drop_idx = []
    kept = []
    for i, col in enumerate(norm):
        low = col.lower()
        if low in ID_COLUMNS:
            drop_idx.append(i)
            continue
        if any(tok in low for tok in REMOVE_TOKENS):
            drop_idx.append(i)
            continue
        kept.append((i, col))
    return drop_idx, kept


def write_public_file(src_path, dst_path, id_map):
    with open(src_path, "r", newline="", encoding="utf-8-sig") as fin, open(
        dst_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        norm_header = [normalize(h) for h in header]

        id_idx = [i for i, h in enumerate(norm_header) if h.lower() in ID_COLUMNS]
        if not id_idx:
            raise ValueError(f"No participant ID column in {src_path}")
        id_idx = id_idx[0]

        _, kept = filter_header(header)
        new_header = ["public_id"] + [col for _, col in kept]
        writer.writerow(new_header)

        row_count = 0
        for row in reader:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            pid = row[id_idx]
            public_id = id_map.get(pid)
            if public_id is None:
                continue
            new_row = [public_id] + [row[i] for i, _ in kept]
            writer.writerow(new_row)
            row_count += 1
    return row_count, len(new_header)


def main():
    parser = argparse.ArgumentParser(description="Create de-identified public data files.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    src_paths = []
    for dst_name, src_name in SOURCE_FILES.items():
        src_path = os.path.join(args.input_dir, src_name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        src_paths.append(src_path)

    id_map = build_id_map(src_paths)

    metadata = {
        "created_on": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dir": args.input_dir,
        "id_mapping": "random, not stored",
        "removed_columns": ["participantId", "participant_id", "timestamp"],
        "files": {},
    }

    for dst_name, src_name in SOURCE_FILES.items():
        src_path = os.path.join(args.input_dir, src_name)
        dst_path = os.path.join(args.output_dir, dst_name)
        rows, cols = write_public_file(src_path, dst_path, id_map)
        metadata["files"][dst_name] = {
            "source": src_name,
            "rows": rows,
            "cols": cols,
        }

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Public data written to: {args.output_dir}")


if __name__ == "__main__":
    main()
