import json
import pandas as pd
from collections import defaultdict, Counter

def describe_json(data, parent_key=""):
    summary = {}

    def recurse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                recurse(v, new_key)

        elif isinstance(obj, list):
            for item in obj:
                recurse(item, prefix)

        else:
            if prefix not in summary:
                summary[prefix] = {
                    "types": Counter(),
                    "count": 0,
                    "nulls": 0,
                    "examples": set()
                }

            entry = summary[prefix]
            entry["count"] += 1

            if obj is None:
                entry["nulls"] += 1
                entry["types"]["None"] += 1
            else:
                t = type(obj).__name__
                entry["types"][t] += 1

                if len(entry["examples"]) < 5:
                    entry["examples"].add(obj)

    recurse(data)

    # Pretty print
    for key, info in sorted(summary.items()):
        print(f"\nkey: {key}")
        print(f"  Count: {info['count']}")
        print(f"  Nulls: {info['nulls']}")
        print(f"  Types: {dict(info['types'])}")
        print(f"  Examples: {list(info['examples'])}")


def json_to_dataframe(data):
    """
    Converts nested JSON (list of dicts) into a flat pandas DataFrame.
    Nested keys are flattened using dot notation.
    """
    def flatten_dict(d, parent_key=""):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k

            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key))
            else:
                items[new_key] = v

        return items

    # Flatten each record
    flattened_data = [flatten_dict(entry) for entry in data]

    # Create DataFrame
    df = pd.DataFrame(flattened_data)

    return df

with open("/home/marcel/Repos/Data/kits23/kits23.json") as f:
    data = json.load(f)

df = json_to_dataframe(data)

df_selection = df[df['comorbidities.localized_solid_tumor'] == True]

print(df_selection['case_id'])

