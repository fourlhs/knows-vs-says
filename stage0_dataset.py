import json, os, urllib.request

URL = "https://rome.baulab.info/data/dsets/counterfact.json"


def load_counterfact(path="data/counterfact.json"):
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    return json.load(open(path))


def describe(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{pad}{k}: {type(v).__name__}")
            describe(v, indent + 1)
    elif isinstance(obj, list) and obj:
        print(f"{pad}[{len(obj)} x {type(obj[0]).__name__}]")
        describe(obj[0], indent + 1)


if __name__ == "__main__":
    data = load_counterfact()
    print("records:", len(data))
    print("schema (from record 0):")
    describe(data[0])
    json.dump(data[:10], open("data/raw_sample.json", "w"), indent=2)
    print("wrote data/raw_sample.json (first 10 records)")
