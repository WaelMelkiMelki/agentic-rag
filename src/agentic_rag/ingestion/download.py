import requests
from pathlib import Path
import yaml

def download_corpus(manifest_path: str, output_dir: str):
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for doc in manifest["corpus"]:
        filepath = output / f"{doc['id']}.pdf"
        if filepath.exists():
            print(f"⏭️  {doc['id']} already downloaded")
            continue
        print(f"⬇️  Downloading {doc['id']}...")
        resp = requests.get(doc["url"], timeout=30)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        print(f"✅ Saved ({len(resp.content)} bytes)")

if __name__ == "__main__":
    download_corpus("configs/corpus_manifest.yaml", "data/raw")