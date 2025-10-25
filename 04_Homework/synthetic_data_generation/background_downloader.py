# Usage `uv run background_downloader.py -o ../data/generated_backgrounds -q "street" -q "indoor background" -n 20`

import os
import requests
import click
from duckduckgo_search import DDGS

def download_ddg_images(query, limit, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔎 Lade {limit} Bilder für '{query}' herunter...")

    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=limit)

        for idx, r in enumerate(results):
            url = r.get("image")
            if not url:
                continue
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    ext = ".jpg" if "jpeg" in response.headers.get("Content-Type", "") else ".png"
                    filepath = os.path.join(output_dir, f"{query.replace(' ', '_')}_{idx}{ext}")
                    with open(filepath, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                print(f"⚠️ Fehler bei {url}: {e}")

    print(f"✅ Fertig! Bilder für '{query}' liegen in: {output_dir}")

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-o', '--output-path', required=True, type=click.Path(file_okay=False),
              help="Ordner, in dem die Bilder gespeichert werden.")
@click.option('-q', '--queries', multiple=True, required=True,
              help="Suchbegriffe für Bilder. Mehrfach angeben mit -q 'street' -q 'forest'.")
@click.option('-n', '--num-images', default=50, type=int,
              help="Anzahl Bilder pro Suchbegriff.", show_default=True)
def main(output_path, queries, num_images):
    """
    Lade Bilder von DuckDuckGo für Trainingshintergründe herunter.

    Beispiel:
      python background_generator.py -o ./input -q "street" -q "forest" -q "indoor background" -n 30
    """
    for query in queries:
        download_ddg_images(query, num_images, output_path)

if __name__ == "__main__":
    main()
