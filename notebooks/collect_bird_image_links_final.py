import pandas as pd
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

INPUT_FILE = "birds_master.csv"
OUTPUT_FILE = "bird_image_links.csv"

MAX_WORKERS = 12
IMAGES_PER_SPECIES = 20
SAVE_BATCH = 500

birds = pd.read_csv(INPUT_FILE)

# Resume support
done_species = set()
if os.path.exists(OUTPUT_FILE):
    old = pd.read_csv(OUTPUT_FILE)
    done_species = set(old["scientific_name"].unique())
    print(f"Resuming: {len(done_species)} species already processed")

birds = birds[~birds["scientific_name"].isin(done_species)].reset_index(drop=True)

session = requests.Session()
session.headers.update({"User-Agent": "BirdDatasetBuilder/1.0"})


def fetch_species(row):

    sci = row["scientific_name"]
    com = row["common_name"]
    order = row["order"]
    family = row["family"]

    url = "https://api.inaturalist.org/v1/observations"

    params = {
        "taxon_name": sci,
        "photos": "true",
        "per_page": IMAGES_PER_SPECIES
    }

    for _ in range(3):
        try:
            r = session.get(url, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()

            rows = []

            for obs in data.get("results", []):
                for photo in obs.get("photos", []):

                    img = photo["url"].replace("square", "large")

                    rows.append({
                        "scientific_name": sci,
                        "common_name": com,
                        "order": order,
                        "family": family,
                        "image_url": img
                    })

            return rows

        except Exception:
            time.sleep(2)

    return []


results = []

print("Starting collection...\n")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = [executor.submit(fetch_species, row)
               for _, row in birds.iterrows()]

    for future in tqdm(as_completed(futures), total=len(futures)):

        res = future.result()

        if res:
            results.extend(res)

        if len(results) >= SAVE_BATCH:

            df = pd.DataFrame(results)

            if os.path.exists(OUTPUT_FILE):
                df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
            else:
                df.to_csv(OUTPUT_FILE, index=False)

            results = []


# Save remaining results
if results:
    df = pd.DataFrame(results)

    if os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)

print("\nFinished collecting bird image links.")
print("Output file:", OUTPUT_FILE)