import os
import requests
import zipfile

COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '../coco_data')
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_extract(url, dest_folder):
    local_filename = url.split('/')[-1]
    local_path = os.path.join(dest_folder, local_filename)
    if not os.path.exists(local_path):
        print(f"Downloading {local_filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_path, 'wb') as f:
                from tqdm import tqdm
                for chunk in tqdm(r.iter_content(chunk_size=8192),
                                 total=total_size // 8192,
                                 unit='KB',
                                 desc=f"Downloading {local_filename}"):
                    if chunk:
                        f.write(chunk)
    else:
        print(f"{local_filename} already exists.")
    # Extract if zip
    if local_filename.endswith('.zip'):
        print(f"Extracting {local_filename}...")
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            from tqdm import tqdm
            members = zip_ref.infolist()
            for member in tqdm(members, desc=f"Extracting {local_filename}", unit="file"):
                zip_ref.extract(member, dest_folder)
        print(f"Extracted to {dest_folder}")

if __name__ == "__main__":
    for name, url in COCO_URLS.items():
        download_and_extract(url, DATA_DIR)
    print("COCO data download and extraction complete.")
