import os
import requests
from pymongo import MongoClient

# MongoDB bağlantı bilgileri
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "desoutter"
COLLECTION = "products"
PLACEHOLDER_URL = "https://www.desouttertools.com/images/image-placeholder.svg"
DOWNLOAD_DIR = "downloaded_missing_images"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_image(url, filename):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.headers.get('content-type', '').startswith('image'):
            with open(filename, 'wb') as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"Download error: {e}")
    return False

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION]
    missing = list(col.find({"image_url": PLACEHOLDER_URL}))
    print(f"Eksik görseli olan {len(missing)} ürün bulundu.")
    for prod in missing:
        pn = prod.get("part_number") or prod.get("code") or str(prod.get("_id"))
        filename = os.path.join(DOWNLOAD_DIR, f"{pn}.svg")
        if os.path.exists(filename):
            continue
        success = download_image(PLACEHOLDER_URL, filename)
        if success:
            print(f"İndirildi: {filename}")
        else:
            print(f"İndirilemedi: {pn}")

if __name__ == "__main__":
    main()
