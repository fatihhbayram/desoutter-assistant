import os
import requests
from pymongo import MongoClient
from urllib.parse import quote

# MongoDB connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "desoutter"
COLLECTION = "products"
PLACEHOLDER_URL = "https://www.desouttertools.com/images/image-placeholder.svg"

# Desoutter ana ürün görseli arama URL şablonu (örnek, gerekirse değiştir)
PRODUCT_IMAGE_URL = "https://www.desouttertools.com/en/products/{part_number}/image"


def find_real_image(part_number):
    """
    Gerçek ürün görselini bulmaya çalışır. Bulamazsa None döner.
    """
    url = PRODUCT_IMAGE_URL.format(part_number=quote(part_number))
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.headers.get('content-type', '').startswith('image'):
            return url
    except Exception:
        pass
    return None

def update_missing_images():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION]
    
    missing = list(col.find({"image_url": PLACEHOLDER_URL}))
    print(f"Eksik görseli olan {len(missing)} ürün bulundu.")
    updated = 0
    for prod in missing:
        pn = prod.get("part_number") or prod.get("code")
        if not pn:
            continue
        real_img = find_real_image(pn)
        if real_img:
            col.update_one({"_id": prod["_id"]}, {"$set": {"image_url": real_img}})
            print(f"Güncellendi: {pn} -> {real_img}")
            updated += 1
    print(f"Toplam güncellenen: {updated}")

if __name__ == "__main__":
    update_missing_images()
