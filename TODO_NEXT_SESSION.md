# 📋 TODO - Next Session (December 19, 2025)

## 🚀 Hemen Başlanacak: Scrape İşlemi

Rate limit nedeniyle atlanan serileri scrape et:

```bash
# 1. Rate limit kontrolü
curl -s -o /dev/null -w "%{http_code}" "https://www.desouttertools.com/en/p/xpb-modular-164687"
# 200 ise devam et, 429 ise bekle

# 2. Atlanan serileri scrape et
sudo docker exec -it desoutter-api python3 /app/scripts/scrape_missing.py

# 3. Veya tüm kategorileri yeniden scrape et
sudo docker exec -it desoutter-api python3 /app/scripts/scrape_all.py
```

---

## ⏳ Atlanan Seriler (13 adet)

### Cable Tightening (9 seri):
| Seri | URL |
|------|-----|
| SLBN | https://www.desouttertools.com/en/p/slbn-low-voltage-screwdriver-with-clutch-shut-off-27324 |
| E-Pulse | https://www.desouttertools.com/en/p/e-pulse-electric-pulse-pistol-corded-transducerized-nutrunner-27350 |
| EFD | https://www.desouttertools.com/en/p/efd-electric-fixtured-direct-nutrunner-130856 |
| EFM | https://www.desouttertools.com/en/p/efm-electric-fixtured-multi-nutrunner-191845 |
| ERF | https://www.desouttertools.com/en/p/erf-fixtured-electric-spindles-326679 |
| EFMA | https://www.desouttertools.com/en/p/efma-transducerized-angle-head-spindle-718240 |
| EFBCI | https://www.desouttertools.com/en/p/efbci-fast-integration-spindles-straight-718237 |
| EFBCIT | https://www.desouttertools.com/en/p/efbcit-fast-integration-spindles-straight-telescopic-718238 |
| EFBCA | https://www.desouttertools.com/en/p/efbca-fast-integration-spindles-angled-715011 |

### Electric Drilling (4 seri):
| Seri | URL |
|------|-----|
| XPB Modular | https://www.desouttertools.com/en/p/xpb-modular-164687 |
| XPB One | https://www.desouttertools.com/en/p/xpb-one-164685 |
| Tightening Head | https://www.desouttertools.com/en/p/tightening-head-679250 |
| Drilling Head | https://www.desouttertools.com/en/p/drilling-head-679249 |

---

## 📊 Mevcut Durum (18 Aralık 2025)

| Metrik | Değer |
|--------|-------|
| **Toplam ürün** | 277 |
| **Battery Tightening** | 151 ✅ |
| **Cable Tightening** | 126 (kısmi) |
| **Electric Drilling** | 0 (bekliyor) |
| **Gerçek görsel** | 167 |
| **Placeholder görsel** | 110 |

---

## 🖼️ Görsel Güncelleme

Scrape sonrası placeholder görselleri güncellenecek. Frontend'de placeholder kontrolü eklendi - placeholder olan ürünler 📷 ikonu gösteriyor.

---

## ✅ Tamamlanan İşler (18 Aralık)

1. ✅ ProductModel Schema v2 - Kategorilendirme
2. ✅ `product_categorizer.py` - Helper fonksiyonlar
3. ✅ `mongo_client.py` - Smart upsert
4. ✅ `desoutter_scraper.py` - Schema v2 entegrasyonu
5. ✅ WiFi detection logic (3 iterasyon)
6. ✅ 277 ürün scrape edildi
7. ✅ Frontend placeholder filter

---

## 📁 Hazır Script'ler

| Script | Açıklama |
|--------|----------|
| `/app/scripts/scrape_missing.py` | Sadece atlanan serileri scrape eder (30sn aralıklarla) |
| `/app/scripts/scrape_all.py` | Tüm kategorileri scrape eder |
| `/app/scripts/scrape_single.py` | Tek seri scrape eder |

---

## 🔧 Yarın Kontrol Edilecek

1. Rate limit durumu (curl ile test)
2. Scrape missing series
3. Görsel URL'lerini kontrol et
4. Frontend'de görselleri doğrula
