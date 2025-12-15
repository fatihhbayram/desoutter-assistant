# 🗺️ Desoutter Servis Yönetim Sistemi - Geliştirme Yol Haritası

> **Son Güncelleme:** 14 Aralık 2025  
> **Durum:** Tech Page UI Redesign Başlandı ✅ | MongoDB Config Fixed ✅

---

## 📋 Özet

Bu belge, Desoutter Repair Assistant'a eklenecek **Servis Yönetim Sistemi** ve **KPI Dashboard** özelliklerinin detaylı planını içerir.

---

## ✅ Tamamlanan Özellikler

### Tech Page UI Redesign - Wizard Flow (14 Aralık 2025)
- [x] TechWizard component oluşturma (4-step flow)
  - Step 1: Product Search & Filter
  - Step 2: Product Selection
  - Step 3: Fault Description
  - Step 4: Diagnosis Results & Feedback
- [x] Wizard CSS styling (responsive, mobile-friendly)
- [x] Progress bar with step indicators
- [x] Backend feedback endpoint validation (HTTP 422 fix)
- [x] Feedback button integration (positive/negative)
- [x] Docker build integration

### Database Configuration Fix (14 Aralık 2025)
- [x] MongoDB config updated (localhost instead of Docker IP)
- [x] MongoDBClient enhanced with collection_name parameter
- [x] tool_units collection verified (7 CVI3 controller units)
- [x] products collection verified (237 tools)
- [x] Data integrity confirmed

### CVI3 Function Units Scraper Recreation (14 Aralık 2025)
- [x] scrape_cvi3_function_units.py recreated after re-clone
- [x] Script tested and verified
- [x] Async HTTP + BeautifulSoup implementation
- [x] MongoDB save functionality

### Project Re-organization (14 Aralık 2025)
- [x] Fresh GitHub clone completed
- [x] Docker compose configuration verified
- [x] All 7 services running (Ollama, MongoDB, n8n, Frontend, API, etc.)
- [x] npm dependencies installed

### Admin Dashboard (9 Aralık 2025)
- [x] Overview cards (total, today, week, active users)
- [x] Daily trend chart (son 7 gün)
- [x] Confidence breakdown (high/medium/low)
- [x] Feedback statistics
- [x] Top diagnosed products
- [x] Common fault keywords
- [x] System status
- [x] Tab-based admin navigation
- [x] GET /admin/dashboard endpoint

### Self-Learning RAG Feedback Sistemi (9 Aralık 2025)
- [x] Kullanıcı geri bildirimi (👍/👎)
- [x] Negatif feedback için neden seçimi
- [x] Feedback'ten öğrenme mekanizması
- [x] Diagnosis history kaydı
- [x] Learned mappings (fault-solution)
- [x] API endpoints (/diagnose/feedback, /diagnose/history)
- [x] Frontend feedback UI

---

## 🔄 Devam Edilecek İşler (Next Steps)

### Documentation & RAG Enhancement - Öncelikli
- [ ] CVI3 ünitelere bağlanabilen toollar için veri taşı
- [ ] Tool bulletins (ürün bültenlerine ait PDF'ler) yükle
- [ ] Tool maintenance dosyaları (bakım dökümanları) ekle
- [ ] Admin panel aracılığıyla RAG'a ingest et (Document Upload)
- [ ] ChromaDB'ye vektör arama entegrasyonu
- [ ] Diagnosis sonuçlarında tool dökümanları referans göster

### Tech Page Wizard - Yakında
- [ ] TechWizard componentini App.jsx'e entegre et
- [ ] Eski renderTechnicianPanel kodunu comment'e al
- [ ] Wizard flow'unu production'da test et
- [ ] Mobile responsiveness doğrula
- [ ] Kullanıcı feedback'ini topla

### Admin Page Redesign - Planlanan
- [ ] Admin panel layout basitleştir
- [ ] User management sayfası iyileştir
- [ ] Document upload/ingestion workflow düzenle
- [ ] KPI dashboard'u optimize et

### Servis Talepleri Modülü - Yüksek Öncelik
- [ ] service_requests koleksiyonu oluştur
- [ ] Servis talepleri API endpoint'leri ekle
- [ ] Servis form UI geliştir
- [ ] Servis durum takibi ekle

---

### 1. Cihaz Kaydı Sistemi
- [ ] Seri numarası ile cihaz kaydı
- [ ] Otomatik ürün eşleştirme (part number → model)
- [ ] Müşteri bağlantısı
- [ ] Garanti takibi
- [ ] Servis geçmişi

### 2. Servis Kayıt Sistemi
- [ ] Yeni servis talebi oluşturma
- [ ] Garanti durumu seçimi:
  - WARRANTY (Garantili)
  - PAID (Ücretli)
  - GOODWILL (İyi Niyet)
  - CONTRACT (Sözleşme Kapsamı)
- [ ] Servis tipi seçimi:
  - SMART_CARE
  - BASIC_CARE
  - REPAIR
  - CALIBRATION
  - REPAIR_CAL (Repair + Calibration)
- [ ] Durum takibi (workflow)
- [ ] AI teşhis entegrasyonu
- [ ] Parça kullanımı kaydı
- [ ] Kalibrasyon sertifikası

### 3. Müşteri Yönetimi
- [ ] Müşteri kaydı (kurumsal/bireysel)
- [ ] İletişim bilgileri
- [ ] Sözleşme yönetimi
- [ ] Cihaz envanteri
- [ ] Servis geçmişi

### 4. KPI Dashboard (Supervisor/Manager)
- [ ] Servis metrikleri (toplam, tamamlanan, bekleyen)
- [ ] Zamanında teslim oranı
- [ ] Gelir analizi
- [ ] Teknisyen performansı
- [ ] Ürün güvenilirlik analizi
- [ ] Müşteri bazlı analiz
- [ ] Trend grafikleri
- [ ] Dışa aktarım (Excel/PDF)

### 5. Rol Yapısı Güncelleme
- [ ] Manager rolü ekleme
- [ ] Supervisor rolü ekleme
- [ ] Yetki matrisi uygulama

---

## 🗄️ Veritabanı Şeması

### devices (Cihazlar)
```javascript
{
  serial_number: String,      // Benzersiz seri no
  part_number: String,
  model_name: String,
  customer: { id, name, contact, email, phone },
  purchase_date: Date,
  warranty_end_date: Date,
  contract: { type, start_date, end_date, contract_number },
  service_history: [String],  // Servis kayıt ID'leri
  status: "active" | "retired" | "lost"
}
```

### service_requests (Servis Talepleri)
```javascript
{
  request_id: String,         // SR-YYYYMMDD-XXX
  device: { serial_number, part_number, model_name },
  customer: { id, name, reference },
  service_type: "SMART_CARE" | "BASIC_CARE" | "REPAIR" | "CALIBRATION" | "REPAIR_CAL",
  warranty_status: "WARRANTY" | "PAID" | "GOODWILL" | "CONTRACT",
  fault_description: String,
  priority: "urgent" | "high" | "normal" | "low",
  ai_diagnosis: { suggestion, confidence, sources, diagnosed_at },
  status: "pending" | "in_progress" | "waiting_parts" | "completed" | "delivered",
  status_history: [{ status, date, user, note }],
  assigned_to: String,
  cost: { labor, parts, total, currency, invoice_number },
  parts_used: [{ part_number, name, quantity, price }],
  calibration: { performed, certificate_number, next_calibration_date },
  feedback: { rating, comment, submitted_at }
}
```

### customers (Müşteriler)
```javascript
{
  customer_id: String,
  name: String,
  type: "corporate" | "individual",
  industry: String,
  address: { street, city, country },
  contacts: [{ name, role, email, phone }],
  contract: { type, devices_covered, annual_value },
  devices: [String],          // Seri numaraları
  service_history_count: Number,
  total_revenue: Number
}
```

### diagnoses (Teşhis Geçmişi)
```javascript
{
  diagnosis_id: String,
  service_request_id: String,
  device_serial: String,
  part_number: String,
  fault_description: String,
  ai_suggestion: String,
  confidence: String,
  sources: Array,
  technician: String,
  feedback: { helpful, resolved, rating, actual_solution }
}
```

---

## 🔌 API Endpoint'leri

### Cihaz Yönetimi
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | /api/devices | Cihaz listesi |
| GET | /api/devices/{serial} | Cihaz detayı |
| POST | /api/devices | Yeni cihaz |
| PUT | /api/devices/{serial} | Cihaz güncelle |
| GET | /api/devices/{serial}/history | Servis geçmişi |

### Servis Kayıtları
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | /api/services | Servis listesi |
| GET | /api/services/{id} | Servis detayı |
| POST | /api/services | Yeni servis |
| PUT | /api/services/{id} | Servis güncelle |
| PUT | /api/services/{id}/status | Durum güncelle |
| POST | /api/services/{id}/diagnose | AI teşhis |
| GET | /api/services/{id}/certificate | Kalibrasyon sertifikası |

### Müşteri Yönetimi
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | /api/customers | Müşteri listesi |
| GET | /api/customers/{id} | Müşteri detayı |
| POST | /api/customers | Yeni müşteri |
| GET | /api/customers/{id}/devices | Müşteri cihazları |

### KPI & Raporlar
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | /api/kpi/overview | Genel özet |
| GET | /api/kpi/services | Servis istatistikleri |
| GET | /api/kpi/technicians | Teknisyen performansı |
| GET | /api/kpi/products | Ürün analizi |
| GET | /api/kpi/revenue | Gelir analizi |
| GET | /api/reports/export | Dışa aktarım |

---

## 🔐 Yetki Matrisi

| Özellik | Admin | Manager | Supervisor | Technician |
|---------|:-----:|:-------:|:----------:|:----------:|
| Servis oluştur | ✅ | ✅ | ✅ | ✅ |
| Servis düzenle | ✅ | ✅ | ✅ | 🔸 |
| Servis sil | ✅ | ✅ | ❌ | ❌ |
| İyi Niyet seç | ✅ | ✅ | ✅ | ❌ |
| Cihaz kaydı | ✅ | ✅ | ✅ | ✅ |
| Müşteri kaydı | ✅ | ✅ | ✅ | ❌ |
| Sözleşme yönetimi | ✅ | ✅ | ❌ | ❌ |
| KPI Dashboard | ✅ | ✅ | ✅ | ❌ |
| Tüm teknisyen verileri | ✅ | ✅ | ❌ | ❌ |
| Gelir raporları | ✅ | ✅ | ❌ | ❌ |
| Kullanıcı yönetimi | ✅ | ❌ | ❌ | ❌ |
| Doküman yönetimi | ✅ | ✅ | ❌ | ❌ |

🔸 = Sadece kendi kayıtları

---

## 📅 Uygulama Sırası

| # | Modül | Süre | Öncelik |
|---|-------|------|---------|
| 1 | Veritabanı koleksiyonları | 2 saat | 🔴 Yüksek |
| 2 | Servis CRUD API | 4 saat | 🔴 Yüksek |
| 3 | Cihaz/Müşteri API | 3 saat | 🔴 Yüksek |
| 4 | Servis kayıt formu (UI) | 4 saat | 🔴 Yüksek |
| 5 | Servis listesi sayfası | 3 saat | 🟡 Orta |
| 6 | Servis detay sayfası | 3 saat | 🟡 Orta |
| 7 | KPI API endpoint'leri | 4 saat | 🟡 Orta |
| 8 | KPI Dashboard UI | 5 saat | 🟡 Orta |
| 9 | Rol yapısı güncelleme | 2 saat | 🟢 Düşük |
| 10 | Raporlama/Dışa aktarım | 3 saat | 🟢 Düşük |

**Toplam Tahmini Süre: ~33 saat**

---

## 🚀 Mevcut Durum (14 Aralık 2025)

**Tamamlanan:**
- ✅ Backend: FastAPI çalışıyor (http://192.168.1.125:8000)
- ✅ Frontend: React çalışıyor (http://192.168.1.125:3001)
- ✅ Database: MongoDB çalışıyor (237 products + 7 CVI3 units)
- ✅ RAG Engine: Ollama LLM + ChromaDB
- ✅ Admin Dashboard: Tamamen işlevsel
- ✅ Tech Page: Yeni Wizard component oluşturuldu

**Yakında Yapılacak:**
1. TechWizard componentini production'a al
2. Admin page UI iyileştirmeleri
3. Servis talepleri modülü
4. KPI raporları

---

## 📝 Son Yapılan Çalışmalar

### 14 Aralık 2025 - Tech Page Wizard & Infrastructure Fix

**Backend Fixes:**
```
✅ MongoDB config: localhost ile çalışıyor
✅ MongoDBClient: collection_name parameter eklendi
✅ CVI3 scraper: Tekrar oluşturuldu
✅ 7 CVI3 units: Database'de doğrulandı (tool_units collection)
```

**Frontend Changes:**
```
✅ TechWizard.jsx: 4-step wizard component
✅ TechWizard.css: Responsive styling
✅ Feedback endpoint: HTTP 422 fix yapıldı (feedback → feedback_type)
✅ Docker build: npm dependencies + new components
```

**Docker Compose:**
```
✅ Tüm 7 servis running
✅ Frontend rebuild: TechWizard entegre
✅ API rebuild: collection_name parameter
```

**Planlanan İşler (Hazırlanıyor):**
```
📋 CVI3 ünitelere bağlanabilen tool datası taşınacak
📄 Tool bulletins (ürün bültenlerine ait PDF'ler) yüklenecek
🔧 Tool maintenance dosyaları (bakım dökümanları) eklenecek
🧠 RAG'a ingest edilecek (ChromaDB vektör arama)
```

---

## 🚀 Başlangıç Noktası (Sonraki Aşama)

**Hemen Yapılacak:**
1. **CVI3 tool datası** - Bağlanabilen toolları database'e taşı
2. **Dokümantasyon yükleme** - Bulletins + Maintenance dosyalarını upload et
3. **RAG ingest** - Admin panel > Documents > Ingest ile vektör arama'ya ekle
4. **Test** - Diagnosis yaptığında tool dökümanları referans alınsın

**Ardından:**
1. TechWizard entegrasyonu - App.jsx'e import et
2. Admin page iyileştirmeleri - Layout basitleştir
3. Servis talepleri modülü - Database schema + API

---

*Bu belge, geliştirme sürecinde güncellenecektir.*
