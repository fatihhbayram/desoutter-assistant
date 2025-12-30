# 🗺️ Desoutter Service Management System - Development Roadmap

> **Last Update:** December 30, 2025  
> **Status:** 🎉 RAG QUALITY IMPROVEMENTS (Priorities 1-5 ✅) | ChromaDB 10,866 vectors ✅ | BM25 19,032 terms ✅ | Domain 351 terms ✅


---

## 📋 Summary

This document contains the detailed plan for the **Service Management System** and **KPI Dashboard** features to be added to the Desoutter Repair Assistant.

---

## ✅ Completed Features

### RAG Enhancement Roadmap - ALL PHASES COMPLETE (December 22, 2025)
- [x] Phase 1: Semantic Chunking (Dec 15)
- [x] Phase 2: Hybrid Search + Response Cache (Dec 16)
- [x] Phase 3.3-3.4: Source Relevance + Context Optimization (Dec 17)
- [x] Phase 4.1: Metadata Filtering & Boosting (Dec 17)
- [x] Phase 5.1: Performance Metrics (Dec 22)
- [x] Phase 3.5: Multi-turn Conversation (Dec 22)
- [x] Phase 6: Self-Learning Feedback Loop (Dec 22)
- [x] **Phase 3.1: Domain Embeddings** (Dec 22) - 351 Desoutter terms, query enhancement

**Details:** [RAG_ENHANCEMENT_ROADMAP.md](RAG_ENHANCEMENT_ROADMAP.md)

### RAG Quality Improvements - 2026 Roadmap (December 29-30, 2025)
- [x] **Priority 1: Response Grounding & "I Don't Know" Logic** (Dec 29)
  - Context sufficiency scoring (multi-factor: similarity, doc count, term coverage)
  - "I don't know" responses (EN/TR) when context inadequate
  - Target: 10-15% "I don't know" rate
  - Test coverage: 7/7 passing (100%)
  
- [x] **Priority 2: Response Validation (Hallucination Detection)** (Dec 29)
  - Uncertainty phrase detection (6 patterns)
  - Numerical value verification (ensures numbers exist in context)
  - Product mismatch detection
  - Forbidden content blocking (WiFi/battery on non-capable products)
  - Auto-flagging for admin review
  - Test coverage: 7/8 passing (87.5%)
  
- [x] **Priority 3: Intent-Based Dynamic Prompts** (Dec 29-30)
  - Intent detector with 8 query types (troubleshooting, specs, installation, calibration, maintenance, connection, error codes, general)
  - 8 specialized system prompts with strict grounding rules
  - EN/TR keyword support
  - ✅ Integrated into RAG Engine (Dec 30)
  - ✅ API metadata exposure (Dec 30)

- [x] **Priority 4: Content Deduplication** (Dec 30)
  - SHA-256 content hashing
  - Duplicate detection before indexing
  - Configurable via `ENABLE_DEDUPLICATION`
  - Test coverage: 100%

- [x] **Priority 5: Adaptive Chunk Sizing** (Dec 30)
  - Document type-based sizing (200-400 tokens)
  - Troubleshooting: 200 tokens (precision)
  - Manuals: 400 tokens (context)
  - Test coverage: 100%

**Files:**
- `src/llm/context_grounding.py` (260 lines)
- `src/llm/response_validator.py` (380 lines)
- `src/llm/intent_detector.py` (250 lines)
- `scripts/test_context_grounding.py` (262 lines)
- `scripts/test_response_validator.py` (370 lines)

**Detaylar:** [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md), [walkthrough.md](/.gemini/antigravity/brain/9929f311-5135-4784-88de-b8959ce3b72a/walkthrough.md)


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

### Scraping - Rate Limit Sonrası (Bekliyor)
- [ ] 11 kalan seri scrape et (Cable Tightening + Electric Drilling)

### TechWizard Entegrasyonu - Planlanan
- [ ] TechWizard componentini App.jsx'e entegre et
- [ ] Eski renderTechnicianPanel kodunu comment'e al
- [ ] Wizard flow'unu production'da test et
- [ ] Mobile responsiveness doğrula

### Embedding Fine-tuning - Opsiyonel (100+ pair gerekli)
- [ ] Contrastive pair toplama (şu an: 0 pair)
- [ ] Domain-specific embedding modeli eğit

---

## ✅ Tamamlanan RAG Fazları (Chronological)

### Phase 1: Semantic Chunking - TAMAMLANDI ✅ (15 Aralık 2025)
- [x] SemanticChunker module (400+ lines) - Recursive chunking with structure preservation
- [x] DocumentTypeDetector - 5 document type classifications
- [x] FaultKeywordExtractor - 9 repair domain categories
- [x] ChunkMetadata - 14-field metadata per chunk
- [x] DocumentProcessor integration - Full semantic chunking pipeline
- [x] Test suite - 4/4 tests PASSED

### Phase 2.1: Document Re-ingestion - TAMAMLANDI ✅ (16 Aralık 2025)
- [x] Path configuration fix (DOCUMENTS_DIR)
- [x] 276 documents re-processed with semantic chunking
- [x] 1229 new semantic chunks generated
- [x] Total vectors in ChromaDB: 2309

### Phase 2.2: Hybrid Search - TAMAMLANDI ✅ (16 Aralık 2025)
- [x] HybridSearcher class (700+ lines)
- [x] BM25Index - Keyword search (13026 unique terms)
- [x] QueryExpander - Domain synonym expansion (9 categories)
- [x] RRF Fusion - Reciprocal Rank Fusion algorithm
- [x] Test suite - 5/5 tests PASSED

### Phase 2.3: Response Caching - TAMAMLANDI ✅ (16 Aralık 2025)
- [x] LRU cache for repeated queries
- [x] TTL-based expiration
- [x] ~100,000x speedup for cache hits

### Phase 3.3-3.4: Source Relevance + Context - TAMAMLANDI ✅ (17 Aralık 2025)
- [x] Per-document relevance feedback UI
- [x] Context window optimization (8K token budget)
- [x] Deduplication and warning prioritization

### Phase 4.1: Metadata Boosting - TAMAMLANDI ✅ (17 Aralık 2025)
- [x] Service bulletin boost (1.5x)
- [x] Procedure boost (1.3x)
- [x] Warning boost (1.2x)

### Phase 5.1: Performance Metrics - TAMAMLANDI ✅ (22 Aralık 2025)
- [x] Query latency tracking (retrieval, LLM, total)
- [x] Cache hit/miss rate monitoring
- [x] P95/P99 latency percentiles
- [x] Health status monitoring
- [x] New endpoints: /admin/metrics/*

### Phase 3.5: Multi-turn Conversation - TAMAMLANDI ✅ (22 Aralık 2025)
- [x] Session management (30 min timeout)
- [x] Context preservation
- [x] Reference resolution
- [x] New endpoints: /conversation/*

### Phase 6: Self-Learning Feedback Loop - TAMAMLANDI ✅ (22 Aralık 2025)
- [x] Feedback signal propagation
- [x] Wilson score source ranking
- [x] Keyword-to-source mapping
- [x] Training data collection
- [x] New endpoints: /admin/learning/*

### Phase 3.1: Domain Embeddings - TAMAMLANDI ✅ (22 Aralık 2025)
- [x] DomainVocabulary (351 terms)
- [x] 27 product series, 29 error codes
- [x] Query enhancement with synonyms
- [x] Entity extraction
- [x] Term weight learning
- [x] New endpoints: /admin/domain/*

**Detaylar:** [RAG_ENHANCEMENT_ROADMAP.md](RAG_ENHANCEMENT_ROADMAP.md)

---

### Documentation & RAG Enhancement - Tamamlandı ✅ (15 Aralık 2025)
- [x] CVI3 ünitelere bağlanabilen toollar için veri taşı
- [x] Tool bulletins (ürün bültenlerine ait PDF'ler) yükle
- [x] Tool maintenance dosyaları (bakım dökümanları) ekle
- [x] Admin panel aracılığıyla RAG'a ingest et (Document Upload) - 276 doc, 1080 chunks ✅
- [x] ChromaDB'ye vektör arama entegrasyonu ✅ (1080 chunks in vector DB)
- [x] Diagnosis sonuçlarında tool dökümanları referans göster ✅ (Sources returned)
- [x] RAG Retrieval Quality Optimization - Dynamic similarity threshold ✅

**Detaylar:** [CHANGELOG.md](CHANGELOG.md#-15-aralık-2025-pazar) - RAG Retrieval Quality Optimization section

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

## 🚀 Mevcut Durum (15 Aralık 2025)

**Tamamlanan:**
- ✅ Backend: FastAPI çalışıyor (http://localhost:8000)
- ✅ Frontend: React çalışıyor (http://localhost:3001)
- ✅ Database: MongoDB çalışıyor (237 products + 7 CVI3 units)
- ✅ RAG Engine: Ollama LLM + ChromaDB (1080 chunks, 5 sources per diagnosis)
- ✅ Admin Dashboard: Tamamen işlevsel
- ✅ Tech Page: TechWizard component (4-step wizard - ready to integrate)
- ✅ Documentation: 276 dokument ingested (bulletins + manuals)
- ✅ Vector DB: ChromaDB fully operational with similarity search
- ✅ Excel Support: PDF, DOCX, PPTX, XLSX, XLS parsing

**Yakında Yapılacak:**
1. TechWizard entegrasyonu (App.jsx'e import)
2. Admin page UI iyileştirmeleri
3. Servis talepleri modülü (service_requests collection)
4. KPI raporları ve dashboards

---

## 📝 Son Yapılan Çalışmalar

### 15 Aralık 2025 - RAG Dokumentasyon & ChromaDB Integration

**Tamamlanan:**
```
✅ Excel desteği: XLSX, XLS parsing eklendi
✅ Dokument yükleme: 276 dokument (bulletins + manuals)
✅ RAG Ingest: 1080 chunk oluşturuldu ve ChromaDB'ye eklendi
✅ Vector Search: Diagnosis'te 5 kaynak bulunuyor (similarity score ile)
✅ API Test: Grinding noise → CVI3 evolution, ExD measurement dökümanları
✅ Sources: Diagnosis sonuçlarında referans gösteriliyyor
```

**Docker Compose:**
```
✅ ai-stack.yml ile 7 servis running
✅ Tüm bileşenler healthy ve synced
✅ ChromaDB persistent volume çalışıyor
```

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

---

## 🚀 Başlangıç Noktası (Sonraki Aşama - Sırada)

**Hemen Yapılacak (Priority Order):**
1. **[HIGH]** TechWizard entegrasyonu - App.jsx'e import et (Sources göster)
2. **[HIGH]** Admin page UI iyileştirmeleri - Doküman yönetimi basitleştir
3. **[MEDIUM]** Servis talepleri modülü - service_requests collection + API
4. **[MEDIUM]** KPI Dashboard - Real-time metrics ve raporlar

**Daha Sonra:**
1. Cihaz kaydı sistemi (device registry)
2. Müşteri yönetimi
3. Rol yapısı güncelleme (Manager, Supervisor roles)
4. Raporlama ve Excel dışa aktarım

---

## 🚩 2026 KALİTE VE GÜVENİLİRLİK ODAKLI YENİ NESİL RAG YOL HARİTASI

### Kritik Yapılacaklar (2026 Q1)

- [ ] **Document processor:**
    - Section-aware chunking (başlık, tablo, kod blokları, paragraflar)
    - Zengin metadata (doc_type, product_line, section_title, content_hash, has_tables, has_numbers, timestamp)
    - SHA256 hash ile deduplikasyon ve loglama
- [ ] **Retrieval:**
    - Hybrid search (vector + BM25)
    - Metadata filtering (ürün, doc_type, sayısal içerik)
    - Top-10 retrieval, semantic re-ranking ile top-5
- [ ] **Prompt engineering:**
    - Sorgu tipine göre dinamik, grounded prompt
    - Context injection formatı ve kaynak gösterimi
    - "Bilmiyorum" cevabı zorunlu
- [ ] **User profile:**
    - Kullanıcı profili şeması ve yanıt özelleştirme (beginner/advanced)
    - Retrieval’da ürün önceliğiyle filtreleme
- [ ] **Feedback loop:**
    - Yanıt sonrası thumbs up/down, rating, yorum ve otomatik flagging
    - Flaglenen yanıtlar için log ve haftalık analiz

#### Başarı Kriterleri
- Relevance Rate: >80% top-5 chunk gerçekten ilgili
- User rating ≥4/5: >70% soruda
- "Bilmiyorum" oranı: 10-15% (hallucination yok)
- Yanıt süresi: <3 sn end-to-end

---

*Bu belge, geliştirme sürecinde güncellenecektir.*
