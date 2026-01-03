# TODO: Next Session

## Status: Source Citation Enhancement COMPLETE ✅

We have successfully fixed the page number extraction issue and re-ingested all documents. The RAG system now has accurate metadata.

## Next Priorities

### 1. Phase 2: Retrieval Enhancement (Week 1)
- [ ] **Hybrid Retrieval:** Implement `rank_bm25` to combine keyword search with vector search.
- [ ] **Re-Ranking:** Implement a re-ranker (e.g., Cross-Encoder) to filter top-k results for better relevance.
- [ ] **Metadata Filtering:** Add filters for `product_line` and `doc_type` in the API query.

### 2. Phase 3: Dynamic Prompts (Week 2)
- [ ] **Intent Detection:** Refine the `IntentDetector` to better classify queries (Troubleshooting vs. Specs).
- [ ] **Prompt Templates:** Create specific system prompts for each intent type.

## Technical Debt / Maintenance
- [ ] **Unit Tests:** Add more comprehensive unit tests for `DocumentProcessor`.
- [ ] **CI/CD:** Consider setting up a GitHub Action for automated testing.

---

## 🎫 PHASE 4: Freshdesk Ticket Scraper Entegrasyonu ✅ COMPLETE

**Kaynak:** Desoutter Support Portal (Freshdesk) - Gerçek müşteri soruları ve destek çözümleri
**Değer:** Q&A formatında gerçek dünya sorunları + PDF attachment içerikleri
**Durum:** ✅ Tüm kod tamamlandı - Test ve kullanım aşamasında

### Oluşturulan Dosyalar
- [x] `src/scraper/ticket_scraper.py` - Async ticket scraper (aiohttp)
- [x] `src/database/models.py` - TicketModel, TicketComment, TicketAttachment
- [x] `scripts/scrape_tickets.py` - Ticket scraping script
- [x] `scripts/ingest_tickets.py` - Ticket'ları RAG'a ekleme

### Yapılan Değişiklikler
- [x] `requirements.txt` - `pdfplumber`, `PyPDF2` eklendi
- [x] `src/database/mongo_client.py` - `tickets` collection desteği
- [x] `config/settings.py` - Freshdesk credentials config
- [x] `src/database/__init__.py` - TicketModel exports

### Kullanım

```bash
# 1. Environment variables ayarla
export FRESHDESK_EMAIL="your-email@company.com"
export FRESHDESK_PASSWORD="your-password"

# 2. Test scrape (son 3 sayfa)
python scripts/scrape_tickets.py --test

# 3. Son 50 sayfa scrape
python scripts/scrape_tickets.py --pages 50

# 4. Tam scrape (1675 sayfa)
python scripts/scrape_tickets.py --full

# 5. PDF indirmeden hızlı scrape
python scripts/scrape_tickets.py --pages 100 --no-pdf

# 6. Yarıda kaldıysa devam et
python scripts/scrape_tickets.py --resume

# 7. Ticket'ları RAG'a ekle
python scripts/ingest_tickets.py

# 8. Sadece çözülmüş ticket'ları ekle
python scripts/ingest_tickets.py --resolved-only
```

### Data Locations
- Ticket IDs: `data/tickets/ticket_ids.json`
- Checkpoint: `data/tickets/checkpoint.json`
- RAG Export: `data/tickets/tickets_rag.json`
- Downloaded PDFs: `data/ticket_pdfs/`

---

## 🚩 YARIN YAPILACAKLAR

- [ ] Son 200 ticketı çek
- [ ] Son 200 ticketı önişlemeden geçir
- [ ] Son 200 ticketı vector db'ye ekle
- [ ] Tüm değişiklikleri commit et
 