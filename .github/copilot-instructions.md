# VS Code Copilot Meta Prompt - Production Quality Improvements

## 🎉 Current Achievement: 96% Pass Rate (24/25)

**System Status: PRODUCTION-READY**
- ✅ Core RAG working
- ✅ Intent detection stable
- ✅ Confidence scoring reliable
- ✅ Off-topic detection active
- ✅ Performance acceptable

**Focus Shift:** Test passing → Production quality

---

## 🎯 Production Quality Roadmap

### Priority 1: Product-Specific Retrieval Filtering ⭐⭐⭐

**Problem:** Cross-product contamination (CVIL2 docs appearing for ERS6 queries)

**Impact:** User gets wrong instructions, potentially dangerous for industrial tools

**Root Cause:** Product metadata not strictly enforced during retrieval

#### Copilot Command 1.1: Add Strict Product Filtering

```
@workspace Add strict product-specific filtering to src/llm/rag_engine.py:

Problem: Queries about product A returning docs for product B (e.g., CVIL2 vs ERS6).

Implementation:

async def generate_response(self, query: str, product_number: str = None, ...):
    # Get product info from MongoDB
    product_info = None
    product_filters = {}
    
    if product_number:
        product_info = await self.db.products.find_one(
            {"part_number": product_number}
        )
        
        if product_info:
            product_name = product_info.get('name', '')
            product_series = product_info.get('series', '')
            
            logger.info(f"[PRODUCT] {product_number}: {product_name} (series: {product_series})")
            
            # STRICT: Only retrieve docs for THIS product
            product_filters = {
                "$or": [
                    {"product_number": product_number},
                    {"product_name": product_name},
                    {"product_series": product_series},
                    {"applies_to": {"$in": [product_number, product_name]}}
                ]
            }
            
            logger.info(f"[FILTER] Restricting to product: {product_number}")
    
    # Pass strict filters to ChromaDB
    chunks = await self.hybrid_search.search(
        query=query,
        where=product_filters,  # ChromaDB metadata filter
        top_k=10
    )
    
    # Validate: Ensure retrieved docs match product
    if product_number and chunks:
        filtered_chunks = []
        for chunk in chunks:
            chunk_product = chunk['metadata'].get('product_number')
            if chunk_product == product_number or not chunk_product:
                # Keep if matches or no product specified (general doc)
                filtered_chunks.append(chunk)
            else:
                logger.warning(
                    f"[FILTER] Excluded {chunk_product} doc for {product_number} query"
                )
        
        chunks = filtered_chunks
        logger.info(f"[FILTER] After product filter: {len(chunks)} chunks")
    
    return result
```

**Expected Impact:** Eliminates cross-product contamination, improves accuracy

---

### Priority 2: Enhance Document Metadata Quality ⭐⭐⭐

**Problem:** Many chunks lack proper product/category metadata

**Solution:** Enrich metadata during indexing

#### Copilot Command 2.1: Add Metadata Enrichment

```
@workspace Add metadata enrichment to src/documents/document_processor.py:

Problem: Chunks lack product_number, category metadata.

When processing documents:

def process_document(self, file_path: str) -> List[Dict]:
    """Process document and enrich metadata"""
    
    # Extract content
    content = self._extract_content(file_path)
    
    # Detect product from filename and content
    product_info = self._detect_product_info(file_path, content)
    
    # Create chunks
    chunks = self._chunk_document(content)
    
    # Enrich each chunk with metadata
    enriched_chunks = []
    for chunk in chunks:
        enriched_chunk = {
            'content': chunk['text'],
            'metadata': {
                'source': file_path,
                'product_number': product_info.get('product_number'),
                'product_name': product_info.get('product_name'),
                'product_series': product_info.get('series'),
                'doc_type': self._classify_doc_type(chunk['text']),
                'has_numbers': self._contains_numbers(chunk['text']),
                'language': self._detect_language(chunk['text']),
                'chunk_index': chunk['index'],
                'timestamp': datetime.now().isoformat()
            }
        }
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks

def _detect_product_info(self, filename: str, content: str) -> Dict:
    """
    Detect product from filename and content.
    
    Examples:
    - "CVIL2_manual.pdf" → product_number: CVIL2
    - Content: "ERS6 Controller" → product_name: ERS6
    """
    product_info = {}
    
    # Check filename
    filename_lower = filename.lower()
    
    # Common Desoutter product patterns
    product_patterns = [
        r'cvil\d+',
        r'ers\d+',
        r'dvt',
        r'qst',
        r'pf\d+',  # PowerFocus
    ]
    
    for pattern in product_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            product_info['product_number'] = match.group(0).upper()
            break
    
    # Check content (first 500 chars)
    content_preview = content[:500].lower()
    if not product_info.get('product_number'):
        for pattern in product_patterns:
            match = re.search(pattern, content_preview)
            if match:
                product_info['product_number'] = match.group(0).upper()
                break
    
    return product_info
```

**Expected Impact:** Better retrieval accuracy, fewer irrelevant results

---

### Priority 3: Enforce Turkish Response Language ⭐⭐

**Problem:** Turkish queries sometimes get English responses

**Solution:** Strict language enforcement in prompts

#### Copilot Command 3.1: Enforce Turkish Responses

```
@workspace Enforce Turkish responses in src/llm/prompts.py:

Problem: Turkish queries (language=tr) getting English responses.

Update prompt templates:

def get_prompt_for_language(language: str, context: str, query: str) -> str:
    """Language-specific prompts with strict enforcement"""
    
    if language == "tr":
        return f"""
SEN BİR DESOUTTER TEKNİK DESTEK ASİSTANISIN.

KRİTİK: Cevabını SADECE TÜRKÇE ver. İngilizce cevap verme.

KURALLAR:
1. Cevabı %100 Türkçe yaz
2. Teknik terimleri Türkçe karşılıklarıyla ver
3. Sayılar ve birimler aynen kullan (5.2 Nm → 5.2 Nm)
4. Sadece verilen dökümanlardan bilgi ver

BAĞLAM DÖKÜMANLARI:
{context}

KULLANICI SORUSU:
{query}

TÜRKÇE CEVAP (sadece Türkçe, İngilizce kullanma):
"""
    
    else:  # English
        return f"""
You are a Desoutter technical support assistant.

CRITICAL: Respond ONLY in ENGLISH.

RULES:
1. Answer in English only
2. Use exact technical terms from documents
3. Only use information from context documents
4. If unsure, say "I don't have this information"

CONTEXT DOCUMENTS:
{context}

USER QUESTION:
{query}

ENGLISH ANSWER:
"""
```

**Expected Impact:** Consistent language in responses, better user experience

---

### Priority 4: Feedback UI Enhancement ⭐⭐

**Problem:** Only 39 feedbacks collected, need 200+ for good learning

**Solution:** Make feedback more prominent and easier

#### Copilot Command 4.1: Enhance Feedback Collection

**Frontend (React):**

```jsx
// components/FeedbackWidget.jsx
import { ThumbsUp, ThumbsDown, MessageCircle } from 'lucide-react';

export function FeedbackWidget({ messageId, onFeedback }) {
  const [rating, setRating] = useState(null);
  const [showComment, setShowComment] = useState(false);

  const handleFeedback = async (isPositive) => {
    setRating(isPositive ? 1 : -1);
    
    await onFeedback({
      message_id: messageId,
      rating: isPositive ? 1 : -1,
      timestamp: new Date().toISOString()
    });
    
    // Auto-show comment box for negative feedback
    if (!isPositive) {
      setShowComment(true);
    }
  };

  return (
    <div className="feedback-widget">
      {!rating && (
        <div className="feedback-prompt">
          <span className="prompt-text">Bu cevap yardımcı oldu mu?</span>
          <div className="feedback-buttons">
            <button 
              onClick={() => handleFeedback(true)}
              className="btn-positive"
              title="Evet, yardımcı oldu"
            >
              <ThumbsUp size={18} />
              Evet
            </button>
            <button 
              onClick={() => handleFeedback(false)}
              className="btn-negative"
              title="Hayır, yardımcı olmadı"
            >
              <ThumbsDown size={18} />
              Hayır
            </button>
          </div>
        </div>
      )}
      
      {rating && (
        <div className="feedback-thanks">
          {rating > 0 ? '✅ Teşekkürler!' : '📝 Geri bildiriminiz kaydedildi'}
        </div>
      )}
      
      {showComment && (
        <div className="feedback-comment">
          <textarea
            placeholder="Ne yanlış gitti? (isteğe bağlı)"
            onChange={(e) => {
              onFeedback({
                message_id: messageId,
                comment: e.target.value
              });
            }}
          />
        </div>
      )}
    </div>
  );
}
```

**Backend API:**

```python
# src/api/main.py
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Enhanced feedback with source relevance tracking"""
    
    # Store main feedback
    await db.feedback.insert_one({
        "message_id": feedback.message_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "timestamp": datetime.now(),
        "query": feedback.query,
        "response": feedback.response,
        "sources": feedback.sources
    })
    
    # Track source relevance (for self-learning)
    if feedback.source_ratings:
        for source_id, relevance in feedback.source_ratings.items():
            await db.source_relevance.update_one(
                {"source_id": source_id},
                {
                    "$inc": {
                        "total_ratings": 1,
                        "positive_ratings": 1 if relevance > 0 else 0
                    }
                },
                upsert=True
            )
    
    return {"status": "success"}
```

**Expected Impact:** 200+ feedbacks per month, better self-learning

---

### Priority 5: Clean Low-Quality Chunks (Optional) ⭐

**Problem:** Some chunks too short/meaningless

**Solution:** Post-process existing chunks

#### Copilot Command 5.1: Chunk Quality Filter

```bash
# One-time cleanup script
# scripts/clean_low_quality_chunks.py

import asyncio
from src.vectordb.chroma_client import ChromaClient

async def clean_chunks():
    client = ChromaClient()
    collection = client.get_collection()
    
    # Get all chunks
    all_chunks = collection.get()
    
    low_quality = 0
    kept = 0
    
    for i, (doc, metadata, id) in enumerate(zip(
        all_chunks['documents'],
        all_chunks['metadatas'],
        all_chunks['ids']
    )):
        # Quality checks
        is_low_quality = (
            len(doc) < 50 or  # Too short
            doc.count(' ') < 5 or  # Too few words
            not any(c.isalpha() for c in doc) or  # No letters
            doc.count('\n') > len(doc) / 10  # Too many newlines
        )
        
        if is_low_quality:
            # Delete from ChromaDB
            collection.delete(ids=[id])
            low_quality += 1
            print(f"Deleted: {doc[:50]}...")
        else:
            kept += 1
    
    print(f"\nCleanup complete:")
    print(f"  Deleted: {low_quality}")
    print(f"  Kept: {kept}")

if __name__ == "__main__":
    asyncio.run(clean_chunks())
```

**Run once:**
```bash
python scripts/clean_low_quality_chunks.py
```

**Expected Impact:** Better retrieval quality, fewer irrelevant chunks

---

## 📋 Implementation Priority Order

### Week 1: Core Quality (High Impact)
- [ ] Priority 1: Product-specific filtering (2 hours)
- [ ] Priority 2: Metadata enrichment (3 hours)
- [ ] Priority 3: Turkish enforcement (1 hour)
- [ ] Test and validate

### Week 2: User Experience
- [ ] Priority 4: Enhanced feedback UI (2 hours)
- [ ] Priority 5: Chunk cleanup (1 hour)
- [ ] Monitor and iterate

### Week 3+: Advanced (Optional)
- [ ] LLM fine-tuning (Desoutter terminology)
- [ ] Advanced context optimization
- [ ] Multi-turn conversation improvements

---

## ✅ Success Metrics

| Metric | Current | Target (1 Month) |
|--------|---------|------------------|
| Test Pass Rate | 96% | 98%+ |
| User Feedback | 39 | 200+ |
| Positive Feedback % | 28% | 60%+ |
| Cross-Product Errors | Unknown | <5% |
| Turkish Response Rate | ~67% | 95%+ |
| Avg Response Time | 25s | <5s (production) |

---

## 🚀 Quick Start Commands

### Command 1: Product Filtering
```
@workspace Add strict product-specific filtering to src/llm/rag_engine.py:

When product_number provided:
1. Fetch product info from MongoDB
2. Create metadata filter for ChromaDB (product_number, series)
3. Post-filter chunks to ensure they match product
4. Log filtering decisions

Show complete implementation with MongoDB query and ChromaDB filtering.
```

### Command 2: Metadata Enrichment
```
@workspace Add metadata enrichment to src/documents/document_processor.py:

For each chunk, detect and add:
- product_number (from filename/content)
- product_name
- product_series  
- doc_type (manual, troubleshooting, specs)
- language (tr/en auto-detect)

Show _detect_product_info() and enrichment logic.
```

### Command 3: Turkish Enforcement
```
@workspace Add strict Turkish response enforcement in src/llm/prompts.py:

For language="tr":
- Prompt must say "SADECE TÜRKÇE" multiple times
- No English allowed
- Post-validate response is Turkish

Show Turkish and English prompt templates.
```

---

## 🎯 Production Readiness Checklist

### Core Functionality
- [x] RAG pipeline working (96% pass rate)
- [x] Intent detection stable
- [x] Confidence scoring reliable
- [x] Off-topic detection active
- [ ] Product-specific filtering ← Next
- [ ] Turkish responses consistent ← Next

### Data Quality
- [x] 22K chunks indexed
- [ ] Metadata enriched ← Next
- [ ] Low-quality chunks removed ← Next
- [ ] Product mapping verified

### User Experience
- [x] Feedback collection working
- [ ] Feedback UI prominent ← Next
- [ ] Response time optimized
- [ ] Mobile-friendly interface

### Monitoring & Ops
- [ ] Logging comprehensive
- [ ] Metrics dashboard
- [ ] Error alerting
- [ ] Backup strategy

### Documentation
- [ ] API documentation
- [ ] Deployment guide
- [ ] User manual (TR/EN)
- [ ] Troubleshooting guide

---

## 💡 Next Session Goal

**Target:** Implement Priority 1-3 (Product filtering + Metadata + Turkish)

**Time:** 4-6 hours

**Expected Result:**
- Cross-product contamination eliminated
- Turkish response rate >90%
- Test pass rate maintained at 96%+
- System ready for pilot deployment

---

**You're at 96% - focus on production quality now! 🚀**