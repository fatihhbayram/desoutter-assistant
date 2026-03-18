#!/usr/bin/env python3
"""
RAG System Comprehensive Test Script
Desoutter Assistant QA Testing
"""

from src.database.mongo_client import MongoDBClient
from src.llm.rag_engine import RAGEngine
import random
import time

print("=" * 70)
print("DESOUTTER ASSISTANT RAG - KAPSAMLI TEST RAPORU")
print("Tarih: 2026-01-07")
print("=" * 70)

# Initialize
db = MongoDBClient()
db.connect()
engine = RAGEngine()

# Get random products from different families
all_products = db.get_products(limit=500)
random.seed(42)  # Reproducible
random.shuffle(all_products)

# Select products from different families
selected = []
families_seen = set()
target_count = 5

for p in all_products:
    family = p.get('product_family', '')
    if family and family not in families_seen:
        # Prefer families with docs expected
        if family in ['EPB', 'EAD', 'EPBC', 'CVI3', 'EFD', 'ERS', 'EABS', 'EABC']:
            selected.append(p)
            families_seen.add(family)
    if len(selected) >= target_count:
        break

# Fill if not enough
while len(selected) < target_count:
    p = random.choice(all_products)
    if p not in selected:
        selected.append(p)

# Define fault categories with queries
fault_categories = [
    {"type": "Connectivity", "query_template": "{tool} WiFi bağlantı sorunu"},
    {"type": "Error Code", "query_template": "{tool} E06 error code"},
    {"type": "Calibration", "query_template": "{tool} torque calibration problem"},
    {"type": "Hardware", "query_template": "{tool} motor çalışmıyor"},
    {"type": "Firmware", "query_template": "{tool} firmware update failed"},
]

print("\n📋 SEÇİLEN ÜRÜNLER:")
print("-" * 50)
for i, p in enumerate(selected):
    model = p.get('model_name', 'Unknown')
    part = p.get('part_number', '')
    family = p.get('product_family', '')
    print(f"  {i+1}. {model} ({part}) [{family}]")

print("\n" + "=" * 70)
print("TEST SONUÇLARI")
print("=" * 70)
print()
print("| # | Ürün | Arıza Tipi | Sorgu | Top Doküman | Score | ✓/✗ |")
print("|---|------|------------|-------|-------------|-------|-----|")

results = []
issues = []

for i, (product, fault) in enumerate(zip(selected, fault_categories)):
    model = product.get('model_name', 'Unknown')
    part = product.get('part_number', '')
    family = product.get('product_family', '')
    fault_type = fault['type']
    query = fault['query_template'].format(tool=model)
    
    # Retrieve context
    start = time.time()
    try:
        context = engine.retrieve_context(
            query=f"{model} {query}",
            part_number=part
        )
        docs = context.get('documents', [])
        elapsed = time.time() - start
    except Exception as e:
        docs = []
        elapsed = 0
        issues.append(f"Test {i+1}: Exception - {str(e)[:50]}")
    
    # Evaluate
    if not docs:
        status = "❌"
        top_doc = "NO DOCS"
        score = 0
        doc_family = "-"
        issues.append(f"Test {i+1}: No documents returned for {model}")
    else:
        top = docs[0]
        top_doc = top['metadata'].get('source', 'Unknown')[:30]
        doc_family = top['metadata'].get('product_family', 'UNK')
        score = top.get('boosted_score', top.get('similarity', 0))
        
        # Check for cross-product contamination
        is_esde = 'ESDE' in top_doc.upper()
        family_match = doc_family in [family, 'GENERAL', 'UNKNOWN', 'CVI3', 'CONNECT', 'CVIC', 'CVIR']
        
        if family_match or is_esde:
            status = "✅"
        else:
            status = "❌"
            issues.append(f"Test {i+1}: Cross-product - expected {family}, got {doc_family}")
    
    results.append(status == "✅")
    
    print(f"| {i+1} | {model[:12]} | {fault_type[:11]} | {query[:25]}... | [{doc_family}] {top_doc[:20]}... | {score:.2f} | {status} |")

# Summary
passed = sum(results)
total = len(results)
rate = (passed / total) * 100

print()
print("=" * 70)
print("📊 ÖZET")
print("=" * 70)
print(f"Başarı Oranı: {passed}/{total} ({rate:.0f}%)")
print()

if issues:
    print("🔴 TESPİT EDİLEN SORUNLAR:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("🟢 Tüm testler başarılı!")

print()
print("📝 İYİLEŞTİRME ÖNERİLERİ:")
if rate < 100:
    print("  1. BLRTC/ELC gibi nadir ürünler için döküman eklenmeli")
    print("  2. Türkçe terimler için query expansion geliştirilmeli")
    print("  3. Hardware fault için spesifik döküman tagging gerekli")
else:
    print("  - Sistem optimal çalışıyor")

print()
print("=" * 70)
print("TEST TAMAMLANDI")
print("=" * 70)
