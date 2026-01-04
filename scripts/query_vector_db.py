from src.vectordb.chroma_client import ChromaDBClient
from src.documents.embeddings import EmbeddingsGenerator

embeddings = EmbeddingsGenerator()
chroma = ChromaDBClient()

query = 'CVI3'
query_embedding = embeddings.generate_embedding(query)
results = chroma.query(query_text=query, query_embedding=query_embedding, n_results=3, where={'doc_type': 'support_ticket'})
documents = results.get('documents', [[]])[0]
metadatas = results.get('metadatas', [[]])[0]
distances = results.get('distances', [[]])[0]

print(f"\n✅ Found {len(documents)} results:\n")
for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
    score = 1 - dist
    print(f'--- Result {i+1} (similarity: {score:.3f}) ---')
    print(f'📋 Ticket: {meta.get("ticket_id")}')
    print(f'Title: {meta.get("title", "")[:60]}')
    print(f'Content: {doc[:300]}...')
    print()
