# Desoutter Assistant - Architecture Diagrams

**Visual representations for presentations and documentation**

---

## Table of Contents

1. [Complete System Architecture](#1-complete-system-architecture)
2. [RAG Pipeline Flow](#2-rag-pipeline-flow)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Self-Learning Loop](#4-self-learning-loop)
5. [Deployment Architecture](#5-deployment-architecture)
6. [Component Interaction Diagram](#6-component-interaction-diagram)

---

## 1. Complete System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[React Frontend<br/>Port 3001]
        B[FastAPI REST API<br/>Port 8000]
    end

    subgraph "Authentication & Security"
        C[JWT Auth<br/>PyJWT + Bcrypt]
    end

    subgraph "RAG Engine Core"
        D[Query Processor<br/>Language Detection<br/>Intent Classification]
        E[Hybrid Search<br/>Semantic 60%<br/>BM25 40%<br/>RRF Fusion]
        F[Context Optimizer<br/>8K Token Budget<br/>Deduplication<br/>Bulletin Boost 4.0x]
        G[Ollama LLM<br/>Qwen2.5:7b<br/>GPU Accelerated]
        H[Response Validator<br/>Hallucination Check<br/>Confidence Scoring]
        I[Response Cache<br/>LRU + TTL<br/>100,000x Speedup]
    end

    subgraph "Self-Learning System"
        J[Feedback Engine<br/>Wilson Score Ranking]
        K[Learned Mappings<br/>MongoDB Storage]
    end

    subgraph "Document Processing"
        L[Document Processor<br/>PDF/DOCX/PPTX]
        M[Semantic Chunker<br/>Context-Aware Splitting]
        N[Product Extractor<br/>Pattern Recognition]
        O[Embeddings Generator<br/>all-MiniLM-L6-v2]
    end

    subgraph "Data Layer"
        P[(MongoDB<br/>Users, Feedback, Mappings)]
        Q[(ChromaDB<br/>Vector Embeddings<br/>BM25 Index)]
        R[Ollama Service<br/>Model Storage]
    end

    subgraph "Data Ingestion"
        S[Web Scraper<br/>Products & Tickets]
        T[Document Upload<br/>Manuals & Bulletins]
    end

    A -->|HTTP/REST| B
    B --> C
    C --> D
    D --> I
    I -->|Cache Miss| E
    E --> F
    F --> G
    G --> H
    H -->|Store Response| I
    H -->|User Feedback| J
    J --> K
    K -->|Boost Queries| E

    S --> L
    T --> L
    L --> M
    M --> N
    N --> O
    O --> Q

    E <--> Q
    G <--> R
    B <--> P
    K <--> P

    style D fill:#e1f5ff
    style E fill:#fff9e1
    style G fill:#ffe1f5
    style H fill:#e1ffe1
    style J fill:#f5e1ff
```

---

## 2. RAG Pipeline Flow

```mermaid
flowchart TD
    Start([User Query]) --> Stage1{1. Off-topic<br/>Detection}
    Stage1 -->|Relevant| Stage2[2. Language<br/>Detection<br/>TR/EN]
    Stage1 -->|Irrelevant| Reject[Reject Query]

    Stage2 --> Stage3{3. Cache<br/>Check}
    Stage3 -->|Hit| Return[Return Cached<br/>Response]
    Stage3 -->|Miss| Stage4[4. Self-Learning<br/>Context]

    Stage4 --> Stage5[5. Hybrid Retrieval<br/>Semantic + BM25]
    Stage5 --> Stage6[6. Product<br/>Filtering]
    Stage6 --> Stage7[7. Capability<br/>Filtering]

    Stage7 --> Stage8{8. Context<br/>Grounding}
    Stage8 -->|Insufficient| IDK[I don't know<br/>response]
    Stage8 -->|Sufficient| Stage9[9. Context<br/>Optimization]

    Stage9 --> Stage10[10. Intent<br/>Detection]
    Stage10 --> Stage11[11. LLM<br/>Generation]

    Stage11 --> Stage12{12. Response<br/>Validation}
    Stage12 -->|Invalid| IDK
    Stage12 -->|Valid| Stage13[13. Confidence<br/>Scoring]

    Stage13 --> Stage14[14. Save & Cache]
    Stage14 --> End([AI Response])

    End --> Feedback{User Feedback}
    Feedback -->|👍 Positive| Learn[Learn Mapping]
    Feedback -->|👎 Negative| AntiLearn[Record Anti-Pattern]

    Learn --> Stage4
    AntiLearn --> Stage4

    style Stage5 fill:#fff9e1
    style Stage11 fill:#ffe1f5
    style Stage12 fill:#e1ffe1
    style Learn fill:#f5e1ff
```

---

## 3. Data Processing Pipeline

```mermaid
flowchart LR
    subgraph Input
        A[PDF Manuals]
        B[DOCX Files]
        C[PPTX Slides]
        D[Web Scraping]
    end

    subgraph Processing
        E[Text Extraction<br/>PyPDF2/pdfplumber/docx]
        F[Semantic Chunking<br/>Context-aware splitting]
        G[Product Extraction<br/>40+ regex patterns]
        H[Metadata Enrichment<br/>doc_type, section, importance]
        I[Embedding Generation<br/>all-MiniLM-L6-v2]
        J[Deduplication<br/>SHA256 hashing]
    end

    subgraph Storage
        K[(ChromaDB<br/>Vector Store)]
        L[(MongoDB<br/>Metadata)]
    end

    A --> E
    B --> E
    C --> E
    D --> E

    E --> F
    F --> G
    G --> H
    H --> I
    I --> J

    J --> K
    J --> L

    style F fill:#e1f5ff
    style I fill:#fff9e1
```

---

## 4. Self-Learning Loop

```mermaid
graph LR
    A[User Query] --> B[RAG Retrieval]
    B --> C[LLM Response]
    C --> D[User Feedback]

    D --> E{Feedback Type}
    E -->|👍 Positive| F[Store Positive<br/>Mapping]
    E -->|👎 Negative| G[Store Negative<br/>Pattern]

    F --> H[Calculate<br/>Wilson Score]
    G --> H

    H --> I[(Learned<br/>Mappings DB)]
    I --> J[Query Boosting]
    J --> B

    style E fill:#f5e1ff
    style H fill:#ffe1f5
    style I fill:#e1ffe1
```

---

## 5. Deployment Architecture

```mermaid
graph TB
    subgraph "Proxmox VM - Ubuntu 22.04 LTS"
        subgraph "Docker Network: ai-net"
            A[MongoDB<br/>Port: 27017<br/>Volume: mongodb_data]
            B[Ollama<br/>Port: 11434<br/>GPU: RTX A2000<br/>Volume: ollama_models]
            C[Desoutter API<br/>Port: 8000<br/>GPU: Shared<br/>Volumes: data, documents, cache]
            D[Frontend<br/>Port: 3001<br/>Nginx Server]
        end

        E[NVIDIA RTX A2000<br/>6GB VRAM]
        F[32GB RAM]
        G[500GB SSD]
    end

    H[Users] -->|HTTPS| D
    D -->|REST API| C
    C --> A
    C --> B

    B -.GPU.-> E
    C -.GPU.-> E

    A -.Storage.-> G
    B -.Storage.-> G
    C -.Storage.-> G

    style B fill:#ffe1f5
    style E fill:#fff9e1
```

---

## 6. Component Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant FastAPI
    participant RAGEngine
    participant Cache
    participant HybridSearch
    participant ChromaDB
    participant Ollama
    participant MongoDB

    User->>Frontend: Submit Query
    Frontend->>FastAPI: POST /diagnose
    FastAPI->>RAGEngine: diagnose()

    RAGEngine->>Cache: Check cache
    alt Cache Hit
        Cache-->>RAGEngine: Return cached response
    else Cache Miss
        RAGEngine->>HybridSearch: hybrid_search()
        HybridSearch->>ChromaDB: Semantic search
        ChromaDB-->>HybridSearch: Top-K vectors
        HybridSearch->>ChromaDB: BM25 search
        ChromaDB-->>HybridSearch: Top-K keywords
        HybridSearch-->>RAGEngine: Fused results (RRF)

        RAGEngine->>Ollama: generate_response()
        Ollama-->>RAGEngine: LLM response

        RAGEngine->>RAGEngine: Validate response
        RAGEngine->>RAGEngine: Calculate confidence

        RAGEngine->>MongoDB: Save diagnosis
        RAGEngine->>Cache: Store in cache
    end

    RAGEngine-->>FastAPI: Response + confidence
    FastAPI-->>Frontend: JSON response
    Frontend-->>User: Display answer

    User->>Frontend: Submit feedback (👍/👎)
    Frontend->>FastAPI: POST /diagnose/feedback
    FastAPI->>MongoDB: Save feedback
    MongoDB->>MongoDB: Update learned mappings
    MongoDB-->>FastAPI: Success
    FastAPI-->>Frontend: Confirmation
    Frontend-->>User: Thank you message
```

---

## Usage Notes

### For Presentations

1. **Copy the Mermaid code** from this document
2. **Paste into:**
   - **Mermaid Live Editor**: https://mermaid.live/
   - **GitHub/GitLab Markdown**: Renders automatically
   - **Notion**: Use `/code` block with `mermaid` language
   - **PowerPoint**: Export as PNG from Mermaid Live
   - **Draw.io**: Import Mermaid syntax

### Customization

To modify colors, add these lines at the end of any diagram:

```mermaid
style NodeName fill:#colorcode
```

Color codes used:
- `#e1f5ff` - Light blue (Query Processing)
- `#fff9e1` - Light yellow (Retrieval)
- `#ffe1f5` - Light pink (Generation)
- `#e1ffe1` - Light green (Validation)
- `#f5e1ff` - Light purple (Learning)

---

## Export Instructions

### Method 1: Mermaid Live Editor
1. Visit https://mermaid.live/
2. Paste diagram code
3. Click "Export" → Choose format (PNG, SVG, PDF)

### Method 2: GitHub README
Simply paste the Mermaid code block into your README.md - GitHub will render it automatically.

### Method 3: VS Code Extension
1. Install "Markdown Preview Mermaid Support" extension
2. Open this file in VS Code
3. Right-click diagram → "Export to PNG/SVG"

---

## Additional Diagrams

### Simplified Overview (For Non-Technical Audiences)

```mermaid
graph LR
    A[User Asks Question] --> B[AI Searches<br/>28,414 Documents]
    B --> C[AI Generates<br/>Answer]
    C --> D[User Receives<br/>Solution + Sources]
    D --> E{Was it helpful?}
    E -->|Yes 👍| F[AI Learns<br/>& Improves]
    E -->|No 👎| G[AI Adjusts<br/>Strategy]
    F --> B
    G --> B

    style B fill:#fff9e1
    style C fill:#ffe1f5
    style F fill:#f5e1ff
```

### Technology Stack Diagram

```mermaid
graph TB
    subgraph "Frontend"
        A[React 18.2]
        B[Vite 5.0]
        C[Axios 1.6]
    end

    subgraph "Backend"
        D[FastAPI 0.109]
        E[Python 3.11]
        F[PyJWT Auth]
    end

    subgraph "AI/ML"
        G[Ollama + Qwen2.5:7b]
        H[LangChain 0.1]
        I[Sentence Transformers 2.2]
    end

    subgraph "Data"
        J[MongoDB 7.0]
        K[ChromaDB 0.4]
        L[BM25 Custom]
    end

    subgraph "Infrastructure"
        M[Docker + Compose]
        N[NVIDIA RTX A2000]
        O[Proxmox VM]
    end

    A --> D
    B --> D
    C --> D

    D --> G
    D --> J
    E --> H

    G --> N
    H --> I
    I --> K

    K --> L

    M --> O

    style G fill:#ffe1f5
    style K fill:#fff9e1
    style N fill:#e1ffe1
```

---

**Last Updated:** January 15, 2026
**Version:** 1.0.0
