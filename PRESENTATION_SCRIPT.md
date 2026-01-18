# Desoutter Repair Assistant - Presentation Script
## Teams Meeting Speaking Notes

---

## SLIDE 1: Title

**[Speaking]**

"Good morning/afternoon everyone. Today I'm going to present the Desoutter Repair Assistant - an AI-powered technical support system that I've been developing.

My name is Fatih Bayram, and I've spent the last 14 years working as a service technician."

---

## SLIDE 2: The Problem

**[Speaking]**

"Let me start with WHY I built this.

Since 2011, I've worked in technical service departments. The biggest challenge I faced every day was simple: **finding the right answer takes too long**.

When a technician encounters a fault, they need to:
- Search through PDF manuals
- Look at service bulletins
- Check old support tickets
- Sometimes ask senior colleagues

This process can take 15 to 30 minutes for a complex fault. And often, different technicians give different answers depending on their experience.

The real problem? When experienced technicians leave the company, their knowledge leaves with them. We lose years of expertise overnight."

---

## SLIDE 3: The Vision

**[Speaking]**

"So I asked myself: What if we had an AI assistant that could answer these questions instantly?

But here's the important part - I didn't want to use cloud services like ChatGPT or Claude API. Why?

Because our technical documentation contains sensitive information. Product designs, error codes, internal procedures - this data should NOT be sent to third-party servers.

So the vision became: **Build a self-hosted AI that keeps all data on our own servers.**

Three core principles:
1. Data stays on-premise - nothing leaves our network
2. Self-hosted LLM - no monthly API costs, no data leakage
3. The system learns from feedback - it gets smarter over time"

---

## SLIDE 4: Infrastructure

**[Speaking]**

"To make this work, I built a dedicated AI server from scratch.

It runs on Proxmox - that's a virtualization platform - with an NVIDIA RTX A2000 GPU. This GPU has 6 gigabytes of memory, which is enough to run a 7 billion parameter language model locally.

The key difference from cloud AI:

With OpenAI or Claude, you pay per token - roughly 500 to 1000 dollars per month for heavy usage. And your data goes to their servers.

With our self-hosted solution, after the initial hardware investment, the running cost is essentially zero. And all data stays in-house.

The server can process 40 to 50 tokens per second, which means response times of 8 to 12 seconds for a typical query."

---

## SLIDE 5: How It Works - RAG Pipeline

**[Speaking]**

"Now let me explain how the system actually works. It uses something called RAG - Retrieval Augmented Generation.

When a technician asks a question like 'Motor makes grinding noise', here's what happens:

First, **Hybrid Search** - the system searches our document database using two methods: semantic search that understands meaning, and keyword search for exact terms.

Second, **Product Filter** - it only retrieves documents relevant to the specific product. If you're asking about an EADC tool, you won't get results from EPB manuals.

Third, **LLM Generation** - the retrieved documents are sent to our local language model, which generates a human-readable answer.

Fourth, **Validation** - before showing the response, we validate it to prevent hallucinations - made-up information.

The result is an answer with a confidence score and source citations."

---

## SLIDE 6: Hybrid Search

**[Speaking]**

"Let me go deeper into the hybrid search, because this is what makes our retrieval accurate.

Traditional search has a problem. Semantic search understands meaning - it knows 'motor failure' and 'engine not working' mean similar things. But it might miss exact error codes.

Keyword search finds exact terms like 'E804' perfectly, but doesn't understand context.

Our system combines both: 60% semantic, 40% keyword. Then it fuses the results using a technique called Reciprocal Rank Fusion.

The result? 35% better retrieval accuracy compared to semantic search alone.

When a technician searches for 'E804 error', the system finds both documents that mention E804 exactly AND documents that discuss similar error conditions."

---

## SLIDE 7: Semantic Chunking

**[Speaking]**

"Before we can search documents, we need to process them. This is called chunking.

We take a 100-page PDF manual and break it into smaller pieces - chunks of about 500 tokens each. But we do this intelligently.

Each chunk includes metadata:
- Which product family it belongs to
- What type of document it is - service bulletin, manual, troubleshooting guide
- Extracted fault keywords
- Page numbers for citations

Currently, we have over 28,000 chunks in our database, from 541 documents.

This metadata is crucial for the next feature..."

---

## SLIDE 8: Product Filtering

**[Speaking]**

"Product filtering solves a critical problem.

Without it, if you ask about an EADC tool, the system might return results from:
- EAD20 manuals - similar name, different product
- EPB battery tools - completely different
- CVI3 controllers - not even a tool

This causes confusion and wrong answers.

Our solution: when you select a product, the system adds a filter to the database query. It ONLY retrieves chunks that match that product family.

Result: we eliminate 90% of irrelevant noise. The technician gets answers specific to their actual product."

---

## SLIDE 9: Self-Learning

**[Speaking]**

"This is probably the most exciting feature. The system learns from user feedback.

After every response, the technician can click thumbs up or thumbs down.

If they click thumbs up, the system records: 'This document source gave a good answer for this type of question.' It increases the source's score.

If they click thumbs down, the opposite happens. The source gets penalized for future similar queries.

We use something called Wilson Score - a statistical method that balances positive and negative feedback, accounting for sample size.

Over time, the system learns which documents are most reliable for which types of faults. It essentially learns from the collective experience of all technicians using it."

---

## SLIDE 10: Hallucination Prevention

**[Speaking]**

"AI hallucination is a real risk. The model might make up information that sounds correct but isn't.

We have multiple layers of protection:

First, Context Grounding - we check if the answer actually comes from the retrieved documents. If the model invents information not in the sources, we detect it.

Second, Response Validation - we scan for forbidden patterns, like the model saying 'contact support' instead of giving a real answer.

Third, Confidence Scoring - we calculate a score based on multiple factors. Low score means low reliability.

If the system isn't confident, it simply says: 'I don't have enough information to answer this question.'

Our test suite shows less than 2% hallucination rate."

---

## SLIDE 11: Response Caching

**[Speaking]**

"Performance matters. Technicians don't want to wait.

A fresh query takes 8 to 12 seconds because it needs to:
- Search the database
- Retrieve documents
- Send to the LLM
- Generate response

But here's the thing - many questions are asked repeatedly. 'What is error E804?' might be asked 50 times a month.

So we cache responses. When the same or very similar question is asked again, we return the cached answer instantly - less than 1 millisecond.

That's a speedup of about 100,000 times.

Our cache uses LRU - Least Recently Used - eviction with a one-hour TTL. So answers stay fresh."

---

## SLIDE 12: Intent Detection

**[Speaking]**

"Not all questions are the same. Someone asking 'What is error E804?' needs different handling than 'How do I calibrate this tool?'

We detect 8 different intent types:
- Troubleshooting - fault diagnosis
- Error codes - specific error lookup
- Specifications - technical specs
- Calibration - calibration procedures
- Maintenance - service intervals

Each intent gets a specialized prompt template. This improves answer quality because the LLM knows what type of response is expected."

---

## SLIDE 13: Web Interface - Technician View

**[Speaking]**

"Let me show you the user interface. We built a React-based web application that technicians use daily.

The technician workflow is a simple 4-step wizard:

**Step 1: Product Search**
The technician types the product name or part number. They can filter by series, tool type, or wireless capability. The system shows matching products from our database of 451 tools.

**Step 2: Describe the Fault**
They select their language - we support English and Turkish - and type a description of the problem. For example: 'Motor makes grinding noise when starting.'

**Step 3: AI Response**
Within 8-12 seconds, they get a response with:
- A confidence score showing how reliable the answer is
- Step-by-step diagnosis or solution
- Source citations with page numbers - so they can verify in the original manual if needed

**Step 4: Feedback**
Two simple buttons: thumbs up or thumbs down. This feedback feeds into our self-learning system, improving future responses."

---

## SLIDE 14: Web Interface - Admin Dashboard

**[Speaking]**

"For administrators, we have a comprehensive dashboard.

The dashboard shows:
- **System Stats**: Total products, documents, and chunks in the database
- **Performance Metrics**: Average response time, cache hit rate, test pass rate
- **User Management**: Create and delete users, assign roles
- **Document Management**: Upload new PDFs or Word documents, trigger re-ingestion
- **Learning Insights**: See which sources perform best, view feedback statistics
- **Cache Control**: View cache entries, clear cache when needed

We have two user roles:
- **Technician**: Can query the system and submit feedback
- **Admin**: Full access including user management, document uploads, and system configuration

This role separation ensures technicians have a simple, focused interface while admins can manage the entire system."

---

## SLIDE 15-16: System Metrics

**[Speaking]**

"Let me share some numbers from our current production system.

We have a test suite with 25 standard queries covering different scenarios. Our pass rate is 96% - 24 out of 25 queries return correct, helpful answers.

The database contains:
- 451 products - both tools and controllers
- 28,414 document chunks
- 541 source documents
- Over 2,200 Freshdesk tickets that we've ingested

The LLM is Qwen 2.5 with 7 billion parameters, running locally on our GPU. Embeddings use the MiniLM model with 384 dimensions."

---

## SLIDE 14: Live Demo

**[Speaking]**

"Now let me show you how this works in practice.

*[Switch to browser/demo]*

I'll demonstrate three scenarios:

First, a basic troubleshooting query: 'Motor makes grinding noise' for an EADC tool.

*[Show query and response]*

Notice the confidence score and the source citations at the bottom.

Second, an error code lookup: 'What is error E804?'

*[Show query and response]*

The system finds the exact error code definition.

Third, a Turkish query to show multi-language support: 'Alet çalışmıyor'

*[Show query and response]*

Finally, I'll show the feedback buttons. When I click thumbs up, this information is recorded for the self-learning engine.

*[Click feedback button]*"

---

## SLIDE 15: Summary & Thank You

**[Speaking]**

"To summarize what we've built:

Hybrid Search gives us 35% better retrieval accuracy.
Product Filtering eliminates 90% of irrelevant noise.
Self-Learning means the system improves over time from real usage.
Hallucination Prevention keeps our error rate below 2%.
Response Caching provides 100,000x speedup for repeated queries.
And everything runs on-premise - full data control, no cloud dependency.

The impact? What used to take 15 to 30 minutes of manual searching now takes seconds. And the quality is consistent regardless of who's asking.

Thank you for your attention. I'm happy to take any questions.

The project documentation is available on GitHub - I'll share the links in the chat."

---

## Q&A Preparation

### Likely Questions and Answers:

**Q: What happens if the AI gives a wrong answer?**
A: "The confidence score indicates reliability. Low scores mean the technician should verify. Plus, the feedback system lets users report bad answers, which improves future responses."

**Q: How long did this take to build?**
A: "The core system took about 3-4 months of development. Document ingestion and tuning is ongoing."

**Q: Can this work for other products/companies?**
A: "Yes, the architecture is generic. You would need to ingest your own documentation and possibly adjust the domain vocabulary."

**Q: What about updates to manuals?**
A: "We can re-ingest documents anytime. The admin panel has document upload and re-indexing features."

**Q: Is it secure?**
A: "Yes. All data stays on-premise. External access uses Cloudflare Tunnel with zero open ports. Authentication is JWT-based with role separation."

**Q: What's the hardware cost?**
A: "The GPU is the main expense - an RTX A2000 costs around $400-500. Total server setup under $2000 for a capable system."

---

## Time Estimates

| Section | Duration |
|---------|----------|
| Introduction (Slides 1-3) | 3 min |
| Infrastructure (Slide 4) | 2 min |
| Technical Features (Slides 5-12) | 8 min |
| Web Interface (Slides 13-14) | 3 min |
| Metrics (Slide 15) | 1 min |
| Live Demo (Slide 16) | 5 min |
| Summary & Q&A (Slide 17) | 3 min |
| **Total** | **~25 min** |

---

## Before the Presentation

Checklist:
- [ ] Test the live demo system is working
- [ ] Have backup screenshots in case demo fails
- [ ] Prepare GitHub links to paste in chat
- [ ] Test screen sharing in Teams
- [ ] Close unnecessary applications
- [ ] Mute notifications
