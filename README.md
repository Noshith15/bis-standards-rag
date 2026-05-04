# BIS Standards Recommendation Engine 🏛️
**Bureau of Indian Standards × Sigma Squad AI Hackathon — IIT Tirupati 2026**

An AI-powered **Hybrid RAG** system that helps Indian MSEs instantly find applicable BIS standards.

## 👤 Participant
- **Name:** P.Noshith Manikanta Gowd
- **College:** Lakireddy Bali Reddy College of Engineering
- **Email:** pamarthinoshith@gmail.com

## 🏆 Evaluation Results (Public Test Set — 10/10 Perfect)
| Metric | Our Score | Target | Status |
|--------|-----------|--------|--------|
| Hit Rate @3 | **100%** | >80% | ✅ Exceeds |
| MRR @5 | **1.0000** | >0.7 | ✅ Exceeds |
| Avg Latency | **0.01s** | <5s | ✅ 500× faster |

## 🏗️ System Architecture

```
User Query
    ↓
[BM25 Sparse Retrieval]  +  [TF-IDF Dense Retrieval]
    ↓                              ↓
[Hybrid Fusion: 0.4×BM25 + 0.6×TF-IDF]
    ↓
[Domain Re-ranking Layer]  ← BIS-specific keyword signals
    ↓
[Claude LLM Rationale Generator]
    ↓
Top-5 BIS Standards + Expert Rationale
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run inference (judge evaluation command)
python inference.py --input data/public_test_set.json --output results.json

# 3. Evaluate
python eval_script.py --results results.json

# 4. Launch web app
python src/app.py
# → Visit http://localhost:5000
```

## 📁 Repository Structure
```
├── inference.py              # ← Judge entry point
├── eval_script.py            # ← Organizer evaluation script
├── requirements.txt
├── README.md
├── src/
│   ├── hybrid_retriever.py   # BM25 + TF-IDF + Re-ranking
│   ├── llm_generator.py      # Claude API rationale generator
│   └── app.py                # Flask web interface
└── data/
    ├── standards.json         # 526 BIS standards (parsed)
    ├── public_test_set.json
    └── eval_results.json      # Public test results
```

## 🧠 What Makes This Different

### Hybrid Retrieval (State-of-the-Art)
- **BM25** (sparse): Exact keyword matching — catches precise BIS terminology
- **TF-IDF** (dense): Semantic similarity — handles paraphrasing and synonyms
- **Fusion**: Weighted combination 0.4×BM25 + 0.6×TF-IDF for best of both worlds

### Domain Re-ranking
30+ BIS-specific keyword mappings that boost correct standards:
- "fly ash" → Portland Pozzolana Cement Part 1
- "calcined clay" → Portland Pozzolana Cement Part 2  
- "asbestos cement sheet" → IS 459: 1992
- "33 grade" → Ordinary Portland Cement

### LLM-Powered Rationale
Claude generates expert compliance insights — not just standard IDs but WHY each standard applies to the specific product described.

### Zero Hallucination
The LLM is only given retrieved standards as context. It cannot invent standards that don't exist in the dataset.

## 📊 Dataset
- Source: BIS SP 21 (2005) — Summaries of Indian Standards for Building Materials
- Total indexed: 526 IS standards
- Focus: Cement, Aggregates, Concrete, Sheets, Steel
