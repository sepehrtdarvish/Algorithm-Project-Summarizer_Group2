# Algorithmic Summarization Engine (Track 3)

[cite_start]**Course:** Algorithm Design - Fall 2025-2026 [cite: 8]  
**University:** K.N. [cite_start]Toosi University of Technology [cite: 5]  
[cite_start]**Instructor:** Dr. Pishgoo [cite: 2]  

---

## 👥 Group Members
* **Mohammad Shojaei**
* **Sepehr Darvishi**

[cite_start][cite: 285]

---

## 📖 Project Overview
[cite_start]This repository contains the implementation of the **Algorithmic Summarization Engine** (Track 3), designed as the final project for the Algorithm Design course[cite: 216].

[cite_start]The goal is to build a **Hybrid Summarization System** that combines the statistical precision of classical graph algorithms with the semantic understanding of Large Language Models (LLMs)[cite: 310].

### 🎯 Core Objectives
1.  [cite_start]**Classical Approach:** Implement **TextRank** (a graph-based ranking algorithm) to identify key sentences based on statistical relationships ($TF-IDF$ & Cosine Similarity)[cite: 216, 338].
2.  [cite_start]**LLM Integration:** Use a Large Language Model as an **Oracle** to provide semantic "ground truth" summaries[cite: 333].
3.  [cite_start]**Hybrid Merge:** Design a merging algorithm to re-rank sentences by combining their statistical scores with their semantic value[cite: 387].

---

## ⚙️ Planned Architecture

[cite_start]The project is divided into three main computational modules[cite: 336]:

### 1. The Classical Engine (TextRank)
* **Method:** Unsupervised Graph-based Ranking.
* **Process:** Vectorization $\rightarrow$ Similarity Matrix $\rightarrow$ PageRank.
* [cite_start]**Complexity:** $O(N^2)$ due to dense graph construction[cite: 428].

### 2. The LLM Oracle
* **Role:** Acts as a consultant, not the sole writer.
* [cite_start]**Function:** Generates a high-quality abstractive summary to be used for semantic validation[cite: 372].

### 3. The Merger
* **Logic:** A linear combination of statistical importance and semantic similarity.
* [cite_start]**Formula:** $Score = (\alpha \times TextRank) + (\beta \times SemanticSimilarity)$[cite: 382].

---

## 📂 Repository Structure (Initial)

```text
Algorithm-Project-Summarization/
├── data/               # Placeholder for input text files
├── src/                # Source code (to be implemented)
├── docs/               # Analysis reports and diagrams
├── tests/              # Unit tests
└── README.md           # Project documentation