# Advanced AI/ML Text & Document Analytics Platform

A production-ready Enterprise Document Parsing and Natural Language Processing (NLP) Dashboard built using pure Python and Flask. This system extracts raw textual streams from heterogeneous corporate payloads (PDF, DOCX, TXT) and executes real-time statistical density modeling and heuristic summarization without relying on external heavy NLP packages.

## 🚀 Key Features
* **Multi-Format Ingestion Engine**: Automated text extraction from complex `.pdf`, `.docx`, and `.txt` architectures.
* **Lexical Stopwords Matrix**: Custom pure-python pipeline for clean tokenization and term frequency profiling.
* **Heuristic Extractive Summarizer**: An advanced scoring system that isolates top-density sentences to generate clean 3-line summaries.
* **JSON Analytical Reports**: Every processed document automatically dumps a structured metadata and data analysis payload inside the `reports/` file system.

## 🛠️ Project Structure
```text
├── aimlfinalproject.py   # Core Production Code (Flask, Parsers, NLP Engines)
├── README.md              # Project Documentation & Architecture Blueprint
├── dashboard.png          # Visual Platform Reference (Screenshot)
├── uploads/               # Dynamic server directory for raw input streams
└── reports/               # Auto-generated JSON feature extraction outputs