# Vexoo Assignment

## Overview
This repository contains a dual-part solution focused on **Advanced RAG Ingestion** and **LLM Fine-tuning**. 

### Part 1: Sliding Window + Knowledge Pyramid
- **Sliding Window:** Implemented with a 25% overlap to ensure no context is lost at chunk boundaries.
- **Knowledge Pyramid:** A 4-layer hierarchical data structure (Raw -> Summary -> Theme -> Keywords) for multi-granular retrieval.

### Part 2: LLaMA 3.2 Fine-Tuning
- **Task:** Mathematical reasoning on the GSM8K dataset.
- **Optimization:** Utilized **LoRA (Low-Rank Adaptation)** to reduce trainable parameters, enabling fine-tuning of LLaMA 3.2 1B on consumer-grade GPUs.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

Run RAG Ingestion Test:
python ingestion.py

Run Fine-Tuning Script:
python finetune.py
