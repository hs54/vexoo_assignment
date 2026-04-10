import uuid
from typing import List, Dict

class KnowledgePyramid:
    """
    Implements a 4-layer Knowledge Pyramid for optimized RAG retrieval.
    Architecture: Raw -> Summary -> Theme -> Distilled Keywords
    """
    def __init__(self, raw_text: str, window_size=1500, overlap=400):
        self.raw_text = raw_text
        self.pyramid = []
        self._process_ingestion(window_size, overlap)

    def _get_sliding_window_chunks(self, text: str, size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            if end >= len(text): break
            start += (size - overlap)
        return chunks

    def _process_ingestion(self, size, overlap):
        # Part 1: Sliding Window Implementation
        chunks = self._get_sliding_window_chunks(self.raw_text, size, overlap)
        
        for chunk in chunks:
            # Part 2: Hierarchical Pyramid Construction
            node = {
                "id": str(uuid.uuid4()),
                "layers": {
                    "L1_Raw": chunk,
                    "L2_Summary": f"SIMULATED SUMMARY: This section discusses {chunk[:50]}...",
                    "L3_Theme": "Technical Logic / Documentation", # Rule-based tag
                    "L4_Distilled": list(set(chunk.lower().split()[:12])) # Mocked embeddings/keywords
                }
            }
            self.pyramid.append(node)

    def retrieve(self, query: str) -> Dict:
        """
        Simple Semantic Retrieval Simulation
        Uses keyword overlap to find the most relevant chunk across levels.
        """
        query_terms = query.lower().split()
        best_node = None
        max_score = 0
        
        for node in self.pyramid:
            # Scoring logic across raw and distilled layers
            score = sum(1 for term in query_terms if term in node["layers"]["L1_Raw"].lower())
            if score > max_score:
                max_score = score
                best_node = node
        
        return best_node["layers"] if best_node else {"error": "No matches found."}

if __name__ == "__main__":
    # Test Data
    sample_doc = "Artificial Intelligence is transforming industries. Sliding windows help preserve context. " * 50
    engine = KnowledgePyramid(sample_doc)
    
    print("--- RAG Retrieval Test ---")
    result = engine.retrieve("What is the sliding window strategy?")
    print(f"Top Layer (Summary): {result['L2_Summary']}")
    print(f"Keywords (Distilled): {result['L4_Distilled']}")
