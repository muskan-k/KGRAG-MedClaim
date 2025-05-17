# KGRAG-MedClaim: A Graph-RAG Approach for Medical Misinformation Detection

This project implements a Graph-RAG pipeline to verify biomedical claims using both structured (knowledge graph) and unstructured (PubMed abstracts) evidence, paired with reasoning-enhanced large language models. Built on top of the [HealthVer](https://github.com/sarrouti/HealthVer) dataset [[Sarrouti et al., 2021]](https://aclanthology.org/2021.emnlp-main.507/).

## Key Features

- **Knowledge Graph Construction**: Extracts ⟨subject, predicate, object⟩ triples from HealthVer evidence using LLM-based prompts, stored in a Neo4j database.
- **PubMed Retrieval**: Uses SciSpaCy + MeSH expansion + BM25 ranking to fetch relevant abstracts from PubMed.
- **RAG + CoT Reasoning**: Fuses KG triples and abstracts in a Retrieval-Augmented Generation setup, with instruction fine-tuned Mistral-7B and hidden scratchpad reasoning.
- **Structured Outputs**: Returns JSON-formatted verdicts (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`) with rationales.

## Experimental Setups

| Setup                        | Accuracy | F1     | Precision | Recall |
|-----------------------------|----------|--------|-----------|--------|
| B0 Majority Class           | 31.8%    | 0.209  | 0.106     | 0.333  |
| B1 BM25 + Zero-Shot         | 33.8%    | 0.225  | 0.297     | 0.311  |
| B2 SFT + KG + IFT           | 38.1%    | 0.328  | 0.533     | 0.351  |
| B3 SFT + KG + CoT           | 47.3%    | 0.450  | 0.560     | 0.444  |
| **B4 SFT + KG + CoT + IFT** | **62.7%**| **0.611**| **0.689** | **0.552** |

To run the script, simply execute `llm_gen.py`. The GitHub codebase contains early-stage experimental code. Once the pipeline was finalized, development transitioned to Google Colab for streamlined execution.

## Sample Input and Output

The following examples demonstrate how the model evaluates factual claims using knowledge graph evidence.

**Input Claim:**
Energy drinks reduce the risk of heart attacks.

**Model Output:**
```json
{
  "verdict": "REFUTED",
  "justification": "The knowledge graph evidence does not contain any triple that supports the claim that energy drinks reduce the risk of heart attacks."
}
```

---

**Input Claim:**
The novel coronavirus can survive on surfaces.

**Model Output:**
```json
{
  "verdict": "SUPPORTED",
  "justification": "The provided knowledge graph evidence includes a statement that human coronaviruses can survive on surfaces."
}
```
