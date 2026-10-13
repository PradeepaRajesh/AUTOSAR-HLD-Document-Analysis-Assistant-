# Evaluation summary
| Metric | Result |
|---|---|
| Questions (answerable / out-of-scope) | 8 / 2 |
| Retrieval hit rate (expected page in top-5) | 8/8 |
| Keyword match (proxy for answer correctness) | 6/8 |
| Groundedness (answers with valid citations) | 8/8 |
| Correct refusals on out-of-scope questions | 2/2 |
| Wrongful refusals on answerable questions | 0/8 |
| Average latency | 6.4s |

Config: chunk=300 words, overlap=50, top-k=5,
threshold=0.55, embeddings=BAAI/bge-small-en-v1.5, LLM=openai/gpt-oss-20b.
