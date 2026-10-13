"""Automated evaluation.  Run:  python Code/evaluate.py

Reads  Evaluation_Results/eval_questions.csv  with columns:
  id, question, answerable (Y/N), expected_page (e.g. 12 or 12;13), expected_keywords (a;b;c)
Writes Evaluation_Results/eval_results.csv and eval_summary.md

Automatic metrics are proxies. After the run, open eval_results.csv and fill the
'answer_correct_manual' column yourself (Y/N) by checking each answer against the PDF.
"""
import csv
import time
import config
import rag

IN = config.EVAL_DIR / "eval_questions.csv"
OUT = config.EVAL_DIR / "eval_results.csv"
SUMMARY = config.EVAL_DIR / "eval_summary.md"


def split(v):
    return [x.strip() for x in (v or "").split(";") if x.strip()]


def main():
    rows = list(csv.DictReader(open(IN, encoding="utf-8")))
    results = []
    for r in rows:
        res = rag.answer_question(r["question"])
        answerable = r["answerable"].strip().upper() == "Y"
        retrieved_pages = [h["page"] for h in res["hits"]]
        exp_pages = [int(p) for p in split(r["expected_page"]) if p.isdigit()]
        hit = (any(p in retrieved_pages for p in exp_pages) if (answerable and exp_pages) else None)
        kws = [k.lower() for k in split(r["expected_keywords"])]
        kw_ok = (any(k in res["answer"].lower() for k in kws) if (answerable and kws and not res["refused"]) else None)
        results.append({
            "id": r["id"], "question": r["question"], "answerable": "Y" if answerable else "N",
            "expected_page": r["expected_page"], "retrieved_pages": ";".join(map(str, retrieved_pages)),
            "top_distance": round(res["hits"][0]["distance"], 3) if res["hits"] else "",
            "retrieval_hit": hit, "refused": res["refused"], "citations_valid": res["citations_valid"],
            "keyword_match": kw_ok, "latency_s": round(res["latency"], 1),
            "answer": res["answer"].replace("\n", " "), "answer_correct_manual": ""})
        print(f"[{r['id']}] hit={hit} refused={res['refused']} dist={results[-1]['top_distance']}")
        time.sleep(4)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    def rate(vals):
        vals = [v for v in vals if v is not None]
        return (sum(bool(v) for v in vals), len(vals))

    ans = [x for x in results if x["answerable"] == "Y"]
    oos = [x for x in results if x["answerable"] == "N"]
    hit_n, hit_d = rate([x["retrieval_hit"] for x in ans])
    kw_n, kw_d = rate([x["keyword_match"] for x in ans])
    cit_n, cit_d = rate([x["citations_valid"] for x in ans if not x["refused"]])
    ref_n, ref_d = rate([x["refused"] for x in oos])
    wrong_ref = sum(1 for x in ans if x["refused"])
    avg_lat = sum(x["latency_s"] for x in results) / len(results)

    md = f"""# Evaluation summary
| Metric | Result |
|---|---|
| Questions (answerable / out-of-scope) | {len(ans)} / {len(oos)} |
| Retrieval hit rate (expected page in top-{config.TOP_K}) | {hit_n}/{hit_d} |
| Keyword match (proxy for answer correctness) | {kw_n}/{kw_d} |
| Groundedness (answers with valid citations) | {cit_n}/{cit_d} |
| Correct refusals on out-of-scope questions | {ref_n}/{ref_d} |
| Wrongful refusals on answerable questions | {wrong_ref}/{len(ans)} |
| Average latency | {avg_lat:.1f}s |

Config: chunk={config.CHUNK_WORDS} words, overlap={config.CHUNK_WORDS - config.CHUNK_STEP}, top-k={config.TOP_K},
threshold={config.THRESHOLD}, embeddings={config.EMBED_MODEL}, LLM={config.OLLAMA_MODEL if config.LLM_BACKEND == 'ollama' else config.API_MODEL}.
"""
    SUMMARY.write_text(md, encoding="utf-8")
    print("\n" + md)
    print("Next: open eval_results.csv and fill 'answer_correct_manual' (Y/N) for each answerable row.")
    print("Tip: if out-of-scope questions are NOT refused, lower THRESHOLD; if good questions ARE refused, raise it.")


if __name__ == "__main__":
    main()
