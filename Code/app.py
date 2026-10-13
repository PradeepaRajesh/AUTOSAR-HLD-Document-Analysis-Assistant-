"""Streamlit UI.  Run:  streamlit run Code/app.py"""
import pandas as pd
import streamlit as st

import config
import ingest
import rag

st.set_page_config(page_title="AUTOSAR HLD Assistant", layout="wide")
st.title("AUTOSAR HLD Document Analysis Assistant")
st.caption("Local RAG: answers are grounded in the uploaded document, cited by page, and reviewed by an engineer.")

# ---------------- Sidebar: knowledge base ----------------
with st.sidebar:
    st.header("1. Knowledge base")
    up = st.file_uploader("Upload an AUTOSAR PDF", type="pdf")
    if up and st.button("Ingest document"):
        config.INPUT_DIR.mkdir(exist_ok=True)
        path = config.INPUT_DIR / up.name
        path.write_bytes(up.getbuffer())
        with st.spinner("Parsing, embedding and indexing..."):
            n = ingest.ingest_pdfs([str(path)])
        st.success(f"Indexed {n} chunks from {up.name}")
    st.divider()
    st.caption(f"LLM backend: **{config.LLM_BACKEND}** "
               f"({config.OLLAMA_MODEL if config.LLM_BACKEND == 'ollama' else config.API_MODEL})")
    st.caption(f"Embeddings: {config.EMBED_MODEL}")
    st.caption(f"Top-k: {config.TOP_K} | Refusal threshold: {config.THRESHOLD}")

if not rag.collection_ready():
    st.info("No document indexed yet. Upload a PDF in the sidebar (or run `python Code/ingest.py <pdf>`).")
    st.stop()

tab_qa, tab_comp, tab_audit = st.tabs(["Ask the document", "Component inventory", "Audit log"])

# ---------------- Tab 1: Q&A ----------------
with tab_qa:
    q = st.text_input("Ask a question about the document",
                      placeholder="e.g. What is the role of the RTE?")
    if st.button("Ask") and q.strip():
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            st.session_state["result"] = rag.answer_question(q.strip())

    res = st.session_state.get("result")
    if res:
        st.subheader("Answer")
        if res["refused"]:
            st.warning(res["answer"])
        else:
            st.write(res["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", res["confidence"].split(" ")[0])
        c2.metric("Valid citations", {True: "Yes", False: "No", None: "n/a"}[res["citations_valid"]])
        c3.metric("Latency", f"{res['latency']:.1f}s")
        if res["confidence"].startswith("Low (no valid"):
            st.error("The answer's citations do not match the retrieved pages. Verify manually.")

        with st.expander("Sources used (retrieved evidence)"):
            for h in res["hits"]:
                st.markdown(f"**Page {h['page']}**: {h['source']} (distance {h['distance']:.2f})")
                st.text(h["text"][:500] + ("..." if len(h["text"]) > 500 else ""))

        st.markdown("#### Engineer review (human oversight)")
        verdict = st.radio("Decision", ["Pending", "Accept", "Edit needed", "Reject"], horizontal=True)
        if st.button("Save review"):
            rag.log_review(res["question"], res["answer"], res["citations"], verdict)
            st.success("Saved to audit log")

# ---------------- Tab 2: Component extractor ----------------
with tab_comp:
    st.write("Extracts software components, ports and interfaces from the most relevant sections.")
    if st.button("Extract component inventory"):
        with st.spinner("Retrieving and extracting..."):
            st.session_state["components"] = rag.extract_components()
    comps = st.session_state.get("components")
    if comps is not None:
        if not comps:
            st.warning("No components found in the retrieved sections.")
        else:
            df = pd.DataFrame(comps)
            for col in ("ports", "interfaces"):
                if col in df:
                    df[col] = df[col].apply(lambda v: ", ".join(v) if isinstance(v, list) else v)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "component_inventory.csv", "text/csv")
            st.caption("AI-extracted: must be reviewed against the source pages before use.")

# ---------------- Tab 3: Audit log ----------------
with tab_audit:
    if config.AUDIT_LOG.exists():
        st.dataframe(pd.read_csv(config.AUDIT_LOG), use_container_width=True)
    else:
        st.info("No reviews saved yet.")
