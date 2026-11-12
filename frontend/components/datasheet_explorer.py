"""Datasheet Evidence Explorer Component for CircuitMind AI Streamlit UI.

Displays grounded datasheet evidence passages, PDF source citations, page numbers,
RRF relevance scores, and provides direct RAG search testing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from frontend.api_client import CircuitMindAPIClient
from frontend.styles import citation_badge


def render_datasheet_explorer(
    analysis_data: Optional[Dict[str, Any]] = None,
    api_client: Optional[CircuitMindAPIClient] = None,
    target_components: Optional[List[str]] = None,
) -> None:
    """Render datasheet evidence passages and interactive search tool.

    Args:
        analysis_data: Optional full AnalyzeResponse dictionary from FastAPI.
                       If None, falls back to st.session_state["latest_analysis"].
        api_client: CircuitMindAPIClient instance for live retrieval requests.
        target_components: Optional active component scope.
    """
    if analysis_data is None:
        analysis_data = st.session_state.get("latest_analysis")

    if api_client is None:
        api_client = st.session_state.get("api_client") or CircuitMindAPIClient()

    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 0;">
                🔍 GROUNDED DATASHEET EVIDENCE EXPLORER
            </h3>
            <p style="font-size: 0.82rem; color: #94A3B8; margin: 2px 0 0 0;">
                Inspect verifiable technical evidence retrieved directly from ingested PDF component datasheets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    docs = analysis_data.get("retrieved_documents", []) if analysis_data else []

    col_view, col_search = st.columns([3, 2])

    with col_view:
        st.markdown("#### Retrieved Evidence Passages for Active Analysis")
        if not docs:
            st.info("No datasheet passages in current context. Run an analysis in the Engineering Workspace or use the search panel on the right.")
        else:
            st.caption(f"Showing {len(docs)} verified passages ordered by Reciprocal Rank Fusion (RRF):")
            for idx, doc in enumerate(docs, 1):
                src = doc.get("source", "datasheet.pdf")
                page = doc.get("page", 1)
                chunk_id = doc.get("chunk_id", "N/A")
                score = doc.get("rrf_score")
                content = doc.get("content", "")

                score_str = f"RRF Score: {score:.5f}" if score is not None else "Dense Match"

                with st.container():
                    st.markdown(
                        f"""
                        <div class="tech-card">
                            <div class="tech-card-header">
                                <div>
                                    <span style="font-weight: 700; color: #38BDF8; margin-right: 8px;">#{idx}</span>
                                    {citation_badge(src, page)}
                                    <span style="font-size: 0.72rem; color: #64748B; margin-left: 6px; font-family: monospace;">ID: {chunk_id}</span>
                                </div>
                                <div>
                                    <span class="badge badge-info">{score_str}</span>
                                </div>
                            </div>
                            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; font-family: ui-monospace, monospace; background: #0B0F19; padding: 10px; border-radius: 4px; border: 1px solid #1E293B;">
                                {content}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with col_search:
        st.markdown("#### Direct Datasheet Retrieval Tool")
        st.caption("Probe vector store and BM25 directly via POST /api/v1/retrieve without LLM synthesis:")

        search_query = st.text_input(
            "Datasheet Search Query",
            placeholder="e.g. LM35 supply voltage, quiescent current, layout",
            key="direct_retrieval_query",
        )
        col_k, col_strat = st.columns([1, 1])
        with col_k:
            k_val = st.number_input("Passages (k)", min_value=1, max_value=10, value=3)
        with col_strat:
            strat_val = st.selectbox("Strategy", options=["hybrid", "semantic", "mmr", "keyword"], index=0)

        if st.button("🔎 Execute Pure Retrieval", use_container_width=True):
            if not search_query.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Retrieving grounded datasheet passages..."):
                    try:
                        ret_res = api_client.retrieve_passages(
                            query=search_query,
                            k=k_val,
                            strategy=strat_val,
                            target_components=target_components,
                        )
                        st.session_state["direct_retrieval_results"] = {
                            "query": search_query,
                            "strategy": strat_val,
                            "k": k_val,
                            "documents": ret_res.get("documents", []),
                            "retrieval_time_ms": ret_res.get("retrieval_time_ms", 0.0),
                        }
                    except Exception as exc:
                        st.error(f"Retrieval error: {exc}")

        # Render Direct Retrieval Results from Session State
        direct_data = st.session_state.get("direct_retrieval_results")
        if direct_data:
            found_docs = direct_data.get("documents", [])
            s_query = direct_data.get("query", "")
            s_strat = str(direct_data.get("strategy", "hybrid")).upper()
            time_ms = direct_data.get("retrieval_time_ms", 0.0)

            st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 12px 0;' />", unsafe_allow_html=True)
            st.caption(f"Direct Search Results for '{s_query}' ({len(found_docs)} passages via {s_strat}, {time_ms:,.1f} ms):")
            for d in found_docs:
                st.markdown(
                    f"""
                    <div style="background: #111827; padding: 10px; border-radius: 6px; border: 1px solid #1E293B; margin-bottom: 8px;">
                        <div style="margin-bottom: 6px;">{citation_badge(d.get('source'), d.get('page'))}</div>
                        <div style="font-size: 0.78rem; color: #CBD5E1; font-family: monospace; line-height: 1.4;">
                            {d.get('content')[:300]}...
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No direct retrieval queries executed yet. Submit a query above to test direct datasheet retrieval.")

