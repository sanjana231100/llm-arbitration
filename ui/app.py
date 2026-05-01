import streamlit as st
import httpx
import json
import os

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="LLM Arbitration System",
    page_icon="⚖",
    layout="wide",
)

st.title("LLM Arbitration System")
st.caption("Multi-agent LLM output evaluation with parallel critic dispatch")

tab1, tab2, tab3 = st.tabs(["Verdict view", "Critic comparison", "Batch mode"])


def severity_color(severity: str) -> str:
    return {"high": "#ff4444", "medium": "#ffaa00", "low": "#4488ff"}.get(severity, "#888")


def call_arbitrate(llm_output: str, original_prompt: str | None) -> dict:
    payload = {"llm_output": llm_output}
    if original_prompt:
        payload["original_prompt"] = original_prompt
    with httpx.Client(timeout=120) as client:
        response = client.post(f"{API_BASE}/v1/arbitrate", json=payload)
        response.raise_for_status()
        return response.json()


with tab1:
    st.subheader("Single arbitration")

    original_prompt = st.text_input(
        "Original prompt (optional)",
        placeholder="What question or task generated this output?",
    )
    llm_output = st.text_area(
        "LLM output to arbitrate",
        height=200,
        placeholder="Paste any LLM-generated text here...",
    )

    if st.button("Run arbitration", type="primary"):
        if not llm_output.strip():
            st.error("Please paste an LLM output to arbitrate.")
        else:
            with st.spinner("Running three critics in parallel then adjudicating..."):
                try:
                    result = call_arbitrate(llm_output, original_prompt or None)
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            verdict = result["verdict"]
            st.session_state["last_result"] = result

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall score", f"{verdict['overall_score']}/10")
            col2.metric("Confidence", verdict["confidence"].upper())
            col3.metric("Confirmed issues", len(verdict["confirmed_issues"]))

            st.markdown("---")
            st.subheader("Summary")
            st.write(verdict["summary"])

            if verdict["confirmed_issues"]:
                st.subheader("Confirmed issues")
                for issue in verdict["confirmed_issues"]:
                    color = severity_color(issue["severity"])
                    with st.container():
                        st.markdown(
                            f'<div style="border-left: 4px solid {color}; padding: 8px 16px; margin: 8px 0; background: #1a1a1a;">'
                            f'<strong style="color:{color}">[{issue["severity"].upper()}]</strong> {issue["issue"]}<br>'
                            f'<code style="color:#aaa">"{issue["quote"]}"</code><br>'
                            f'<small style="color:#888">Evidence: {issue["evidence"]}</small><br>'
                            f'<small style="color:#888">Flagged by: {", ".join(issue["flagged_by"])}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            if verdict["dismissed_flags"]:
                with st.expander(f"Dismissed flags ({len(verdict['dismissed_flags'])})"):
                    for flag in verdict["dismissed_flags"]:
                        st.markdown(
                            f'**{flag["flagged_by"]}** flagged: "{flag["quote"]}"  \n'
                            f'Dismissed because: {flag["dismissal_reason"]}'
                        )

            st.markdown("---")
            st.subheader("Annotated text")
            annotated = llm_output
            for issue in verdict["confirmed_issues"]:
                quote = issue["quote"]
                color = severity_color(issue["severity"])
                if quote in annotated:
                    annotated = annotated.replace(
                        quote,
                        f'<mark style="background:{color}22; border-bottom: 2px solid {color}; '
                        f'padding: 1px 2px;" title="{issue["issue"]}">{quote}</mark>',
                    )
            st.markdown(
                f'<div style="line-height:2; font-size:15px;">{annotated}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.subheader("Raw JSON")
            st.json(result)


with tab2:
    st.subheader("Critic comparison")

    if "last_result" not in st.session_state:
        st.info("Run an arbitration in the Verdict view tab first.")
    else:
        result = st.session_state["last_result"]
        reports = result["critic_reports"]
        disagreements = result["disagreements"]

        cols = st.columns(3)
        for col, report in zip(cols, reports):
            with col:
                score = report["score"]
                color = "#44bb44" if score >= 6 else "#ff4444"
                st.markdown(
                    f'<div style="border: 1px solid {color}; border-radius: 8px; padding: 16px;">'
                    f'<h4 style="color:{color}">{report["critic_name"]}</h4>'
                    f'<p>Score: <strong>{score}/10</strong> &nbsp; Verdict: <strong>{report["verdict"].upper()}</strong></p>'
                    f'<p style="color:#aaa; font-size:13px;">{report["reasoning"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if report["flags"]:
                    st.markdown("**Flags:**")
                    for flag in report["flags"]:
                        color2 = severity_color(flag["severity"])
                        st.markdown(
                            f'<small style="color:{color2}">▸ [{flag["severity"]}] {flag["issue"]}</small>',
                            unsafe_allow_html=True,
                        )

        if disagreements:
            st.markdown("---")
            st.subheader("Disagreements")
            for d in disagreements:
                st.warning(d["description"])
        else:
            st.success("All critics are in agreement.")


with tab3:
    st.subheader("Batch mode")

    batch_input = st.text_area(
        "Paste multiple outputs — separate each with a line containing only ---",
        height=300,
        placeholder="First LLM output here...\n---\nSecond LLM output here...\n---\nThird LLM output here...",
    )

    if st.button("Run batch arbitration", type="primary"):
        if not batch_input.strip():
            st.error("Please paste at least one output.")
        else:
            items_raw = [t.strip() for t in batch_input.split("\n---\n") if t.strip()]
            if len(items_raw) > 20:
                st.error("Maximum 20 outputs per batch.")
            else:
                payload = {"items": [{"llm_output": t} for t in items_raw]}
                with st.spinner(f"Arbitrating {len(items_raw)} outputs..."):
                    try:
                        with httpx.Client(timeout=300) as client:
                            response = client.post(f"{API_BASE}/v1/arbitrate/batch", json=payload)
                            response.raise_for_status()
                            batch_result = response.json()
                    except Exception as e:
                        st.error(f"API error: {e}")
                        st.stop()

                rows = []
                for i, res in enumerate(batch_result["results"]):
                    v = res["verdict"]
                    rows.append({
                        "Output": items_raw[i][:80] + "..." if len(items_raw[i]) > 80 else items_raw[i],
                        "Score": v["overall_score"],
                        "Confidence": v["confidence"],
                        "Issues": len(v["confirmed_issues"]),
                        "Disagreements": len(res["disagreements"]),
                        "ID": res["arbitration_id"],
                    })

                st.dataframe(
                    rows,
                    use_container_width=True,
                    column_config={
                        "Score": st.column_config.NumberColumn(min_value=1, max_value=10),
                    },
                )
