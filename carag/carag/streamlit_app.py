import traceback

import streamlit as st

from main import run as pipeline_run


st.set_page_config(page_title="CarAG RAG QA", layout="wide")

st.title("CarAG – Contradiction-Aware RAG")
st.markdown(
    "Enter a question below. The app will search the web, build a contradiction-aware view, "
    "and stream the pipeline logs in real time."
)

query = st.text_area("Your question", height=100, placeholder="e.g. Is coffee good or bad for heart health?")

run_button = st.button("Run analysis", type="primary")

log_placeholder = st.empty()
result_placeholder = st.empty()


def _make_logger():
    logs = []

    def log_fn(message):
        # Append and re-render full log each time for a streaming-like effect
        logs.append(str(message))
        log_placeholder.text("\n".join(logs))

    return log_fn


if run_button:
    if not query or not query.strip():
        st.warning("Please enter a question before running the analysis.")
    else:
        logger = _make_logger()
        try:
            final_output = pipeline_run(query.strip(), log_fn=logger)
            if final_output is not None:
                result_placeholder.subheader("Answer (text)")
                result_placeholder.text(final_output.get("text_answer", ""))

                with st.expander("Structured output (JSON)", expanded=False):
                    st.json(final_output)
            else:
                st.info("Pipeline finished but no structured output was produced for this query.")
        except Exception as e:
            tb = traceback.format_exc()
            logger(tb)
            st.error(f"An error occurred while running the pipeline: {e}")

