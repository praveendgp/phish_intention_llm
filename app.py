import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

from phishintention.config import Settings
from phishintention.formatter import (
    confidence_icon,
    confidence_level,
    confidence_percentage,
    display_agent,
    human_summary,
    human_text_report,
)
from phishintention.llm import MockClient, OllamaClient
from phishintention.pipeline import PhishIntentionPipeline
from phishintention.retrieval import KnowledgeBase
from phishintention.schema import INTENTS


st.set_page_config(
    page_title="PhishIntentionLLM",
    page_icon="🛡️",
    layout="wide",
)


def render_human_readable_result(
    result,
    threshold: float,
) -> None:
    """
    Render the model result in a friendly Streamlit layout.
    """
    summary = human_summary(
        result=result,
        threshold=threshold,
    )

    st.subheader("Analysis Summary")

    metric_column_1, metric_column_2, metric_column_3 = (
        st.columns(3)
    )

    with metric_column_1:
        st.metric(
            label="Detected intentions",
            value=summary["number_of_intentions"],
        )

    with metric_column_2:
        st.metric(
            label="Overall confidence",
            value=confidence_percentage(
                summary["overall_confidence"]
            ),
        )

    with metric_column_3:
        st.metric(
            label="Evidence consistency",
            value=confidence_percentage(
                summary["evidence_consistency"]
            ),
        )

    confidence = summary["overall_confidence"]

    st.write(
        (
            f"{confidence_icon(confidence)} "
            f"**Confidence level:** "
            f"{confidence_level(confidence)}"
        )
    )

    st.progress(
        max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )
    )

    if summary["passed_threshold"]:
        st.success(
            (
                "The final result passed the configured confidence "
                f"threshold of "
                f"{confidence_percentage(summary['threshold'])}."
            )
        )
    else:
        st.warning(
            (
                "The final result is below the configured confidence "
                f"threshold of "
                f"{confidence_percentage(summary['threshold'])}. "
                "Treat this result as requiring additional manual review."
            )
        )

    if not summary["detected"]:
        st.warning(
            (
                "No phishing intention was supported strongly enough "
                "by the visible screenshot evidence."
            )
        )

        return

    if summary["number_of_intentions"] > 1:
        st.info(
            (
                "This webpage contains evidence of multiple phishing "
                "intentions. The result is therefore a multi-label "
                "classification."
            )
        )

    st.subheader("Detected Phishing Intentions")

    for intention in summary["detected_intentions"]:
        title = (
            f"{intention['icon']} "
            f"{intention['name']} "
            f"• {confidence_percentage(intention['confidence'])}"
        )

        with st.container(border=True):
            st.markdown(f"### {title}")

            st.write(
                intention["description"]
            )

            st.markdown(
                "**Supporting visual evidence**"
            )

            if intention["evidence"]:
                for evidence in intention["evidence"]:
                    st.markdown(
                        f"- {evidence}"
                    )
            else:
                st.caption(
                    (
                        "The model did not return a separate evidence "
                        "statement for this intention."
                    )
                )

    st.subheader("Analysis Workflow")

    st.caption(
        (
            "The following agents participated in producing the "
            "final result."
        )
    )

    for index, agent in enumerate(
        summary["agents_invoked"],
        start=1,
    ):
        st.markdown(
            f"{index}. **{agent}**"
        )

    with st.expander(
        "Model interpretation note",
        expanded=False,
    ):
        st.write(
            (
                "The listed intentions are generated from observable "
                "elements in the supplied screenshot and retrieved "
                "phishing-pattern knowledge. The confidence value is "
                "a model-generated score. It should not be treated as "
                "a statistically calibrated probability unless a "
                "separate calibration experiment has been performed."
            )
        )


def create_pipeline(
    settings: Settings,
    use_mock: bool,
) -> PhishIntentionPipeline:
    """
    Initialise the selected model and the complete pipeline.
    """
    if use_mock:
        llm = MockClient()
    else:
        llm = OllamaClient(
            settings.base_url,
            settings.model,
            settings.max_image_side,
        )

    knowledge_base = KnowledgeBase(
        PROJECT_ROOT / "knowledge"
    )

    return PhishIntentionPipeline(
        llm=llm,
        kb=knowledge_base,
        threshold=settings.threshold,
        top_k=settings.top_k,
    )


st.title("🛡️ PhishIntentionLLM")

st.caption(
    (
        "Multi-agent retrieval-augmented analysis of phishing "
        "website screenshots"
    )
)

analyse_tab, annotation_tab = st.tabs(
    [
        "Analyse Screenshot",
        "Annotate Dataset",
    ]
)


with analyse_tab:
    settings = Settings()

    input_column, configuration_column = st.columns(
        [2, 1]
    )

    with input_column:
        uploaded_file = st.file_uploader(
            "Upload a website screenshot",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
        )

    with configuration_column:
        mode = st.selectbox(
            "Analysis mode",
            options=[
                "gated",
                "always",
                "single",
            ],
            format_func=lambda value: {
                "gated": "Confidence-gated multi-agent",
                "always": "All specialist agents",
                "single": "Single-agent baseline",
            }[value],
        )

        use_mock = st.checkbox(
            "Use mock model",
            help=(
                "Use only for testing the application without "
                "calling Ollama."
            ),
        )

        st.caption(
            (
                f"Model: `{settings.model}`  \n"
                f"Threshold: "
                f"`{settings.threshold:.2f}`"
            )
        )

    if uploaded_file is not None:
        outputs_directory = PROJECT_ROOT / "outputs"
        outputs_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_filename = Path(
            uploaded_file.name
        ).name

        uploaded_path = (
            outputs_directory
            / f"uploaded_{safe_filename}"
        )

        uploaded_path.write_bytes(
            uploaded_file.getvalue()
        )

        image = Image.open(uploaded_path)

        st.image(
            image,
            caption="Screenshot selected for analysis",
            width=700,
        )

        analyse_button = st.button(
            "Analyse Screenshot",
            type="primary",
            use_container_width=True,
        )

        if analyse_button:
            try:
                pipeline = create_pipeline(
                    settings=settings,
                    use_mock=use_mock,
                )

                with st.spinner(
                    "Running the multi-agent analysis..."
                ):
                    result = pipeline.run(
                        image_path=uploaded_path,
                        mode=mode,
                    )

                st.divider()

                output_tab_1, output_tab_2 = st.tabs(
                    [
                        "Human-Readable Report",
                        "Raw JSON",
                    ]
                )

                with output_tab_1:
                    render_human_readable_result(
                        result=result,
                        threshold=settings.threshold,
                    )

                    st.download_button(
                        label="Download Human-Readable Report",
                        data=human_text_report(
                            result=result,
                            threshold=settings.threshold,
                        ),
                        file_name=(
                            f"{result.sample_id}_report.txt"
                        ),
                        mime="text/plain",
                        use_container_width=True,
                    )

                with output_tab_2:
                    result_dictionary = result.model_dump(
                        mode="json"
                    )

                    st.json(
                        result_dictionary,
                        expanded=True,
                    )

                    json_output = json.dumps(
                        result_dictionary,
                        indent=2,
                        ensure_ascii=False,
                    )

                    st.download_button(
                        label="Download JSON Result",
                        data=json_output,
                        file_name=(
                            f"{result.sample_id}_result.json"
                        ),
                        mime="application/json",
                        use_container_width=True,
                    )

            except Exception as error:
                st.error(
                    "The screenshot could not be analysed."
                )

                st.exception(error)


with annotation_tab:
    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "manifest.csv"
    )

    if not manifest_path.exists():
        st.info(
            (
                "Generate the dataset manifest first by running "
                "`python scripts/prepare_data.py`."
            )
        )
    else:
        dataframe = pd.read_csv(
            manifest_path
        ).fillna("")

        pending_dataframe = dataframe[
            dataframe["annotation_status"]
            == "unlabelled"
        ]

        metric_column_1, metric_column_2 = st.columns(2)

        with metric_column_1:
            st.metric(
                "Total records",
                len(dataframe),
            )

        with metric_column_2:
            st.metric(
                "Unlabelled records",
                len(pending_dataframe),
            )

        if pending_dataframe.empty:
            st.success(
                "All manifest records have been labelled."
            )
        else:
            pending_position = st.number_input(
                "Pending record number",
                min_value=0,
                max_value=len(
                    pending_dataframe
                ) - 1,
                value=0,
                step=1,
            )

            selected_row = pending_dataframe.iloc[
                int(pending_position)
            ]

            selected_image_path = Path(
                selected_row["image_path"]
            )

            if selected_image_path.exists():
                st.image(
                    str(selected_image_path),
                    width=700,
                )
            else:
                st.error(
                    (
                        "The image path stored in the manifest "
                        "does not exist."
                    )
                )

            st.write(
                {
                    "sample_id": selected_row[
                        "sample_id"
                    ],
                    "source": selected_row[
                        "source"
                    ],
                    "brand": selected_row[
                        "brand"
                    ],
                    "phishing_status": selected_row[
                        "phishing_status"
                    ],
                }
            )

            selected_intentions = st.multiselect(
                "Select every visible phishing intention",
                options=INTENTS,
                format_func=lambda intent: (
                    intent.replace(
                        "_",
                        " ",
                    ).title()
                ),
            )

            annotator = st.text_input(
                "Annotator"
            )

            notes = st.text_area(
                "Annotation notes"
            )

            save_button = st.button(
                "Save Annotation",
                type="primary",
            )

            if save_button:
                selected_mask = (
                    dataframe["sample_id"]
                    == selected_row["sample_id"]
                )

                for intention in INTENTS:
                    dataframe.loc[
                        selected_mask,
                        intention,
                    ] = (
                        1
                        if intention
                        in selected_intentions
                        else 0
                    )

                dataframe.loc[
                    selected_mask,
                    "annotation_status",
                ] = "labelled"

                dataframe.loc[
                    selected_mask,
                    "annotator",
                ] = annotator

                dataframe.loc[
                    selected_mask,
                    "notes",
                ] = notes

                dataframe.to_csv(
                    manifest_path,
                    index=False,
                )

                st.success(
                    (
                        "The annotation was saved. Refresh the "
                        "page to move to the next pending record."
                    )
                )