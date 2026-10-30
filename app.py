from __future__ import annotations

import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel

# -------------------- Project paths --------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
RECORDINGS_DIR = DATA / "recordings"

MANUAL_REVIEW = OUTPUTS / "manual_error_review.csv"
IMPROVED_FLAGS = OUTPUTS / "speechguard_improved_flags.csv"
USER_RECORDINGS_CSV = OUTPUTS / "user_recordings.csv"

TECHNICAL_TERMS = [
    "python", "code", "compile", "laptop", "ram", "memory",
    "usage", "project", "file", "meeting", "gb", "mb", "tb",
    "ghz", "mhz", "kg", "mg", "km", "cm", "mm",
]
NUMBER_PATTERN = re.compile(
    r"(?<!\w)\d+(?:[.,:]\d+)?\s?(?:gb|mb|tb|kg|mg|km|cm|mm|mph|ghz|mhz)?\b",
    flags=re.IGNORECASE,
)
ARABIC_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF]")
HYPHENATED_TOKEN_PATTERN = re.compile(r"\b[\w]+(?:-[\w]+){2,}\b")

# Conservative, review-only variants observed in the current pilot example.
# These are NOT corrections and must be validated against more recordings.
SUSPECTED_TERM_VARIANTS = {
    "python": {"paitan", "paiton", "paithon", "pithon", "pythan", "pytan"},
    "code": {"kod", "koad", "coad"},
}


# -------------------- Detector helpers --------------------
def find_terms(text: str) -> list[str]:
    lowered = str(text or "").lower()
    return sorted({
        term for term in TECHNICAL_TERMS
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered)
    })


def find_numbers(text: str) -> list[str]:
    return [match.group(0).strip() for match in NUMBER_PATTERN.finditer(str(text or ""))]


def find_suspected_term_mismatches(text: str) -> list[str]:
    """Flag known ASR-like variants for human review; never rewrite the transcript."""
    tokens = re.findall(r"[^\W_]+", str(text or "").lower(), flags=re.UNICODE)
    suspects = []
    for expected_term, variants in SUSPECTED_TERM_VARIANTS.items():
        for token in tokens:
            if token in variants:
                suspects.append(f"{token} (possible {expected_term})")
    return sorted(set(suspects))


def detect_warnings(text: str, avg_logprob=None, no_speech_prob=None,
                    compression_ratio=None) -> tuple[bool, list[str], list[str], list[str]]:
    """Transparent review heuristics: warnings prompt verification, not auto-correction."""
    transcript = str(text or "").strip()
    reasons: list[str] = []
    terms = find_terms(transcript)
    numbers = find_numbers(transcript)
    suspected_mismatches = find_suspected_term_mismatches(transcript)

    if not transcript:
        reasons.append("empty_transcript")
    if transcript and ARABIC_SCRIPT_PATTERN.search(transcript):
        reasons.append("unexpected_script_signal")
    if transcript and HYPHENATED_TOKEN_PATTERN.search(transcript):
        reasons.append("heavily_hyphenated_output")

    # These are model diagnostics, not calibrated probabilities of correctness.
    if avg_logprob is not None and avg_logprob < -1.0:
        reasons.append("low_asr_log_probability")
    if no_speech_prob is not None and no_speech_prob > 0.6:
        reasons.append("high_no_speech_probability")
    if compression_ratio is not None and compression_ratio > 2.4:
        reasons.append("high_compression_ratio")

    # Surface consequential items so a person can check them against the audio.
    if terms:
        reasons.append("technical_terms_to_verify")
    if numbers:
        reasons.append("numbers_or_units_to_verify")
    if suspected_mismatches:
        reasons.append("possible_phonetic_technical_term_mismatch")

    if not reasons:
        reasons.append("no_warning_signal_detected")

    return (
        any(reason != "no_warning_signal_detected" for reason in reasons),
        reasons,
        terms,
        numbers,
    )


# -------------------- Whisper helpers --------------------
@st.cache_resource(show_spinner="Loading the speech recognition model…")
def load_model(model_name: str):
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def _decode_audio(audio_path: str, model_name: str, language: str | None):
    model = load_model(model_name)
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        # Avoid VAD filtering for short clips: it can remove quiet speech entirely.
        vad_filter=False,
    )
    segments = list(segments_iter)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()

    logprobs = [s.avg_logprob for s in segments if getattr(s, "avg_logprob", None) is not None]
    no_speech = [s.no_speech_prob for s in segments if getattr(s, "no_speech_prob", None) is not None]
    compression = [s.compression_ratio for s in segments if getattr(s, "compression_ratio", None) is not None]

    signals = {
        "language": getattr(info, "language", "unknown"),
        "language_probability": getattr(info, "language_probability", None),
        "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else None,
        "no_speech_prob": sum(no_speech) / len(no_speech) if no_speech else None,
        "compression_ratio": sum(compression) / len(compression) if compression else None,
    }
    return transcript, signals


def transcribe_uploaded_file(uploaded_file, model_name: str, language_choice: str):
    """For code-switch mode, decode twice to show alternatives instead of trusting one language guess."""
    suffix = Path(uploaded_file.name).suffix or ".audio"
    audio_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        temp_path = Path(tmp.name)

    try:
        if language_choice == "Telugu-English mixed (compare)":
            auto_text, auto_signals = _decode_audio(str(temp_path), model_name, None)
            te_text, te_signals = _decode_audio(str(temp_path), model_name, "te")
            return {
                "transcript": auto_text,
                "signals": auto_signals,
                "alternatives": [
                    {"label": "Auto-detect", "transcript": auto_text, "signals": auto_signals},
                    {"label": "Telugu-forced", "transcript": te_text, "signals": te_signals},
                ],
            }

        language = None if language_choice == "Auto-detect" else (
            "te" if language_choice == "Telugu" else "en"
        )
        transcript, signals = _decode_audio(str(temp_path), model_name, language)
        return {
            "transcript": transcript,
            "signals": signals,
            "alternatives": [{"label": language_choice, "transcript": transcript, "signals": signals}],
        }
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass


# -------------------- Saved-example helpers --------------------
def read_csv_if_present(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def ensure_user_recordings_file():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_RECORDINGS_CSV.exists():
        pd.DataFrame(columns=[
            "example_id", "created_at", "audio_path", "original_filename",
            "model", "language_mode", "detected_language", "predicted_transcript",
            "reference_transcript", "actual_consequential_error", "human_review_needed",
            "review_recommended", "review_reason", "technical_terms_found",
            "numbers_found", "avg_logprob", "no_speech_prob", "compression_ratio",
            "human_notes",
        ]).to_csv(USER_RECORDINGS_CSV, index=False, encoding="utf-8-sig")


def append_user_example(audio_bytes: bytes, original_filename: str, model_name: str,
                        language_mode: str, transcript: str, signals: dict,
                        review: bool, reasons: list[str], terms: list[str],
                        numbers: list[str], reference: str, actual_label: str,
                        human_review_label: str, human_notes: str) -> str:
    ensure_user_recordings_file()
    example_id = uuid.uuid4().hex[:10]
    suffix = Path(original_filename).suffix or ".audio"
    saved_audio = RECORDINGS_DIR / f"{example_id}{suffix}"
    saved_audio.write_bytes(audio_bytes)

    row = {
        "example_id": example_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "audio_path": str(saved_audio.relative_to(ROOT)),
        "original_filename": original_filename,
        "model": model_name,
        "language_mode": language_mode,
        "detected_language": signals.get("language", "unknown"),
        "predicted_transcript": transcript,
        "reference_transcript": reference.strip(),
        "actual_consequential_error": actual_label,
        "human_review_needed": human_review_label,
        "review_recommended": bool(review),
        "review_reason": ";".join(reasons),
        "technical_terms_found": ";".join(terms),
        "numbers_found": ";".join(numbers),
        "avg_logprob": signals.get("avg_logprob"),
        "no_speech_prob": signals.get("no_speech_prob"),
        "compression_ratio": signals.get("compression_ratio"),
        "human_notes": human_notes.strip(),
    }
    old = pd.read_csv(USER_RECORDINGS_CSV, encoding="utf-8-sig")
    updated = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(USER_RECORDINGS_CSV, index=False, encoding="utf-8-sig")
    return example_id


# -------------------- Evaluation helpers --------------------
def binary_metrics(frame: pd.DataFrame, flag_column: str,
                  label_column: str = "actual_consequential_error") -> dict:
    """Calculate binary metrics only where both human label and detector decision exist.

    label_column may be actual_consequential_error (is transcript wrong?) or
    human_review_needed (should a human review this item?). These are distinct tasks.
    Missing/uncertain labels and missing detector outputs are excluded, not treated as negatives.
    """
    empty = {"evaluated": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
             "precision": None, "recall": None, "f1": None,
             "accuracy": None, "fpr": None}
    if frame is None or frame.empty or label_column not in frame.columns or flag_column not in frame.columns:
        return empty

    work = frame.copy()
    labels = work[label_column].fillna("").astype(str).str.strip().str.lower()
    flag_values = work[flag_column]
    # Missing detector decisions are unknown, not a negative decision.
    valid_flags = flag_values.notna()
    work = work[labels.isin(["yes", "no"]) & valid_flags].copy()
    if work.empty:
        return empty

    labels = work[label_column].astype(str).str.strip().str.lower()
    actual = labels.eq("yes")
    predicted = work[flag_column].astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y"]
    )

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    n = len(work)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    accuracy = (tp + tn) / n if n else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {"evaluated": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "fpr": fpr}


def pct(value):
    return "N/A" if value is None else f"{value:.1%}"


def load_evaluation_rows():
    rows = []

    # Previous challenge set and manual review.
    old = read_csv_if_present(MANUAL_REVIEW)
    if old is not None and not old.empty:
        old = old.copy()
        old["source"] = "Existing challenge set"
        old["baseline_flag"] = old.get("review_recommended", False)
        rows.append(old)

    # New recordings added through the app. They persist across restarts.
    new = read_csv_if_present(USER_RECORDINGS_CSV)
    if new is not None and not new.empty:
        new = new.copy()
        new["source"] = "New app recording"
        new["baseline_flag"] = new.get("review_recommended", False)
        rows.append(new)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True, sort=False)
    # For old rows, pull improved flags from the existing improved-detector output.
    improved = read_csv_if_present(IMPROVED_FLAGS)
    if improved is not None and {"audio_file", "model", "review_recommended"}.issubset(improved.columns):
        combined["_join_audio"] = combined.get("audio_file", pd.Series(index=combined.index, dtype=str)).fillna("")
        combined["_join_model"] = combined.get("model", pd.Series(index=combined.index, dtype=str)).fillna("")
        imp = improved[["audio_file", "model", "review_recommended"]].copy()
        imp = imp.rename(columns={"audio_file": "_join_audio", "model": "_join_model",
                                  "review_recommended": "improved_flag"})
        combined = combined.merge(imp, on=["_join_audio", "_join_model"], how="left")
        combined.drop(columns=["_join_audio", "_join_model"], inplace=True, errors="ignore")
    if "improved_flag" not in combined.columns:
        combined["improved_flag"] = pd.NA

    # Keep absent detector predictions as missing. A missing result must not be
    # silently counted as a negative prediction in evaluation metrics.
    return combined


# -------------------- Streamlit interface --------------------
st.set_page_config(page_title="SpeechGuard | Speech verification", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}
  [data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #f6f8fc 0%, #ffffff 35%);}
  .sg-hero {padding: 1.45rem 1.6rem; border-radius: 20px; background: linear-gradient(120deg,#102a43,#1d4e89); color: white; margin-bottom: 1rem; box-shadow: 0 8px 28px rgba(16,42,67,.12);}
  .sg-hero h1 {color: white; margin: 0; font-size: 2.15rem;}
  .sg-hero p {color: #e5eef9; margin: .45rem 0 0 0; font-size: 1rem;}
  .sg-card {background: white; border: 1px solid #e5eaf1; border-radius: 16px; padding: 1rem 1.15rem; margin: .45rem 0 1rem 0; box-shadow: 0 3px 14px rgba(15,23,42,.035);}
  .sg-label {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #64748b; font-weight: 700;}
  .sg-transcript {background:#f0f7ff; border:1px solid #bfdbfe; border-left:5px solid #2563eb; padding:1.1rem 1.2rem; border-radius:12px; color:#102a43; font-size:1.08rem; line-height:1.7; white-space:pre-wrap; overflow-wrap:anywhere; min-height:70px;}
  div.stButton > button[kind="primary"] {border-radius: 10px; min-height: 2.8rem; font-weight: 650;}
  div[data-testid="stMetric"] {background:#fff; border:1px solid #e5eaf1; padding: .8rem; border-radius: 13px;}
  div[data-testid="stTabs"] button {font-weight:600;}
</style>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="sg-hero"><h1>🎙️ SpeechGuard</h1>'
    '<p>Context-aware speech error detection and verification for Telugu–English and other speech recordings.</p></div>',
    unsafe_allow_html=True,
)
st.caption("Transcribe first. Review important words and numbers. Save verified examples to evaluate the detector.")

with st.sidebar:
    st.header("Transcription settings")
    model_name = st.selectbox("Whisper model", ["tiny", "base"], index=1)
    language_choice = st.selectbox(
        "Language mode",
        ["Auto-detect", "Telugu-English mixed (compare)", "Telugu", "English"],
        index=1,
        help="For Telugu-English speech, compare auto-detection with a Telugu-forced decode. Neither is guaranteed to be correct.",
    )
    st.caption("Runs locally on CPU. First run may download model files.")
    st.divider()
    st.markdown("**Recommended first try**")
    st.caption("Model: base · Language: Telugu-English mixed (compare) · Audio: a clear 5–15 second clip")

tab_demo, tab_eval, tab_about = st.tabs(
    ["🎧 Transcribe & verify", "📊 Evaluation", "ℹ️ About"]
)

with tab_demo:
    st.markdown("## 1 · Add your audio")
    st.write("Upload a recording, then press **Transcribe and check**. The transcript will appear in a blue result panel below.")
    uploaded = st.file_uploader(
        "Choose a short audio file",
        type=["wav", "mp3", "m4a", "mp4", "webm", "ogg", "flac"],
        key="audio_upload",
    )
    if uploaded is not None:
        st.audio(uploaded)
        if st.button("Transcribe and check", type="primary"):
            try:
                with st.spinner("Running speech recognition and checking warning signals…"):
                    result = transcribe_uploaded_file(uploaded, model_name, language_choice)
                st.session_state["latest_result"] = result
                st.session_state["latest_run_id"] = uuid.uuid4().hex[:8]
                st.session_state["latest_audio_bytes"] = uploaded.getvalue()
                st.session_state["latest_filename"] = uploaded.name
                st.session_state["latest_model"] = model_name
                st.session_state["latest_language_mode"] = language_choice
            except Exception as exc:
                st.error(f"Could not process this audio: {exc}")
                st.info("Check the audio format and the terminal error. First model load may take time.")

    result = st.session_state.get("latest_result")
    if result:
        st.markdown("## 2 · Transcript result")
        alternatives = result["alternatives"]
        if len(alternatives) > 1:
            st.info(
                "Code-switching note: Whisper chooses a dominant language for decoding. "
                "For mixed Telugu-English speech, compare both outputs and choose the one that best matches the audio."
            )
            # Show both decoding attempts side by side for easy comparison.
            col_auto, col_telugu = st.columns(2)
            for col, item in zip((col_auto, col_telugu), alternatives):
                with col:
                    st.markdown(f"### {item['label']}")
                    alt_text = item["transcript"].strip()
                    if alt_text:
                        escaped_text = __import__("html").escape(alt_text)
                        st.markdown(
                            f'<div class="sg-transcript">{escaped_text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("No transcript returned for this option.")

            # Choose which alternative to edit and save below.
            labels = [item["label"] for item in alternatives]
            selected_label = st.radio(
                "Choose a transcript to inspect and save",
                labels,
                horizontal=True,
            )
            selected = next(
                item for item in alternatives
                if item["label"] == selected_label
            )
        else:
            selected = alternatives[0]

        transcript = selected["transcript"]
        signals = selected["signals"]
        if transcript.strip():
            st.markdown('<div class="sg-label">Recognized speech</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sg-transcript">{__import__("html").escape(transcript)}</div>', unsafe_allow_html=True)
        else:
            st.error("No transcript was returned for this decoding option.")
            st.markdown("**Try these quick fixes:** switch Language mode to **Auto-detect** or **Telugu**, try the other transcript option above, confirm the audio plays, or test a clearer 5–15 second clip. Short clips are now decoded without voice-activity filtering.")
        review, reasons, terms, numbers = detect_warnings(
            transcript, signals["avg_logprob"], signals["no_speech_prob"],
            signals["compression_ratio"]
        )
        suspected_mismatches = find_suspected_term_mismatches(transcript)

        run_id = st.session_state.get("latest_run_id", "initial")
        edited_key = f"edited_transcript_{run_id}_{selected_label if len(alternatives) > 1 else 'single'}"
        if edited_key not in st.session_state:
            st.session_state[edited_key] = transcript
        edited_transcript = st.text_area(
            "Edit transcript after listening (optional)", height=100, key=edited_key,
            placeholder="If the transcript is empty, try another language mode. You can also type a verified transcript here."
        )
        st.caption("The blue panel is the raw ASR output. Use the reference field below for the human-verified transcript; the raw prediction is preserved for fair evaluation.")
        if review:
            st.warning("Verification recommended")
        else:
            st.success("No configured warning signal detected")
        st.write("**Why review is recommended:** " + (
            ", ".join(reasons) if review else "No configured warning signal detected"
        ))

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Technical terms found:**", ", ".join(terms) if terms else "None detected")
            st.write("**Numbers / units found:**", ", ".join(numbers) if numbers else "None detected")
            st.write(
                "**Possible technical-term mismatches (review only):**",
                ", ".join(suspected_mismatches) if suspected_mismatches else "None detected",
            )
        with col2:
            st.write("**Whisper's dominant-language guess:**", signals.get("language", "unknown"))
            probability = signals.get("language_probability")
            st.write("**Language probability:**", f"{probability:.3f}" if isinstance(probability, (float, int)) else "N/A")

        with st.expander("ASR diagnostic signals (not calibrated correctness scores)"):
            for key in ["avg_logprob", "no_speech_prob", "compression_ratio"]:
                value = signals.get(key)
                st.write(f"**{key}:** {value:.3f}" if value is not None else f"**{key}:** N/A")

        st.info(
            "Please listen to the original audio and verify names, technical terms, numbers, units, "
            "and AM/PM or time expressions before using the transcript."
        )

        st.markdown("## 3 · Save an evaluation example")
        st.write(
            "Saving stores the audio, transcript, detector decision, and your label in the project. "
            "A new example is not used in metric calculations until you choose a confirmed label."
        )
        with st.form("save_example_form", clear_on_submit=False):
            reference = st.text_area(
                "Reference / human-verified transcript (recommended)",
                value="",
                help="Listen to the audio and type what was actually spoken. This supports WER/CER evaluation later.",
            )
            actual_label = st.selectbox(
                "After listening, does the transcript contain a consequential ASR error?",
                ["uncertain", "yes", "no"],
                index=0,
                help="Label transcript correctness only. Use yes/no only after listening and comparing with the reference.",
            )
            human_review_label = st.selectbox(
                "Independently, should a human reviewer check this transcript?",
                ["uncertain", "yes", "no"],
                index=0,
                help="This is a separate label from whether the transcript is wrong. A correct transcript containing a critical number may still merit review.",
            )
            notes = st.text_input("Notes (optional)", placeholder="e.g. RAM value correct; location word mistranscribed")
            save = st.form_submit_button("Save recording and review row")
        if save:
            try:
                edited_text = st.session_state.get(edited_key, transcript).strip()
                example_id = append_user_example(
                    audio_bytes=st.session_state["latest_audio_bytes"],
                    original_filename=st.session_state["latest_filename"],
                    model_name=st.session_state["latest_model"],
                    language_mode=st.session_state["latest_language_mode"],
                    transcript=edited_text,
                    signals=signals,
                    review=review,
                    reasons=reasons,
                    terms=terms,
                    numbers=numbers,
                    reference=reference,
                    actual_label=actual_label,
                    human_review_label=human_review_label,
                    human_notes=notes,
                )
                st.success(f"Saved example {example_id}. It will appear in the Evaluation dataset tab.")
            except Exception as exc:
                st.error(f"Could not save the example: {exc}")

with tab_eval:
    st.markdown("## Evaluation workspace")
    st.write(
        "This combines your original manually reviewed challenge examples with recordings added through the app. "
        "New recordings persist in `data/recordings/` and `outputs/user_recordings.csv`."
    )
    ensure_user_recordings_file()
    combined = load_evaluation_rows()

    if combined is None or combined.empty:
        st.info("No evaluation examples found yet. Upload and save a recording in the first tab.")
    else:
        labels = combined.get("actual_consequential_error", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.lower()
        confirmed_count = int(labels.isin(["yes", "no"]).sum())
        uncertain_count = int(labels.eq("uncertain").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total examples", len(combined))
        c2.metric("Confirmed yes/no labels", confirmed_count)
        c3.metric("Uncertain labels", uncertain_count)

        metrics = []
        existing_rows = combined[combined["source"] == "Existing challenge set"].copy()
        new_rows = combined[combined["source"] == "New app recording"].copy()

        metric_jobs = []
        if not existing_rows.empty:
            metric_jobs.extend([
                ("Review warning vs actual error — baseline (proxy)", existing_rows, "baseline_flag", "actual_consequential_error"),
                ("Review warning vs actual error — improved (proxy)", existing_rows, "improved_flag", "actual_consequential_error"),
            ])
        if not new_rows.empty:
            metric_jobs.extend([
                ("In-app warning vs actual error (proxy)", new_rows, "baseline_flag", "actual_consequential_error"),
                ("In-app warning vs human review-needed label", new_rows, "baseline_flag", "human_review_needed"),
            ])

        for name, subset, column, label_column in metric_jobs:
            m = binary_metrics(subset, column, label_column)
            metrics.append({
                "Evaluation target": name, "Evaluated": m["evaluated"],
                "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
                "Precision": pct(m["precision"]), "Recall": pct(m["recall"]),
                "F1-score": pct(m["f1"]), "Accuracy": pct(m["accuracy"]),
                "False-positive rate": pct(m["fpr"]),
            })
        st.markdown("#### Evaluation metrics")
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
        st.caption(
            "Important: the first target treats a review warning as a proxy for detecting an actual consequential ASR error; it does not directly measure whether the warning was useful. "
            "The separate review-needed target compares the system warning with your independent human_review_needed label. "
            "Only confirmed yes/no labels and non-missing detector decisions are included; uncertain labels and missing predictions are excluded. "
            "N/A means a metric cannot be calculated. Small pilot results are preliminary."
        )

        st.markdown("#### All examples")
        display_columns = [
            "source", "example_id", "audio_file", "original_filename", "model",
            "predicted_transcript", "reference_transcript", "actual_consequential_error",
            "human_review_needed", "baseline_flag", "improved_flag", "review_reason", "error_description", "human_notes",
        ]
        display_columns = [column for column in display_columns if column in combined.columns]
        st.dataframe(combined[display_columns], use_container_width=True, hide_index=True)

        st.download_button(
            "Download combined evaluation CSV",
            data=combined.to_csv(index=False).encode("utf-8-sig"),
            file_name="speechguard_combined_evaluation.csv",
            mime="text/csv",
        )

        st.markdown("#### Update labels for new recordings")
        st.write(
            "You can edit `actual_consequential_error` (whether the transcript is wrong), `human_review_needed` (whether a person should review it), `reference_transcript`, and `human_notes` in `outputs/user_recordings.csv`. "
            "These labels answer different questions. Save the CSV and rerun/refresh the app."
        )

with tab_about:
    st.markdown("### What this prototype demonstrates")
    st.markdown(
        """
        - Pretrained Whisper transcription for uploaded audio.
        - An experimental mixed-language mode that compares auto-detection with Telugu-forced decoding.
        - Review cues for empty outputs, unexpected script, heavily hyphenated output, technical terms,
          numbers/units, and available ASR decoding diagnostics.
        - Persistent addition of new recordings to the evaluation dataset.
        - Evaluation against human-reviewed consequential-error labels.
        """
    )
    st.markdown("### Important limitations")
    st.markdown(
        """
        - Whisper's language identification reports a dominant language; it does not reliably label every word in code-switched speech.
        - Comparing auto-detected and Telugu-forced outputs gives alternatives, not a guarantee of correct Telugu-English transcription.
        - Numbers/technical terms are highlighted for human verification; presence alone does not prove an error.
        - This is an early rule-based prototype, not a trained error-detection model.
        - A small dataset cannot establish general real-world performance.
        """
    )
