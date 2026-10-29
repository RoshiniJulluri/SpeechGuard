# 🎙️ SpeechGuard
### Context-Aware Speech Error Detection and Verification for ASR

**SpeechGuard** is a research-oriented prototype that explores how Automatic Speech Recognition (ASR) systems can miss or misinterpret important information in spoken language.

Instead of treating every transcription error equally, SpeechGuard focuses on potentially consequential errors involving **technical terms, numbers, units, and empty or incomplete transcripts**.

The goal is to help users identify parts of a transcript that may need human verification.

---

## 🎯 Problem Statement

Automatic Speech Recognition systems can produce incorrect or incomplete transcripts, especially when speech contains technical vocabulary, numbers, mixed-language expressions, or challenging audio conditions.

Some mistakes may change the meaning of the original speech.

For example:

| Spoken content | Potential transcription issue |
|---|---|
| "My laptop has 16 GB RAM" | The number or unit may be misrecognized |
| "Run the Python code" | A technical term may be transcribed incorrectly |
| A short or unclear recording | The model may produce an empty transcript |

**SpeechGuard investigates whether context-aware warning rules can help identify these cases for human review.**

## 💡 Proposed Approach

SpeechGuard combines ASR transcription with a lightweight, rule-based verification layer.

1. **Transcribe:** Generate a transcript from uploaded audio using Faster-Whisper.
2. **Inspect:** Check the transcript for potentially important terms, numbers, units, and suspicious output.
3. **Flag:** Highlight segments or transcripts that may require verification.
4. **Review:** Allow the user to edit the transcript, compare it with a reference, and record review labels and notes.
5. **Evaluate:** Explore transcription errors and detector performance using small pilot experiments.

> **Important:** SpeechGuard does not currently guarantee that a flagged transcript is incorrect. It recommends human review based on predefined rules.

## ✨ Features

- 🎧 **Audio transcription** using Faster-Whisper.
- 🌐 **Language options**, including automatic detection and a Telugu-focused mode.
- 🔎 **Context-aware warning rules** for technical vocabulary, numbers, units, and empty transcripts.
- ✍️ **Transcript editing** for manual correction.
- 📋 **Reference comparison** to support error analysis.
- 🏷️ **Human review labels and notes** for documenting observations.
- 📊 **Pilot evaluation tools** for examining transcription and detection results.
- 💻 **Local execution** on a CPU, without requiring an NVIDIA GPU.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Faster-Whisper | Speech recognition |
| Pandas | Data handling and experiment outputs |
| JiWER | Word Error Rate evaluation |
| Scikit-learn | Supporting evaluation utilities |
| Matplotlib | Experiment visualizations |

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies
- Git
- A working internet connection for installing dependencies and downloading the speech recognition model when needed

### 1. Clone the repository

```bash
git clone https://github.com/RoshiniJulluri/SpeechGuard.git
cd SpeechGuard
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, follow the Python documentation for your system's execution policy or activate the environment using another supported method.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run SpeechGuard_app_fixed.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use SpeechGuard.

**Note:** The first run may take longer because the speech recognition model may need to be downloaded. CPU inference speed depends on your hardware and the selected model.

## 🔬 Research and Evaluation

SpeechGuard is being developed as an exploratory project to study consequential speech recognition errors and lightweight verification methods.

Current experimentation includes:

- Comparing Faster-Whisper model outputs.
- Examining errors involving technical vocabulary and numerical information.
- Comparing predictions against reference transcripts.
- Recording human judgments about potentially consequential errors.
- Evaluating detection results using precision, recall, F1-score, and other suitable metrics when the available data supports them.

### Evaluation methodology

The project uses standard evaluation concepts:

- **Word Error Rate (WER):** Measures word-level differences between a reference transcript and a predicted transcript.
- **Precision:** Measures how many flagged cases are genuinely positive among all flagged cases.
- **Recall:** Measures how many genuinely positive cases the detector successfully identifies.
- **F1-score:** Combines precision and recall into a single metric.

Evaluation results should be interpreted in the context of the dataset size, label quality, and the distribution of positive and negative examples. Small pilot experiments are not sufficient to establish general performance.

## ⚠️ Current Limitations

- The consequential-error detector is **rule-based**, not a trained machine learning classifier.
- A warning indicates a potential issue; it does not prove that the transcription is wrong.
- The current experiments use a small pilot dataset and are not representative of all speakers, languages, accents, or recording conditions.
- Telugu-English code-mixed speech requires further systematic evaluation.
- Model comparisons and evaluation metrics need larger, carefully annotated datasets before drawing strong conclusions.
- SpeechGuard is a research prototype, not a production-ready or safety-critical verification system.

## 🚀 Future Work

- Build a larger, more diverse, and carefully annotated evaluation dataset.
- Improve support for Telugu-English code-mixed speech.
- Investigate confidence-aware and context-aware error detection.
- Compare rule-based methods with learned approaches when sufficient labeled data is available.
- Evaluate detection quality across different speech recognition models and recording conditions.
- Study the relationship between transcription errors and their potential impact on meaning.

## 👩‍💻 Project

**Author:** Roshini Julluri  
**Project:** SpeechGuard  
**Focus:** Speech Recognition, Error Analysis, and Context-Aware Verification

This project is an ongoing exploration of how speech recognition systems can be made easier to verify and more reliable in situations where specific words, numbers, or units matter.

---

*SpeechGuard is an experimental research prototype. It is intended to support investigation and human review, not to replace them.*
