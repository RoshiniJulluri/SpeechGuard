\# SpeechGuard — Context-Aware Speech Error Detection and Verification for ASR



SpeechGuard is a research-oriented prototype that explores how potentially consequential errors in Automatic Speech Recognition (ASR) outputs can be flagged for human verification.



The project focuses on technical terms, numbers, units, and Telugu–English code-mixed speech.



\## Features



\- Audio transcription using Faster-Whisper.

\- Whisper `tiny` and `base` model options.

\- Rule-based warnings for technical vocabulary and numerical information.

\- Review indicators for suspected term variants and empty transcripts.

\- Editable transcripts and reference-transcript entry.

\- Human review labels and notes.

\- A Streamlit interface and evaluation dashboard.



\## How It Works



1\. Upload an audio recording.

2\. Generate a transcript using the selected ASR configuration.

3\. Inspect rule-based warnings.

4\. Review and edit the transcript where necessary.

5\. Save review information and examine available evaluation metrics.



\## Technology Stack



\- Python

\- Streamlit

\- Faster-Whisper

\- Pandas



\## Installation



Requires Python and a CPU-capable computer. A dedicated NVIDIA GPU is not required by the current CPU-based configuration.



Clone the repository:



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd SpeechGuard

```



Create and activate a virtual environment on Windows:



```powershell

python -m venv .venv

.venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the application:



```bash

streamlit run SpeechGuard\_app\_fixed.py

```



The first run may need internet access to download the selected speech recognition model.



\## Preliminary Experiments



Initial experiments compared Whisper model outputs on a small collection of short recordings involving programming terms, laptop memory, meeting times, and conversational speech.



These experiments helped investigate transcription differences and test the warning workflow. The available sample is small and does not establish general performance.



\## Limitations



\- The current warning detector is rule-based, not a trained error-detection model.

\- A warning indicates that review may be useful; it does not prove that a transcript is wrong.

\- The pilot dataset is small.

\- Telugu–English code-mixed speech and pronunciation variants require further evaluation.

\- Evaluation results depend on the quality and consistency of human labels.



\## Future Work



\- Expand and annotate the evaluation dataset.

\- Evaluate on unseen speakers and recording conditions.

\- Improve coverage of Telugu–English code-mixed expressions.

\- Compare the rule-based approach with appropriate verification baselines.

\- Report results on a larger, carefully reviewed dataset.



\## Project Information



\*\*Author:\*\* Roshini Julluri  

\*\*Institution:\*\* RGUKT Basar  

\*\*Project status:\*\* Working research prototype



