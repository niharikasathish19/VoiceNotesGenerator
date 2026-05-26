# 🎙️ AI Voice-to-Text Notes Generator

A smart audio companion that cuts out manual transcribing by converting your voice notes, lectures, and meeting files into text automatically. It processes the raw recordings locally on your machine to deliver full transcripts alongside quick, high-level summaries.


## 🚀 App Workflows
* **📁 Upload Audio:** Drop in local audio tracks (`.mp3`, `.wav`, or `.m4a`) with a built-in instant playback engine.

* **📝 Speech-to-Text Conversion:** Processes your tracks natively using an offline OpenAI Whisper engine.

* **💡 Smart Summarization:** Uses a localized DistilBART transformer network to automatically generate clean summaries.

* **💾 Simple Asset Export:** Packages your full transcripts and notes into a downloadable `.txt` dashboard file.

* **🧹 Fresh Session Flush:** Instantly purges browser file uploader caches to let you start a brand new audio workspace.


## ⚙️ How to Setup & Run

1. Make sure you have Python installed on your Windows system.
2. Open your terminal window and install the project dependencies:
   ```bash
   
   pip install streamlit whisper transformers torch