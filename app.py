import streamlit as st
import os


st.set_page_config(
    page_title="AI Voice-to-Text Notes Generator", 
    page_icon="🔊", 
    layout="centered"
)


if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

st.title("🎙️ AI Voice-to-Text Notes Generator")
st.caption("💻 Powered by - Local OpenAI Whisper & Transformers Pipelines")

st.markdown("""
### 📋 Description:
A smart audio companion that cuts out manual transcribing by converting your 
voice notes, lectures, and meeting files into text automatically. It processes 
the raw recordings locally on your machine to deliver full transcripts alongside 
quick, high-level summaries.
""")

st.divider()


st.subheader("📁 Upload Audio ")


uploaded_file = st.file_uploader(
    "Choose an audio recording track ", 
    type=["mp3", "wav", "m4a"],
    key=f"audio_uploader_{st.session_state['uploader_key']}"
)

if uploaded_file is not None:
    st.info(f"Loaded: `{uploaded_file.name}`")
    st.audio(uploaded_file)
    
    st.divider()
    st.subheader("🎯 NOW - AI Execution Workflow")
    
    if st.button("💬 Run Audio to Summarize", use_container_width=True):
        
        with st.spinner("🧠 Initializing.... It will take a while to process, please wait.."):
            import whisper
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            whisper_model = whisper.load_model("base") 
            
            model_name = "sshleifer/distilbart-cnn-12-6"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        temp_filename = "temp_audio_file.mp3"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())
            
        try:
            with st.spinner("📝 Running Speech-to-Text...."):
                result = whisper_model.transcribe(temp_filename)
                transcript = result["text"]
            
            st.success("🎤 Process Completed!")
            st.markdown("### 📃 Here is your: Full Transcript")
            st.text_area(label="", value=transcript, height=220, disabled=False)
            
            with st.spinner("💡 Running Summarization..."):
                inputs = tokenizer([transcript], max_length=1024, return_tensors="pt", truncation=True)
                words_count = len(transcript.split())
                max_len = min(140, int(words_count * 0.7)) if words_count > 10 else 10
                min_len = min(30, int(words_count * 0.2)) if words_count > 10 else 2
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    num_beams=4, 
                    max_length=max_len, 
                    min_length=min_len, 
                    early_stopping=True
                )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
            st.markdown("### ✍️ Summarized Notes: There you goo...")
            st.success(summary)

            final_output_notes = (
                f"=== AI VOICE-TO-TEXT NOTES GENERATOR REPORT ===\n\n"
                f"--- SOURCE FILE ---\n{uploaded_file.name}\n\n"
                f"--- AUTOMATED SUMMARY NOTES ---\n{summary}\n\n"
                f"--- FULL TRANSCRIPT ---\n{transcript}"
            )
            
            st.divider()
            st.subheader("💾 Export Generated Notes ")
            
            st.download_button(
                label="📄 Export and Save Notes (.txt)",
                data=final_output_notes,
                file_name=f"summary_{os.path.splitext(uploaded_file.name)[0]}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Critical runtime breakdown: {e}")
            
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


st.write("")
if st.button("🧹 Clear All & Start Fresh", type="secondary", use_container_width=True):
    st.session_state["uploader_key"] += 1
    st.rerun()