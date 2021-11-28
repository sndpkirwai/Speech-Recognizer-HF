from transformers import Speech2TextProcessor, \
    Speech2TextForConditionalGeneration
from IPython.display import Audio
from flask import Flask
import soundfile as sf
import streamlit as st

app = Flask(__name__)


@app.route('/audio_to_text/')
def audio_to_text():
    st.title('Speech Recognition')
    st.subheader('Upload the any audio file to get the transcript')

    # if the user chooses to upload the data
    file = st.file_uploader('Audio file')
    # browsing and uploading the dataset (strictly in csv format)
    # dataset = pd.DataFrame()
    flag = False

    if file is not None:
        speech, _ = sf.read(file)

        model = Speech2TextForConditionalGeneration.from_pretrained(
            "facebook/s2t-small-librispeech-asr")
        processor = Speech2TextProcessor.from_pretrained(
            "facebook/s2t-small-librispeech-asr")

        inputs = processor(speech, sampling_rate=16_000, return_tensors="pt")
        generated_ids = model.generate(input_ids=inputs["input_features"],
                                       attention_mask=inputs["attention_mask"])
        transcription = processor.batch_decode(generated_ids)

        st.write(Audio(file))
        st.write('Recognized transcript is: ')
        st.write(transcription)


if __name__ == "__main__":
    app.run(debug=True)