import os

from transformers import Speech2TextProcessor, \
    Speech2TextForConditionalGeneration
import soundfile as sf
import streamlit as st


def save_uploadedfile(uploadedfile, path):
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    print("Saved File:{} to upload".format(uploadedfile.name))


st.title('Speech Recognition')
st.subheader('Upload the any audio file to get the transcript')

# if the user chooses to upload the data
file = st.file_uploader('Audio file')
# browsing and uploading the dataset (strictly in csv format)
# dataset = pd.DataFrame()


if file is not None:
    speech, _ = sf.read(file)
    path = os.path.join(os.getcwd(), 'upload')
    file_path = os.path.join(path, file.name)
    save_uploadedfile(file, file_path)
    model = Speech2TextForConditionalGeneration.from_pretrained(
        "facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor.from_pretrained(
        "facebook/s2t-small-librispeech-asr")

    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt")
    generated_ids = model.generate(input_ids=inputs["input_features"],
                                   attention_mask=inputs["attention_mask"])
    transcription = processor.batch_decode(generated_ids)

    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/wav')
    st.write('Recognized transcript is: ')
    st.write(transcription[0])
