import io
import os
import string
import pandas as pd
import numpy as np
import wave
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
import json
import argparse

bucketname = "machine_feed"

filepath = "/Users/axelahdritz/Desktop/thesis/audio_files/"
output_filepath = "/Users/axelahdritz/Desktop/thesis/transcripts/"

word_filepath = "/Users/axelahdritz/Desktop/thesis/transcripts/word_transcripts/"
sentence_filepath = "/Users/axelahdritz/Desktop/thesis/transcripts/sentence_transcripts/"
full_filepath = "/Users/axelahdritz/Desktop/thesis/transcripts/full_transcripts/"

def get_date(audio_file_name):
    year = audio_file_name[:2]
    month = audio_file_name[2:4]
    day = audio_file_name[4:6]
    date = month + '/' + day + '/' + year
    return date

def word_counter(words):
    count = 0
    tokens = words.split()
    for word in tokens:
        count += 1
    return count

def mp3_to_wav(audio_file_name):
    if audio_file_name.split('.')[1] == 'mp3':    
        sound = AudioSegment.from_mp3(audio_file_name)
        audio_file_name = audio_file_name.split('.')[0] + '.wav'
        sound.export(audio_file_name, format="wav")

def frame_rate_channel(audio_file_name):
    print(audio_file_name)
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        print(frame_rate, channels)
        return frame_rate,channels

def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name, timeout=None)

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

def google_transcribe(audio_file_name, audio_file_path, bucketname):
    file_name = audio_file_path

    frame_rate, channels = frame_rate_channel(file_name)
    if channels > 1:
        stereo_to_mono(file_name)
        
    bucket_name = bucketname 
    destination_blob_name = audio_file_name

    upload_blob(bucket_name, file_name, destination_blob_name)

    gcs_uri = 'gs://machine_feed' + "/" + audio_file_name
    transcript = ''

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    metadata = speech.RecognitionMetadata()
    metadata.interaction_type = speech.RecognitionMetadata.InteractionType.DISCUSSION
    metadata.microphone_distance = (
        speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD
    )
    metadata.recording_device_type = (
            speech.RecognitionMetadata.RecordingDeviceType.OTHER_OUTDOOR_DEVICE
    )

    config = speech.RecognitionConfig(
        encoding= speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='en-US',
        alternative_language_codes=['sv-SE'],
        enable_automatic_punctuation=True,
        enable_word_time_offsets = True,
        enable_word_confidence=True,
        metadata=metadata,
    )

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1000000)
    result = response.results

    delete_blob(bucket_name, destination_blob_name)
    
    return result

def data_transcript(word_filepath, transcript_filename, audio_file_name, date_recorded, google_results):
    new_filename = 'words_' + transcript_filename
    column_names = ["word","word_punct","confidence", "start_time","end_time","sentence","sentence_confidence","date", "original_audio"]
    df = pd.DataFrame(columns = column_names)
    
    for r in google_results:
        sentence_transcript = r.alternatives[0].transcript
        sentence_confidence = r.alternatives[0].confidence
        words_info = r.alternatives[0].words
        for word_info in words_info:
            current = word_info.word
            speaker_tag = word_info.speaker_tag
            un_punctuated = current.translate(str.maketrans('', '', string.punctuation))
            confidence = word_info.confidence
            start_time = word_info.start_time
            end_time = word_info.end_time
            df.loc[len(df.index)] = [un_punctuated.lower(), current, confidence, start_time, end_time, sentence_transcript, sentence_confidence, date_recorded, audio_file_name]

    csv_filepath = os.path.join(word_filepath, new_filename)
    df.to_csv(csv_filepath, index=True)


def sentence_data(sentence_filepath, transcript_filename, audio_file_name, date_recorded, google_results):
    new_filename = 'sentence_' + transcript_filename
    column_names = ["sentence", "sentence_confidence","start_time","end_time", "date", "original_audio"]
    df = pd.DataFrame(columns = column_names)
   
    for r in google_results:
        sentence_transcript = r.alternatives[0].transcript
        sentence_confidence = r.alternatives[0].confidence
        words_info = r.alternatives[0].words
        i = 0
        length = len(words_info) - 1
        for word_info in words_info:
            if i == 0:
                start_time = word_info.start_time
            if i == length:
                end_time = word_info.end_time
            i += 1
        df.loc[len(df.index)] = [sentence_transcript, sentence_confidence, start_time, end_time, date_recorded, audio_file_name]

    csv_filepath = sentence_filepath + new_filename
    df.to_csv(csv_filepath, index=True)

    
def transcript_data(transcript_filepath, transcript_filename, audio_file_name, date_recorded, google_results):
    new_filename = 'transcript_' + transcript_filename
    column_names = ["transcript", "word_count", "date", "original_audio"]
    df = pd.DataFrame(columns = column_names)
    
    full_transcript = ''
    for r in google_results:
        full_transcript += r.alternatives[0].transcript + ' '

    word_count = word_counter(full_transcript)
    
    df.loc[len(df.index)] = [full_transcript, word_count, date_recorded, audio_file_name]
    
    csv_filepath = transcript_filepath + new_filename
    df.to_csv(csv_filepath, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', required=True)
    args = parser.parse_args()

    audio_file_path = args.audio_file
    audio_file_name = audio_file_path.split('/')[-1]
    print(audio_file_name)

    if ".wav" in audio_file_path:
        print(audio_file_name)
        word_filename_csv = audio_file_name.split('.')[0] + '.csv'
        transcript_filename_csv = audio_file_name.split('.')[0] + '.csv'
        sentence_filename_csv = audio_file_name.split('.')[0] + '.csv'
        date_recorded = get_date(audio_file_name)
        print('Getting results...')
        google_results = google_transcribe(audio_file_name,audio_file_path,bucketname)
        data_transcript(word_filepath, word_filename_csv, audio_file_name, date_recorded, google_results)
        transcript_data(full_filepath, transcript_filename_csv, audio_file_name, date_recorded, google_results)
        sentence_data(sentence_filepath, sentence_filename_csv, audio_file_name, date_recorded, google_results)
        '''
        response_json = type(google_results).to_json(google_results)
        with open(transcript_filename, 'w') as jsonfile:
            json.dump(response_json, jsonfile)
        print('Results Written!')
        print(' ')
        '''
