import os
from faster_whisper import WhisperModel
import edge_tts
import asyncio



class VoiceAssistant:
    def __init__(self,model_size:str='small'):

        self.detected_language=None
        self.language_probability=None
        self.language_threshold = 0.5
        self.model_size=model_size

        self.stt_model=WhisperModel(
            self.model_size,
            device='cpu',
            compute_type='int8')
        
        self.audio_format='wav'
        self.sample_rate=16000
        
        self.temp_audio_dir='temp_audio'

        os.makedirs(
            self.temp_audio_dir,
              exist_ok=True
              )

        self.arabic_voice='ar-EG-ShakirNeural'
        self.english_voice='en-US-GuyNeural'


    def save_audio(self, audio_data, file_name:str='input.wav') -> str:

        file_path=os.path.join(
            self.temp_audio_dir,
            file_name
            )

        with open(file_path,'wb')as audio_file:
            audio_file.write(audio_data)

        return file_path
    



    def speech_to_text(self,audio_file:str) -> str:

        segments,info=self.stt_model.transcribe(
            audio_file,beam_size=5
            )
        

        self.detected_language=info.language
        self.language_probability=info.language_probability

        text=' '.join(segment.text for segment in segments)
        
        return text.strip()
        


    def text_to_speech(self, text:str)-> str:
        # to make sure that the text is not empty
        if not text.strip():
            raise ValueError(
                "text can't be empty"
                )

        # If language detection confidence is low,
        # assume Arabic as the default language.
        if (self.language_probability is None
            or self.language_probability < self.language_threshold):
            voice = self.arabic_voice

        elif self.detected_language == "en":
            voice = self.english_voice

        else:
            voice = self.arabic_voice


        output_path=os.path.join(
            self.temp_audio_dir,
            'output.mp3'
            )

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice
            )
        
        try:
            asyncio.run(
                communicate.save(output_path)
                )

        except Exception as e:
            raise RuntimeError(
                f'Failed to generate speech:{e}'
                ) from e
        
        return output_path 




    def clean_up(self) -> None:
        """clean up the previous audio files"""
        files=os.listdir(self.temp_audio_dir)

        for file in files:
            if file.endswith(('.wav','.mp3')):
                file_path=os.path.join(
                    self.temp_audio_dir,
                    file
                    )
                try:
                    os.remove(file_path)
                except OSError as e:
                    print (
                        f" Failed to delete {file_path}. Reason: {e}"
                        )