import speech_recognition as sr
r= sr.Recognizer()
print("Print Talk")
with sr.Microphone() as source:
    
    audio_data=r.record(source, duration=5)
    print("Recognizing...")
    audio = r.listen(source)
    text = r.recognize_google(audio)
    print(text,format(text))
    