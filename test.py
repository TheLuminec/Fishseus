audio.start_capture()

while True:
    wav = audio.record_until_silence("latest_speech.wav")

    if not wav:
        continue

    result = stt.transcribe_file(wav)
    print(result.text)

    if result.wake_detected:
        print("Wake detected!")
        print("Command:", result.command_text)
        motion.wiggle(cycles=1)
        motion.speak_text_placeholder(duration_s=1.5)
