from motion.motion_service import MotionService, MotorConfig
from audio.audio_service import AudioService
from stt.stt_service import SttService, SttConfig

if __name__ == "__main__":
    motors = {
        "mouth": MotorConfig(in1=17, in2=27, en=22, forward_speed=82, reverse_speed=55, neutral_return_time=0.04),
        "tail": MotorConfig(in1=23, in2=24, en=25, forward_speed=72, reverse_speed=48, neutral_return_time=0.03),
        "body": MotorConfig(in1=5, in2=6, en=12, forward_speed=68, reverse_speed=45, neutral_return_time=0.03),
    }

    motion = MotionService(motors=motors)
    motion.initialize()

    audio = AudioService()
    audio.initialize()

    stt_config = SttConfig(
        whisper_binary="./stt/whisper.cpp/build/bin/whisper-cli",
        model_path="./stt/whisper.cpp/models/ggml-base.en.bin",
        threads=2,
        wake_words=("fish", "hey fish"),
    )
    
    stt = SttService(stt_config)
    stt.initialize()
    
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
