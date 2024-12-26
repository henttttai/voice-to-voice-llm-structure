import threading

from queue import Queue

from utils.audio_part import AudioStream
from utils.llm_part_gptsovits import V2VLLM


AUDIO_RATE = 16000  # 音频采样率
AUDIO_CHANNELS = 1  # 单声道
CHUNK = 1024  # 音频块大小
VAD_MODE = 0  # VAD 模式 (0-3, 数字越大越敏感)
INPUT_DIR = "wav_input"  # 用户语音输入目录
OUTPUT_DIR = "wav_output"  # LLM语音输出目录
NO_SPEECH_THRESHOLD = 1  # 无效语音阈值，单位：秒

if_tool = False # 是否要使用tool ，tool的使用会调用两次api，推理速度会翻倍

 # cosyvoice模型地址
sensevoice_dir = "E:/TTS/cosvoice/SenseVoice/pretrained_model/SenseVoiceSmall" # sencevoice模型地址
ollama_url = "http://localhost:11434/api/chat" # ollama的api地址
gptsovits_url = "http://127.0.0.1:9880/tts"

try:
    v2vllm = V2VLLM(gptsovits_url, sensevoice_dir, ollama_url)
    audio_stream = AudioStream(AUDIO_RATE, AUDIO_CHANNELS, CHUNK, VAD_MODE, INPUT_DIR, NO_SPEECH_THRESHOLD)

    input_audio_queue = Queue()

    audio_threading = threading.Thread(target=audio_stream.audio_recorder_stream, args=(input_audio_queue,))
    inference_threading = threading.Thread(target=v2vllm.start, args=(input_audio_queue, if_tool, ))

    print("启动录音线程")
    audio_threading.start()
    print("启动推理线程")
    inference_threading.start()


    test = 1
except KeyboardInterrupt:
    print("录制停止中...")
    print("录制已停止")


