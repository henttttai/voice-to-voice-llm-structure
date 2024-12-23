import threading

from queue import Queue

from utils.audio_part import AudioSteam
from utils.llm_part import V2VLLM


AUDIO_RATE = 16000  # 音频采样率
AUDIO_CHANNELS = 1  # 单声道
CHUNK = 1024  # 音频块大小
VAD_MODE = 0  # VAD 模式 (0-3, 数字越大越敏感)
INPUT_DIR = "wav_input"  # 用户语音输入目录
OUTPUT_DIR = "wav_output"  # LLM语音输出目录
NO_SPEECH_THRESHOLD = 1  # 无效语音阈值，单位：秒

cosyvoice_dir = "E:/cosvoice/CosyVoice/pretrained_models/CosyVoice-300M" # cosyvoice模型地址
sensevoice_dir = "E:/cosvoice/SenseVoice/pretrained_model/SenseVoiceSmall" # sencevoice模型地址
ollama_api = "http://localhost:11434/api/chat" # ollama的api地址

v2vllm = V2VLLM(cosyvoice_dir, sensevoice_dir, ollama_api)
audio_stream = AudioSteam(AUDIO_RATE, AUDIO_CHANNELS, CHUNK, VAD_MODE, INPUT_DIR, NO_SPEECH_THRESHOLD)

input_audio_queue = Queue()

audio_threading = threading.Thread(target=audio_stream.audio_recorder_stream, args=(input_audio_queue,))
inference_threading = threading.Thread(target=v2vllm.start, args=(input_audio_queue, ))

audio_threading.start()
inference_threading.start()