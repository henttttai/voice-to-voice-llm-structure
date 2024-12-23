import threading
import time
import wave
import pyaudio
import webrtcvad

from queue import Queue

import winsound
import pygame
from joblib.externals.loky import set_loky_pickler


class AudioSteam:
    def __init__(self, audio_rate, audio_channels, chunk, vad_mode, input_dir, no_speech_threshold):
        self.audio_rate = audio_rate
        self.audio_channels = audio_channels
        self.chunk = chunk
        self.vad_mode = vad_mode
        self.input_dir = input_dir
        self.no_speech_threshold = no_speech_threshold

        self.audio_num = 0
        self.sv_check = False

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.vad_mode)

        self.if_record = True

    def audio_recorder_stream(self, input_audio_queue:Queue):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=self.audio_channels,
                        rate=self.audio_rate,
                        input=True,
                        frames_per_buffer=self.chunk)

        audio_buffer = []
        segments_to_save = []
        print("音频录制已开始")

        while self.if_record:
            data = stream.read(self.chunk)
            audio_buffer.append(data)

            # 每 0.5 秒检测一次 VAD
            if len(audio_buffer) * self.chunk / self.audio_rate >= 0.5:
                # 拼接音频数据并检测 VAD
                raw_audio = b''.join(audio_buffer)
                vad_result = self.check_vad_activity(raw_audio)

                if vad_result:
                    print("检测到语音活动")
                    segments_to_save.append((raw_audio, time.time()))
                else:
                    print("静音中...")

                audio_buffer = []  # 清空缓冲区

            # 检查无效语音时间
            if segments_to_save and time.time() - segments_to_save[-1][-1] > self.no_speech_threshold:
                audio_frames = [seg[0] for seg in segments_to_save]
                segments_to_save.clear()

                if self.sv_check:
                    audio_path = f"{self.input_dir}/sv_check/user_sv_check.wav"
                else:
                    self.audio_num += 1
                    audio_path = f"{self.input_dir}/user_input_{self.audio_num}.wav"

                self.save_audio(audio_path, audio_frames)
                input_audio_queue.put(audio_path)


    # 检测 VAD 活动
    def check_vad_activity(self, audio_data):
        # 将音频数据分块检测
        num, rate = 0, 0.4
        step = int(self.audio_rate * 0.02)  # 20ms 块大小
        flag_rate = round(rate * len(audio_data) // step)

        for i in range(0, len(audio_data), step):
            chunk = audio_data[i:i + step]
            if len(chunk) == step:
                if self.vad.is_speech(chunk, sample_rate=self.audio_rate):
                    num += 1

        if num > flag_rate:
            return True
        return False


    def save_audio(self, audio_output_path, audio_frame):
        wf = wave.open(audio_output_path, 'wb')
        wf.setnchannels(self.audio_channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(self.audio_rate)
        wf.writeframes(b''.join(audio_frame))
        wf.close()
        print(f"音频保存至 {audio_output_path}")




if __name__ == '__main__':
    AUDIO_RATE = 16000  # 音频采样率
    AUDIO_CHANNELS = 1  # 单声道
    CHUNK = 1024  # 音频块大小
    VAD_MODE = 0  # VAD 模式 (0-3, 数字越大越敏感)
    INPUT_DIR = "wav_input"  # 用户语音输入目录
    OUTPUT_DIR = "wav_output"  # LLM语音输出目录
    NO_SPEECH_THRESHOLD = 1  # 无效语音阈值，单位：秒

    audio_queue = Queue()

    audio_stream = AudioSteam(AUDIO_RATE, AUDIO_CHANNELS, CHUNK, VAD_MODE, INPUT_DIR, NO_SPEECH_THRESHOLD)

    audio_threading = threading.Thread(target=audio_stream.audio_recorder_stream, args=(audio_queue, ))
    audio_threading.start()

    time.sleep(20)

    audio_stream.if_record = False


