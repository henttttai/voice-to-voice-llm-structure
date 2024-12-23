import json
import requests
import torchaudio
import threading
import winsound

from cosyvoice.cli.cosyvoice import CosyVoice
from queue import Queue
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class V2VLLM:
    def __init__(self,cosyvoice_dir, sensevoice_dir, ollama_api):
        cosyvoice_dir = cosyvoice_dir
        sensevoice_dir = sensevoice_dir
        api = ollama_api

        self.cosyvoice = CosyVoice(cosyvoice_dir)
        self.sencevoice = AutoModel(
                model=sensevoice_dir,
                trust_remote_code=True,
                vad_model="fsmn-vad",
                remote_code="model.py",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
                hub="hf",
            )

        self.url_generate = api

        self.context = []

    def start(self,input_audio_queue):
        while True:
            audio_path = input_audio_queue.get()
            self.inference(audio_path)

    def inference(self,audio_path):
        context_wenben = ""
        res = self.sencevoice.generate(
            input=audio_path,
            cache={},
            language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])

        context_wenben += "prompt:" + text + "\n"


        if text == "":
            return 0

        user_context_dict = {
            "role": "user",
            "content": f"{text}"
        }

        self.context.append(user_context_dict)

        # data = {
        #     "model": "qwen2.5:1.5b",
        #     "prompt": f"你所收到的prompt是由语音转化而成，故而存在一定的同义词错误，你在识别的时候需要注意这个问题（回答的时候不要涉及到这句话，直接回复'：'后的文本）/"
        #               f"用户的prompt:{text}",
        #     "stream": False
        # }

        data = {
            "model": "qwen2.5:1.5b",
            "messages": self.context,
            "stream": False
        }

        print("--------------------------------开始生成回答-------------------------------------")
        res = self.get_response(self.url_generate, data)
        print("--------------------------------生成回答结束-------------------------------------")

        context_wenben += "answer:" + res + "\n"
        llm_context_dict = {
            "role": "assistant",
            "content": f"{res}"
        }

        self.context.append(llm_context_dict)

        with open("chat_history.txt", "a", encoding="utf-8") as file:
            file.write(context_wenben)


        audio_queue = Queue()

        # 创建并启动音频生成线程
        generator_thread = threading.Thread(target=self.audio_generator, args=(res, audio_queue))
        # 创建并启动音频播放线程
        player_thread = threading.Thread(target=self.audio_player, args=(audio_queue,))

        generator_thread.start()
        player_thread.start()

        # 等待两个线程完成
        generator_thread.join()
        player_thread.join()


    def get_response(self, url, data):
        response = requests.post(url, json=data)
        response_dict = json.loads(response.text)
        response_content = response_dict["message"]["content"]
        return response_content


    def audio_generator(self, text, audio_queue):
        for i, j in enumerate(self.cosyvoice.inference_sft(f'{text}', '中文女', stream=True)):
            wav_path = f'wav_output/zero_shot_{i}.wav'
            torchaudio.save(wav_path, j['tts_speech'], 22050)
            audio_queue.put(wav_path)
        audio_queue.put(None)  # 发送结束信号


    def audio_player(self, audio_queue):
        while True:
            wav_path = audio_queue.get()
            if wav_path is None:  # 检查结束信号
                break
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)