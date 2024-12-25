import json
import requests
import torchaudio
import threading
import winsound
import transformers

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from queue import Queue
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from llm_tools import *


class V2VLLM:
    def __init__(self,cosyvoice_dir, sensevoice_dir, ollama_api):

        self.tools = TOOLS

        self.cosyvoice = CosyVoice2(cosyvoice_dir)
        self.sencevoice = AutoModel(
                model=sensevoice_dir,
                trust_remote_code=True,
                vad_model="fsmn-vad",
                remote_code="model.py",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
                hub="hf",
            )

        self.url_generate = ollama_api

        self.history_context = []

    def start(self,input_audio_queue):
        while True:
            audio_path = input_audio_queue.get()
            self.v2v_inference(audio_path, if_tools=False)


    def transformers_inference(self):
        return 0


    def ollama_tooluse_inference(self, text):
        context_wenben = ""

        context_wenben += "prompt:" + text + "\n"

        if text == "":
            return 0

        user_context_dict = {
            "role": "user",
            "content": f"{text}"
        }

        self.history_context.append(user_context_dict)

        messages = [
            {"role": "system",
             "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
            {"role": "user", "content": text},
        ]

        data = {
            "model": "qwen2.5:1.5b",
            "messages": messages,
            "tools": self.tools,
            "stream": False,
        }

        response_middel = self.get_response(self.url_generate, data, if_tooluse=True)

        messages.append(response_middel["message"])

        if tool_calls := messages[-1].get("tool_calls", None):
            for tool_call in tool_calls:
                if fn_call := tool_call.get("function"):
                    fn_name: str = fn_call["name"]
                    fn_args: dict = fn_call["arguments"]

                    fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

                    messages.append({
                        "role": "tool",
                        "name": fn_name,
                        "content": fn_res,
                    })

        data = {
            "model": "qwen2.5:1.5b",
            "messages": messages,
            "tools": self.tools,
            "stream": False,
        }

        response = self.get_response(self.url_generate,data, True)

        messages.append(response["message"])
        res = messages[-1]["content"]

        context_wenben += "answer:" + res + "\n"
        llm_context_dict = {
            "role": "assistant",
            "content": f"{res}"
        }

        self.history_context.append(llm_context_dict)

        with open("chat_history.txt", "a", encoding="utf-8") as file:
            file.write(context_wenben)

        return res

    def ollama_inference(self,text):
        context_wenben = ""

        context_wenben += "prompt:" + text + "\n"

        if text == "":
            return 0

        user_context_dict = {
            "role": "user",
            "content": f"{text}"
        }

        self.history_context.append(user_context_dict)

        data = {
            "model": "qwen2.5:1.5b",
            "messages": self.history_context,
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

        self.history_context.append(llm_context_dict)

        with open("chat_history.txt", "a", encoding="utf-8") as file:
            file.write(context_wenben)

        return res

    def v2v_inference(self,audio_path, if_tools=False):
        in_text = self.sencevoice.generate(
            input=audio_path,
            cache={},
            language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        in_text = rich_transcription_postprocess(in_text[0]["text"])

        if if_tools:
            out_text = self.ollama_tooluse_inference(in_text)
        else:
            out_text = self.ollama_inference(in_text)

        audio_queue = Queue()

        # 创建并启动音频生成线程
        generator_thread = threading.Thread(target=self.audio_generator, args=(out_text, audio_queue))
        # 创建并启动音频播放线程
        player_thread = threading.Thread(target=self.audio_player, args=(audio_queue,))

        generator_thread.start()
        player_thread.start()

        # 等待两个线程完成
        generator_thread.join()
        player_thread.join()

    def get_response(self, url, data, if_tooluse=False):
        response = requests.post(url, json=data)
        response_dict = json.loads(response.text)

        if if_tooluse:
            response_content = response_dict
            return response_content
        else:
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
