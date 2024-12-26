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
    def __init__(self,gptsovits_api, sensevoice_dir, ollama_api):
        self.tools = TOOLS

        self.sencevoice = AutoModel(
                model=sensevoice_dir,
                trust_remote_code=True,
                vad_model="fsmn-vad",
                remote_code="model.py",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
                hub="hf",
            )

        self.url_ollama = ollama_api
        self.url_gptsovits = gptsovits_api

        self.history_context = []

        self.outputwav_num = 0

        change_gpt_url = "http://127.0.0.1:9880/set_gpt_weights"
        change_sovits_url = "http://127.0.0.1:9880/set_sovits_weights"

        gpt_weights = "E:/TTS/GPT-SoVITS/GPT_weights_v2/井芹仁菜V3-e15.ckpt"
        sovits_weights = "E:/TTS/GPT-SoVITS/SoVITS_weights_v2/井芹仁菜V3_e100_s1800.pth"

        requests.get(f"{change_gpt_url}?weights_path={gpt_weights}")
        requests.get(f"{change_sovits_url}?weights_path={sovits_weights}")

    def start(self,input_audio_queue, if_tool):
        while True:
            audio_path = input_audio_queue.get()
            self.v2v_inference(audio_path, if_tools=if_tool)

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

        response_middel = self.get_response(self.url_ollama, data, if_tooluse=True)

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

        response = self.get_response(self.url_ollama ,data, True)

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
        res = self.get_response(self.url_ollama, data)
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

        if in_text == "":
            return None

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

        json_data = {
            "text": f"{text}",
            "text_lang": "zh",
            "ref_audio_path": "E:/TTS/nina3/nina3 (1).wav",  # str.(required) reference audio path
            "aux_ref_audio_paths": [],  # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
            "prompt_text": "ももかさんってそうなんですね。いますよね、つゆ多めがいいって人。", # str.(optional) prompt text for the reference audio
            "prompt_lang": "ja",  # str.(required) language of the prompt text for the reference audio
            "top_k": 15,  # int. top k sampling
            "top_p": 1,  # float. top p sampling
            "temperature": 1,  # float. temperature for sampling
            "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
            "batch_size": 1,  # int. batch size for inference
            "batch_threshold": 0.75,  # float. threshold for batch splitting.
            "split_bucket": True,  # bool. whether to split the batch into multiple buckets.
            "speed_factor": 1.0,  # float. control the speed of the synthesized audio.
            "streaming_mode": False,  # bool. whether to return a streaming response.
            "seed": -1,  # int. random seed for reproducibility.
            "parallel_infer": True,  # bool. whether to use parallel inference.
            "repetition_penalty": 1.35  # float. repetition penalty for T2S model.
        }


        res = requests.post(self.url_gptsovits, json=json_data)

        wav_path = f"wav_output/output_{self.outputwav_num}.wav"
        self.outputwav_num += 1

        with open(wav_path, "wb") as f:
            f.write(res.content)

        audio_queue.put(wav_path)
        audio_queue.put(None)  # 发送结束信号



    def audio_player(self, audio_queue):
        while True:
            wav_path = audio_queue.get()
            if wav_path is None:  # 检查结束信号
                break
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)

if __name__ == '__main__':
    sensevoice_dir = "E:/TTS/cosvoice/SenseVoice/pretrained_model/SenseVoiceSmall"  # sencevoice模型地址
    ollama_url = "http://localhost:11434/api/chat"  # ollama的api地址
    gptsovits_url = "http://127.0.0.1:9880/tts"

    test = V2VLLM(gptsovits_url, sensevoice_dir, ollama_url)
    test.v2v_inference("E:/TTS/new_structure/wav_input/user_input_10.wav", )