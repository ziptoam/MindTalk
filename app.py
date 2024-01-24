import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gradio as gr
from dataclasses import asdict
from lmdeploy import turbomind as tm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import warnings
warnings.filterwarnings('ignore')
from openxlab.model import download

download(model_repo='ziptoam/mindtalk4bit', 
        output='hf_merge')
os.system("lmdeploy convert  internlm-chat-7b /home/xlab-app-center/hf_merge --model-format awq --group-size 128 --dst_path /home/xlab-app-center/workspace")
model_path = "/home/xlab-app-center/workspace"
user_prompt = ":{user}\n"
robot_prompt = ":{robot}<eoa>\n"
cur_query_prompt = ":{user}<eoh>\n:"

class ChatModel:
    def __init__(self):
        self.tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')

    def _prompt(self, query):
        generator = self.tm_model.create_instance()
        prompt = self.tm_model.model.get_prompt(query)
        input_ids = self.tm_model.tokenizer.encode(prompt)
        
        for outputs in generator.stream_infer(session_id=0, input_ids=[input_ids]):
            res, tokens = outputs[0]

        response = self.tm_model.tokenizer.decode(res.tolist())
        return response

    def get_response(self, question: str, chat_history: list = []):
        if question is None or len(question) < 1:
            return "", chat_history
        try:
            question = question.replace(" ", '')
            response = self._prompt(question)
            chat_history.append((question, response))
            return "", chat_history
        except Exception as e:
            return e, chat_history

# Instantiate chat model object
chat_model = ChatModel()

# Create a Gradio interface
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # Display page title
            gr.Markdown("""<h1><center>SmartCure_MindTalk</center></h1>
                <center>智愈心语-开解唠嗑员</center>
                """)
    with gr.Row():
        with gr.Column(scale=4):
            # Create a chatbot object
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # Create a textbox for inputting prompts/questions
            msg = gr.Textbox(label="Prompt/Question")

            with gr.Row():
                # Create a submit button
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # Create a clear button to clear chatbot content
                clear = gr.ClearButton(components=[chatbot], value="Clear console")
    
        db_wo_his_btn.click(chat_model.get_response, inputs=[msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""Reminder:<br>
    1. 心理互助问答，非心理咨询，仅为心理知识分享；
    2. 如果加载比较慢，可以多等一等；
    2. 数据来源于壹心理清洗，壹心理已过滤任何发布者信息部分，仅使用文本.<br>
    """)

gr.close_all()

# Launch the interface
demo.launch()
