import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gradio as gr
from dataclasses import asdict
from lmdeploy import turbomind as tm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import warnings

from typing import Optional, List, Mapping, Any


from llama_index import (
     VectorStoreIndex,
     ServiceContext, 
     SimpleDirectoryReader, 
     ListIndex,
     StorageContext,
     load_index_from_storage
)
from llama_index.embeddings import LangchainEmbedding
from llama_index.prompts import PromptTemplate
from llama_index.llms import CustomLLM, CompletionResponse, LLMMetadata

current_path = os.path.abspath(__file__)
print("path:\n")
print(current_path)
warnings.filterwarnings('ignore')
from openxlab.model import download

download(model_repo='OpenLMLab/InternLM-chat-7b', 
        output='internlm-chat-7b')
# os.system("lmdeploy convert  internlm-chat-7b /home/xlab-app-center/internlm-7b-chat --model-format awq --group-size 128 --dst_path /home/xlab-app-center/workspace")
# model_path = "/home/xlab-app-center/workspace"

# tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')
# generator = tm_model.create_instance()
model_path = "/home/xlab-app-center/internlm-chat-7b"
print("正在从本地加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
model = self.model.eval()
print("完成本地模型的加载")

class OurLLM(CustomLLM):
    # 基于本地 InternLM 自定义 LLM 类

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window, num_output=num_output
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # response = pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]
        # 在这个函数里将prompt输入给model
        
        print("prompt生成：")
        print(prompt)

        # input_ids = tm_model.tokenizer.encode(prompt)
        
        # for outputs in generator.stream_infer(session_id=0, input_ids=[input_ids]):
        #     res, tokens = outputs[0]

        # response = tm_model.tokenizer.decode(res.tolist())
        response, history = self.model.chat(tokenizer, prompt , history=messages)
        
        print("返回结果生成：")
        # only return newly generated tokens
        print(response)
        
        return CompletionResponse(text=response)
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError()


llm = OurLLM()


# set context window size
context_window = 2048
# set number of output tokens
num_output = 500

service_context = ServiceContext.from_defaults(
    llm=llm, 
    embed_model="local",
    context_window=context_window, 
    num_output=num_output
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    print("正在载入数据")
    documents = SimpleDirectoryReader("./data").load_data()
    print("正在建立index")
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)



# Query and print response
print("正在搜索结果")


qa_prompt_tmpl_str = (
     "你是一个AI-心理治疗师，你将根据下面的条件，回答一个心理问题.\n"
     "- 下面会提供给你一个关于心理治疗辅导的案例.\n"
     "- 你必须给出一个对治疗这个心理问题有用的信息，其他的无用信息不要多说：\n"
     "- 你需要根据案例的回答提取出对回答问题一些有用信息，但是不要说出任何与案例有关的详细信息.\n"
     "- 如果这个心理辅导案例与问题无关，请忽略它，并给出你自己的合理回答\n"
     "- 必须要回答中文，不能说英文.\n"
     "- 不要进行重复回答.\n"
     "案例相关的信息如下所示：\n"
     "---------------------\n"
     "{context_str}\n"
     "---------------------\n"
     "基于上面的案例和你自己学到的相关的心理知识,"
     "用专业心理治疗师的口吻，回答下面这个问题.\n"
     "问题: {query_str}\n"
     "请你根据这个心理问题给出回答: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

refine_prompt_tmpl_str = (
     "你是一个AI-心理治疗师，你将根据下面的条件，回答一个心理问题.\n"
     "回答具有如下要求：\n"
     "- 必须要回答中文，不能说英文.\n"
     "- 不要进行重复回答.\n"
     "这个心理问题是: {query_str}\n"
     "之前已经有一个心理治疗师已经给出了一些见解: {existing_answer}\n"
     "现在你需要结合上一个心理治疗师的见解，并结合下面的一些条件给出你的合理回答.\n"
     "如果需要的话，你可以结合下面这个这个心理治疗辅导的案例.\n"
     "------------\n"
     "{context_msg}\n"
     "------------\n"
     "通过给出的心理辅导的案例，让你的回答变得更加完美\n"
     "如果你觉得这个心理辅导案例对回答这个心理问题没有帮助，请返回原来的答案，不要给出与心理问题毫无关联的回答.\n"
     "新的回答："
)

refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)

query_engine = index.as_query_engine(similarity_top_k=2, text_qa_template=qa_prompt_tmpl, refine_template=refine_prompt_tmpl)



class ChatModel:
    def __init__(self):
        print("chatbot初始化")

    def get_response(self, question: str, chat_history: list = []):
        if question is None or len(question) < 1:
            return "", chat_history
        try:
            question = question.replace(" ", '')
            response = query_engine.query(question)
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
