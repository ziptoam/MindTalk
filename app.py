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
warnings.filterwarnings('ignore')
from openxlab.model import download

download(model_repo='OpenLMLab/InternLM-chat-7b', 
        output='internlm-chat-7b')

model_path = "/home/xlab-app-center/internlm-chat-7b"

print("正在从本地加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
model = model.eval()
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
          system_prompt = ""
          print(prompt)
          messages = [(system_prompt, '')]
          response, history = model.chat(tokenizer, prompt, history=messages)
          
          # only return newly generated tokens
          print(response)
          
          return CompletionResponse(text=response)
     
     def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
          raise NotImplementedError()

# define our LLM
llm = OurLLM()

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
     "用专业心理治疗师的口吻，回答下面这个问题，同时这个问题里会包含一些与用户心理问题相关的聊天记录，你要利用已有的信息\n"
     "问题: {query_str}\n"
     "请你根据这个心理问题给出回答: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

refine_prompt_tmpl_str = (
     "你是一个AI-心理治疗师，你将根据下面的条件，回答一个心理问题.\n"
     "回答具有如下要求：\n"
     "- 必须要回答中文，不能说英文.\n"
     "- 不要进行重复回答.\n"
     "- 这个问题里会包含一些与用户心理问题相关的聊天记录，你要利用已有的信息\n"
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

# while 1 :
#      input_question = input("请输入输入问题: ")
#      response = query_engine.query(input_question)
#      print(response)

def get_response_from_llamaIndex(question) : 
     response = str(query_engine.query(question))
     print(response)
     return response



class Model_center():
    """
    存储问答 Chain 的对象 
    """
    def __init__(self):
        print("Model_center启动")

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            
            chat_history.append(
                (question, get_response_from_llamaIndex(question)))
            return "", chat_history
        except Exception as e:
            return e, chat_history

# Instantiate chat model object
chat_model = Model_center()

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
    
        db_wo_his_btn.click(chat_model.qa_chain_self_answer, inputs=[msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""Reminder:<br>
    1. 心理互助问答，非心理咨询，仅为心理知识分享；
    2. 如果加载比较慢，可以多等一等；
    2. 数据来源于壹心理清洗，壹心理已过滤任何发布者信息部分，仅使用文本.<br>
    """)

gr.close_all()

# Launch the interface
demo.launch()




# 前不久和男朋友分手了，导致现在心情很抑郁怎么办？
# link from https://stackoverflow.com/questions/76625768/importerror-cannot-import-name-customllm-from-llama-index-llms