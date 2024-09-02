import requests
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import json
from langchain_community.llms import Ollama
from baiduqianfan.prompt import PromptTempalte
from baiduqianfan.selector import Selector_template
from baiduqianfan.loader import LoaderTemplate
# def get_access_token():
#     url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=q9IGmb6ZjDxbQjyGvXtwRBYn&client_secret=35EQhSIKW5nS1f02VVGsk0LZKGGFdt6r"
#     payload = json.dumps("")
#     headers = {
#         'Content-Type': 'application/json',
#         'Accept': 'application/json'
#     }
#     response = requests.request("POST", url, headers=headers, data=payload)
#
#     return response.json().get("access_token")


def get_baidu_api_url():
    # api_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_8b?access_token=" + get_access_token()
    api_url = "http://localhost:11343/api/chat"
    return api_url


class SeparatedListOutput(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")

# 自定义大模型连接方式
def llm(prompt):
    api_url = get_baidu_api_url()
    headers = {
        'Content-Type': 'application/json'
    }

    data = json.dumps({
        "model": "llama3",
        "options": {
            "temperature": 0.  # 为0表示不让模型自由发挥，输出结果相对较固定，>0的话，输出的结果会比较放飞自我
        },
        "stream": False,
        "messages": [{
            "role": "user",
            "content": prompt
        }]  # 对话列表
    })
    response = requests.request("POST", api_url, headers=headers, data=data)

    return response.text
#函数示例
def add_sum(x,y):
    z = x + y
    print(z)
    return z

def create_name():
    # prompt = PromptTemplate.from_template(
    #     "你是一个{name}，请模仿示例起3个{country}名字，比如男孩经常被叫做{boy},女孩经常被叫做{girl}"
    # ).format(name="起名大师", country="中国特色", boy="志延", girl="紫萱")
    #实例化提示词模板类
    prompt_instance = PromptTempalte("chat")
    #获取模板类对象
    prompt_object = prompt_instance.process_input()
    #获取模板
    example_prompt = "请根据古诗词再帮我起一个男孩名和一个女孩名"
    prompt = prompt_object("周公",example_prompt)
    # 示例组
    example_group = [
        {"input":"人生到处知何似，应似飞鸿踏雪泥","output1":"何应","output2":"何雪"},
        {"input":"倚杖柴门外，临风听慕蝉","output1":"临风","output2":"慕蝉"},
        {"input":"竹喧归浣女，莲动下渔舟","output1":"宣舟","output2":"宣瑜"}
    ]
    #最终提示词模板
    complex_prompt = PromptTemplate(
        input_variables = ["input","output1","output2"],
        template = "参考示例:诗句:{input}\n男孩名:{output1}\n女孩名:{output2}"
    )
    #使用选择提示组
    selectors = Selector_template("base_similarity")
    selector_object = selectors.process_input()
    example_selector = selector_object(example_group,complex_prompt)
    final_prompt = example_selector.format(adjective=prompt)
    get_baidu_api_url()
    #load加载知识库
    load_template = LoaderTemplate("md")
    load_object = load_template.process_input()
    data = load_object("knowledge/create_name.md")
    chunks = processes_data(data)
    #拼接组合词
    prompt = concat_prompt(chunks,final_prompt)
    #返回结果
    response = llm(prompt)
    output_parser = SeparatedListOutput()
    parsed_response = output_parser.parse(response)
    response_str = '\n'.join(parsed_response)
    print("生成的名字列表:", response_str)


#数据预处理
def processes_data(data):
    text = " ".join([doc.page_content for doc in data])
    # 将大文本分成较小的块，每块的长度由模型的上下文长度决定

    chunks = [text[i:i+500] for i in range (0,len(text),500)]
    return chunks
#提示词组合知识库
def concat_prompt(knowledge_lib,prompt):
    all_chunks = "\n\n".join(knowledge_lib)
    #将prompt与all_chunks合并
    final_prompt = f"{prompt}\n\n{all_chunks}"

    return final_prompt



if __name__ == "__main__":
    create_name()
