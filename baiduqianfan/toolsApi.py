import requests
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import json

from langchain.schema import Document


from FTARiver.baiduqianfan.prompt import PromptTempalte
from FTARiver.baiduqianfan.selector import Selector_template
from FTARiver.baiduqianfan.loader import LoaderTemplate
from FTARiver.baiduqianfan.longcontext import deal_context_template
from FTARiver.baiduqianfan.test.statictextLoder import StaticLoader
from FTARiver.baiduqianfan.embedDocment import chat_doc_template
from FTARiver.baiduqianfan.Chains import Chains_template
from FTARiver.baiduqianfan.Chains import Special_Chains_template
from FTARiver.baiduqianfan.LLMUtils import LocalLLM
from FTARiver.baiduqianfan.agents import AgentsTemplate
from langchain.agents import AgentType
# 经济学文档
doc = "knowledge/plain_text_economic.docx"
chat_yl = chat_doc_template(doc)


def get_baidu_api_url():
    # api_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_8b?access_token=" + get_access_token()
    api_url = "http://localhost:11343/api/chat"
    return api_url


class SeparatedListOutput(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")


# 自定义大模型连接方式
def llm(msgs):
    api_url = get_baidu_api_url()
    headers = {
        'Content-Type': 'application/json'
    }

    # 构建可序列化的消息列表
    messages = [{"role": msg.type, "content": msg.content} for msg in msgs]

    data = json.dumps({
        "model": "llama3",
        "options": {
            "temperature": 0.  # 控制生成的内容
        },
        "stream": False,
        "messages": messages  # 使用构建的消息列表
    })

    response = requests.request("POST", api_url, headers=headers, data=data)

    return response.text


# 函数示例
def add_sum(x, y):
    z = x + y
    print(z)
    return z


def create_name():
    # 实例化提示词模板类
    prompt_instance = PromptTempalte("chat")
    # 获取模板类对象
    prompt_object = prompt_instance.process_input()
    # 获取模板
    example_prompt = "请根据古诗词再帮我起一个男孩名和一个女孩名"
    prompt = prompt_object("周公", example_prompt)
    # 示例组
    example_group = [
        {"input": "人生到处知何似，应似飞鸿踏雪泥", "output1": "何应", "output2": "何雪"},
        {"input": "倚杖柴门外，临风听慕蝉", "output1": "临风", "output2": "慕蝉"},
        {"input": "竹喧归浣女，莲动下渔舟", "output1": "宣舟", "output2": "宣瑜"}
    ]
    # 最终提示词模板
    complex_prompt = PromptTemplate(
        input_variables=["input", "output1", "output2"],
        template="参考示例:诗句:{input}\n男孩名:{output1}\n女孩名:{output2}"
    )
    # 使用选择提示组
    selectors = Selector_template("base_similarity")
    selector_object = selectors.process_input()
    example_selector = selector_object(example_group, complex_prompt)
    final_prompt = example_selector.format(adjective=prompt)
    get_baidu_api_url()
    # load加载知识库
    load_template = LoaderTemplate("md")
    load_object = load_template.process_input()
    data = load_object("knowledge/create_name.md")
    chunks = processes_data(data)
    # 拼接组合词
    prompt = concat_prompt(chunks, final_prompt)
    # 返回结果
    response = llm(prompt)
    output_parser = SeparatedListOutput()
    parsed_response = output_parser.parse(response)
    response_str = '\n'.join(parsed_response)
    print("生成的名字列表:", response_str)




def extract_page_content(docs):
    if isinstance(docs, list):
        # 将每个文档的 page_content 提取后，拼接为一个字符串
        return "\n".join([doc.page_content for doc in docs])
    elif isinstance(docs, Document):
        # 单一 Document 对象，直接返回其 page_content
        return docs.page_content
    else:
        # 抛出类型错误，确保输入符合预期
        raise TypeError("输入的内容不是 Document 或其列表")


# 函数式封装
def llm_runnable(prompt):
    llm = LocalLLM()
    return llm.invoke(prompt)


def economic_plugin():
    text = StaticLoader.get_text()
    reo_docs = deal_context_template.start(text)

    # 调用方法
    prompt = extract_page_content(reo_docs)
    response = llm(prompt)
    output_parser = SeparatedListOutput()
    parsed_response = output_parser.parse(response)
    response_str = '\n'.join(parsed_response)
    print(response_str)


# 和文本机器人聊天
def chatWithyl(question):
    _content = ""
    context = chat_yl.askAndFindFiles(question)
    for i in context:
        _content += i.page_content
    msgs = chat_yl.prompts.format_messages(context=_content, question=question)
    print(msgs)
    response = llm(msgs)
    print(f"answer: {response}")


# 数据预处理
def processes_data(data):
    text = " ".join([doc.page_content for doc in data])
    # 将大文本分成较小的块，每块的长度由模型的上下文长度决定

    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    return chunks


# 提示词组合知识库
def concat_prompt(knowledge_lib, prompt):
    all_chunks = "\n\n".join(knowledge_lib)
    # 将prompt与all_chunks合并
    final_prompt = f"{prompt}\n\n{all_chunks}"

    return final_prompt

# 运行链的测试
def use_chain_llm():


    #经济学模版
    economic_template = """
    You are an economist.\n
    You are good at answering questions in a simple and understandable way.\n
    When you don't know the problem, you admit that you don't.\n
    this is a question at down.\n
    {input}
    """
    economic_prompt = PromptTemplate.from_template(economic_template)
    # 算命学模版
    featureTail_template = """
    You are an Fortune Teller.\n
    You can give people names based on their date of birth and gender.\n
    this is a question at down.\n
    {input}
    """
    fortuneTaller_prompt = PromptTemplate.from_template(featureTail_template)
    #模型
    llm = LocalLLM(api_url="http://localhost:11343/api/chat", model="llama3")
    #提示词模版列表
    prompt_list = [economic_template,featureTail_template]
    #问题
    question = "How to define the stock market in economics?"
    #创建链对象
    chain = Chains_template(llm,prompt_list,question)
    #调用函数
    #chain.route_chain()
    answer = chain.douc_process_chain("What is Economics?")
    print(answer)
# 专用链测试
def use_special_chains():
    llm = LocalLLM(api_url="http://localhost:11343/api/chat", model="llama3")

    chains = Special_Chains_template(
        prompt = PromptTemplate(
            template = "please write an article about {topic}",
            input_variables = ["topic"]
        ),
        llm = llm
    )
    result = chains.invoke({"topic":"economic"})
    print(result)

# langchain内置agent测试
def use_agent():
    llm = LocalLLM(api_url="http://localhost:11343/api/chat", model="llama3")
    agent = AgentsTemplate(llm=llm)
    question = input("请输入您的问题")
    agent.zero_agent(question,AgentType.ZERO_SHOT_REACT_DESCRIPTION)



if __name__ == "__main__":
    # create_name()
    # economic_plugin()
    #chat_yl.splitSentences()
    #chatWithyl("What is Economics?")
    #use_chain_llm()
    #use_special_chains()
    use_agent()

