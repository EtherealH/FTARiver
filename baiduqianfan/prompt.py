from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
from langchain.prompts import StringPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import load_prompt

import inspect
class PromptTempalte:
    def __init__(self,prom_key):
        self.prom_key = prom_key

    def comm_prompt(self,name,country,sex):
        prompt = PromptTemplate.from_template("你是一个{name},帮我起一个具有{country}特色的{sex}名字")
        prompt.format(name=name,country=country,sex=sex)
        return prompt

    def chat_prompt(self, name, user_input):
        # Correct spelling of 'chat_template' and 'robot'
        template_str = (
            "system: 你是一名算命大师，你的名字叫{name}\n"
            "human: 你好,请问你都会什么\n"
            "robot: 你好先生，我精通唐诗宋词，周易占卜\n"
            "human: 你叫什么名字\n"
            "robot: 你好，我叫{name}\n"
            "human: {user_input}"
        )
        chat_template = ChatPromptTemplate.from_template(template_str)
        # Correct 'user_inout' to 'user_input'
        formatted_chat = chat_template.format(name=name, user_input=user_input)
        print(formatted_chat)
        return formatted_chat
    def direct_prompt(self):

        #直接创建消息
        sy = SystemMessage(
            content= "你是一个算命大师",
            addition_kwargs={"大师姓名","陈瞎子"}
        )
        hu = HumanMessage(
            content="请问大师都会什么"
        )
        robat = AIMessage (
            content= "我精通五行八卦，周易占卜"
        )
        tempalte = [sy,hu,robat]
        return  tempalte

    def level_prompt(self,full_pompt):

        #第一层设计(特征)
        Charater_template = """
        你是{persion},你有着{char}.
        """
        Charater_prompt = PromptTemplate.from_template(Charater_template)

        #第二层设计(行为)
        behavior_template = """你要遵循以下行为:{behavior-list}"""
        behavior_prompt = PromptTemplate.from_template(behavior_template)
        #第三层设计(禁止)
        prohibit_template = """你不许有以下行为:{prohibit_list}"""
        prohibit_prompt = PromptTemplate.from_template(prohibit_template)
        #将三层结合起来
        input_prompts = [
            ("Charater",Charater_prompt),
            ("behavior",behavior_prompt),
            ("prohibit",prohibit_prompt)
        ]
        pipeline_prompt = PipelinePromptTemplate(final_prompt=full_pompt,pipeline_prompts=input_prompts)

        return  pipeline_prompt

    #序列化通过文件管理提示词模板
    def load_prompt(self,name,what):
        #加载yaml提示词文件模板
        template = load_prompt("yamls/prompt.yaml")

        #格式化参数
        formatted_prompt = template.format(name=name,what=what)
        print(formatted_prompt)
        return formatted_prompt
    def process_input(self):
        #返回通用模板函数
        if self.prom_key == "common":
            return self.comm_prompt
        #返回对话模板函数
        elif self.prom_key =="chat":
            return self.chat_prompt
        #返回直接对话模板
        elif self.prom_key == "direct":
            return self.direct_prompt
        #返回自定义模板对象
        elif self.prom_key == "custom":
            custom_prompt = CustmPrompt(input_variables=["function_name"])
            return custom_prompt
        elif self.prom_key == "level":
            return self.level_prompt
        elif self.prom_key == "serialize":
            return self.load_prompt
        else:
            return None

#自定义模板
PROMPT = """
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称、源代码、中文解释。
函数名称: {function_name}
源代码:
{source_code}
代码解释:
"""
def get_source_code(function_name):
    #获取源代码
    return inspect.getsource(function_name)

class CustmPrompt(StringPromptTemplate):
    def format(self,**kwargs) -> str:
        #获取源代码
        source_code = get_source_code(kwargs["function_name"])
        #生成提示词模板
        prompt = PROMPT.format(
            function_name = kwargs["function_name"].__name__,
            source_code=source_code
        )
        return prompt