from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMMathChain
from langchain.agents import initialize_agent, AgentType
import os
from langchain.agents import Tool,load_tools
from langchain.memory import ConversationBufferMemory
from langchain.prompts import  MessagesPlaceholder
# serppai的token
os.environ["SERPAPI_API_KEY"] = "95ac0e518f8e578cc81b149144efd7535d5d7ccab87244e946a1cf3bb14ef3e7"
class AgentsTemplate:

    def __init__(self,**kwargs):
        #构建一个搜索工具
        search = SerpAPIWrapper()
        self.prompt = kwargs.get("base_prompt")
        self.llm = kwargs.get("llm")
        llm_math_chain = load_tools(["serpapi", "llm-math"], llm=self.llm)
        self.tools = [
            Tool(
                name="Search",
                func=search.run,
                description= "useful for when you need to answer questions about current events or the current state of the world"
            ),
            Tool(
                name="Math Chain",
                func=llm_math_chain[1].run,
                description="useful for solving mathematical problems"
            )
        ]
        #load_tools(["serpapi", "llm-math"], llm=self.llm)
        # 记忆组件
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        self.agentType = [AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                          AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                          AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                          AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                          AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION]

    #零样本增强式生成ZERO_SHOT_REACT_DESCRIPTION,
    #使用chatModel的零样本增强式生成CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    def zero_agent(self,question,agentType):
        if agentType not in self.agentType:
            raise ValueError("无效的 AgentType，请选择有效的类型！")
        # 动态构建初始化参数
        agent_params = {
            "tools": self.tools,
            "llm": self.llm,
            "agent": agentType,
            "verbose": True,
            "memory": self.memory,
            "agent_kwargs" : {"extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history"),MessagesPlaceholder(variable_name="agent_scratchpad")],
                           },
            "handle_parsing_errors": True
        }
        #初始化代理
        agent = initialize_agent(**agent_params)
        print("-------------------")
        # 输出提示词模板
        prompt = agent.agent.llm_chain.prompt
        print("Prompt Template:")
        print(prompt)
        # print(agent.agent.prompt.messages)
        # print(agent.agent.prompt.messages[0])
        # print(agent.agent.prompt.messages[1])
        # print(agent.agent.prompt.messages[2])
        try:
            response = agent.run(question)
            print(f"运行的代理类型: {agentType}, 提问内容: {question}")
            print(f"agent回答: {response}")
            #self.memory.save_context(question,response)
        except Exception as e:
            print(f"代理运行时出错: {e}")
    #使用chatModel的零样本增强式生成
