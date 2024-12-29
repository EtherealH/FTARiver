from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
import os
from langchain.memory import ConversationBufferMemory

# serppai的token
os.environ["SERPAPI_API_KEY"] = "95ac0e518f8e578cc81b149144efd7535d5d7ccab87244e946a1cf3bb14ef3e7"
class AgentsTemplate:

    def __init__(self,**kwargs):
        self.prompt = kwargs.get("base_prompt")
        self.llm = kwargs.get("llm")
        self.tools = load_tools(["serpapi","llm-math"],llm=self.llm)
        # 记忆组件
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
        )
        self.agentType = [AgentType.ZERO_SHOT_REACT_DESCRIPTION,AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION]

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
        }
        if agentType == AgentType.CONVERSATIONAL_REACT_DESCRIPTION:
            agent_params["memory"] = self.memory
        #初始化代理
        agent = initialize_agent(**agent_params)
        print("-------------------")
        try:
            response = agent.run(question)
            print(f"agent回答: {response}")
        except Exception as e:
            print(f"代理运行时出错: {e}")
    #使用chatModel的零样本增强式生成
