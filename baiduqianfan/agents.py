from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
import os
from langchain.agents import Tool,load_tools
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
#创建toolkits
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit,SQLDatabaseToolkit
from langchain.prompts import PromptTemplate,MessagesPlaceholder
# serppai的token
os.environ["SERPAPI_API_KEY"] = "95ac0e518f8e578cc81b149144efd7535d5d7ccab87244e946a1cf3bb14ef3e7"
class AgentsTemplate:

    def __init__(self,**kwargs):
        #构建一个搜索工具
        search = SerpAPIWrapper()
        self.prompt = kwargs.get("base_prompt")
        self.llm = kwargs.get("llm")
        llm_math_chain = load_tools(["serpapi", "llm-math"], llm=self.llm)
        # 创建一条链总结对话
        template = """
        The following is a conversation between an AI robot and a human:{chat_history}
        Write a conversation summary based on the input and the conversation record above,input:{input}
        """

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        prompt = PromptTemplate(
            input_variable=["input", "chat_history"],
            template=template
        )
        self.shared_memory = ReadOnlySharedMemory(memory=self.memory)
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt = prompt,
            verbose = True,
            memory = self.shared_memory
        )
        self.tools = [
            Tool(
                name="Search",
                func=search.run,
                description= "useful for when you need to answer questions about current events or the current state of the world"
            ),
            Tool(
                name="Summary",
                func=self.SummaryChainFun,
                description="This tool can be used when you are asked to summarize a conversation. The tool input must be a string. Use it only when necessary"
            )
        ]
        #load_tools(["serpapi", "llm-math"], llm=self.llm)
        # 记忆组件


        self.agentType = [AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                          AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                          AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                          AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                          AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION]
    def SummaryChainFun(self, history):
        print("\n============== Summary Chain Execution ==============")
        print("Input History: ", history)
        return self.summary_chain.run(history)

    def createToolkits(self,toolKey):
        toolkit = None
        if toolKey == "azure":
            toolkit = AzureCognitiveServicesToolkit()
        elif toolKey == "sqlData":
            db = SQLDatabase.from_uri("sqlite:///Chinook.db")
            toolkit = SQLDatabaseToolkit(db = db,llm = self.llm)
        return toolkit

    #零样本增强式生成ZERO_SHOT_REACT_DESCRIPTION,
    #使用chatModel的零样本增强式生成CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    def zero_agent(self,question,agentType):
        if agentType not in self.agentType:
            raise ValueError("无效的 AgentType，请选择有效的类型！")
        prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
        suffix = """Begin!"
        {chat_history}
        Question: {input}
        {agent_scratchpad}"""
        #tookits使用
        tookits = self.createToolkits("azure")
        # 动态构建初始化参数
        agent_params = {
            #"tools": self.tools,
            "toolkit": tookits,
            "llm": self.llm,
            "agent": agentType,
            "verbose": True,
            "memory": self.memory,
            "agent_kwargs" : {
                "chat_history": MessagesPlaceholder(variable_name="chat_history"),
                "agent_scratchpad":MessagesPlaceholder(variable_name="agent_scratchpad"),
                "prefix":prefix,
                "sufix":suffix,
                "input":MessagesPlaceholder("input")

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
