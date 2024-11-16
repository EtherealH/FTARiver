
from langchain.prompts import PromptTemplate

from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain_core.prompt_values import StringPromptValue

from langchain_core.runnables.base import RunnableLambda,RunnableMap

from langchain.prompts.base import BasePromptTemplate

from FTARiver.baiduqianfan.LLMUtils import LocalLLM
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from typing import List,Dict,Any,Optional
from langchain.chains.base import Chain
from FTARiver.baiduqianfan.embedDocment import LocalEmbedding
from langchain.callbacks.manager import (
CallbackManagerForChainRun
)

class Chains_template:
    def __init__(self, llm, prompt_template_list,question):
        self.llm = llm
        self.prompt = prompt_template_list
        self.question = question

    # 顺序链
    def seq_chain(self):
        # 链一任务：翻译成中文
        first_chain = RunnableLambda(
            lambda content: self.llm.invoke(str(self.prompt[0].invoke(content))),
            afunc=lambda content: self.llm.ainvoke(str(self.prompt[0].invoke(content)))  # 如果需要异步支持
        )

        # 链二任务：对翻译的中文进行总结
        second_chain = RunnableLambda(
            lambda chinese_review: self.llm.invoke(str(self.prompt[1].invoke(chinese_review))),
            afunc=lambda chinese_review: self.llm.ainvoke(str(self.prompt[1].invoke(chinese_review)))
        )

        # 链三任务：智能识别语言
        third_chain = RunnableLambda(
            lambda summary: self.llm.invoke(str(self.prompt[2].invoke(summary))),
            afunc=lambda summary: self.llm.ainvoke(str(self.prompt[2].invoke(summary)))
        )

        # 链四任务：针对摘要使用语言进行评论
        four_chain = RunnableLambda(
            lambda language: self.llm.invoke(str(self.prompt[3].invoke(language))),
            afunc=lambda language: self.llm.ainvoke(str(self.prompt[3].invoke(language)))
        )

        # 使用 RunnableMap 来串联所有任务
        overall_chain = RunnableMap(
            {
                "Chinese_Rview": first_chain,
                "Chinese_Summary": second_chain,
                "Language": third_chain,
                "Reply": four_chain
            }
        )

        # 读取文件
        content = self.question

        # 依次调用各个链
        ans = overall_chain.invoke(content)
        print(f"chains answer: {ans}")

        return ans

        # 路由链
    def route_chain(self):
        # 模版格式化
        prompt_infos = [
            {
                "name": "economy",
                "description": "Good at answering economic questions",
                "prompt_template": self.prompt[0],
            },
            {
                "name": "zhouyi",
                "description": "Good at fortune telling",
                "prompt_template": self.prompt[1],
            },
        ]

        # 创建提示链
        description_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])

            # 使用 RunnableLambda 创建每个提示链
            chain = RunnableLambda(lambda input_value: self.llm.invoke(str(prompt.invoke(input_value))))
            description_chains[name] = chain

        # 使用 Router 模板生成路由提示
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(template=router_template, input_variables=["input"])

        # 创建路由链
        router_chain = RunnableLambda(lambda input_value: self.llm.invoke(str(router_prompt.invoke(input_value))))

        # 创建默认链
        default_chain = RunnableLambda(lambda input_value: self.llm.invoke(str(input_value)))  # 假设默认链是直接调用 LLM

        # 创建 MultiPromptChain
        chains = RunnableLambda(
            lambda input_value: (
                description_chains.get(self.route(input_value), router_chain).invoke(input_value)
            ),
            afunc=lambda input_value: (
                description_chains.get(self.route(input_value), default_chain).invoke(input_value)
            )
        )

        # 运行链
        ans = chains.invoke(self.question)
        print(f"chains answer: {ans}")

    def route(self, input_value):
        # 示例路由逻辑
        if "economics" in input_value:
            return "economy"
        elif "zhouyi" in input_value:
            return "zhouyi"
        else:
            return None  # 或者返回默认链的名称

    # 文档处理链
    def douc_process_chain(self,question):
        #加载文档
        global summary
        loader = PyPDFLoader("knowledge/economicist2.pdf")
        doucment = loader.load()
        #分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        texts = text_splitter.split_documents(doucment)

        #嵌入生成与向量存储
        embeddings = LocalEmbedding('sentence-transformers/all-MiniLM-L6-v2')
        vertorstore = Chroma.from_documents(texts, embeddings)
        #问答链
        retriever = vertorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever, return_source_documents=True)
        #提问
        query = "Please summarize the contents of this pdf document"

        result = qa_chain({"query":query})
        #摘要链
        summary_prompt_template = """
        You are a professional document analysis assistant. Please summarize the following text:
        {text}
        summary:
        """
        summary_prompt = PromptTemplate(template=summary_prompt_template,input_variable=["text"])
        summary_chain = LLMChain(llm = self.llm,prompt=summary_prompt)
        # 遍历文档分块进行摘要
        summaries = []
        for text in texts:
            prompt_value = summary_prompt.format(text=text.page_content)
            # 如果 prompt_value 是 StringPromptValue，则转换为字符串
            if isinstance(prompt_value, StringPromptValue):
                prompt_text = prompt_value.to_string()
            else:
                prompt_text = prompt_value
            summary = summary_chain.run({"text": prompt_text})
            summaries.append(summary)
        final_summary = "\n".join(summaries)
        #基于生成的摘要回答问题
        answer_prompt_template = f"""
        You have the following summarized information about the document:{summary}
        
        this about summary content can help you answer this question, answer the following question:{question}
        Answer:        
        """
        answer_prompt = PromptTemplate(template=answer_prompt_template,input_variable=["summary","question"])
        answer_chain = LLMChain(llm=self.llm,prompt = answer_prompt)
        final_answer = answer_chain.run({"summary":final_summary,"question":question})
        return final_answer


class Special_Chains_template(Chain):
    """开发一个文章生成器"""
    prompt: BasePromptTemplate
    llm: LocalLLM
    out_key: str = "economic"

    @property
    def input_keys(self) -> List[str]:
        """返回 prompt 所需要的所有 key"""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """将始终返回 economic 键"""
        return [self.out_key]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """运行链"""
        # 使用 prompt 格式化输入内容
        prompt_value = self.prompt.format(**inputs)

        # 调用 LocalLLM 的 invoke 方法来生成响应文本
        response_text = self.llm.invoke(prompt_value)

        # 如果有 run_manager，记录事件
        if run_manager:
            run_manager.on_text("YAli article is written")

        # 返回生成的文本作为字典，键为 out_key
        return {self.out_key: response_text}

    @property
    def _chain_type(self) -> str:
        """链类型"""
        return "YAli_article_chain"