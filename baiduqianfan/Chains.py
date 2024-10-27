
from langchain.prompts import PromptTemplate

from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

from langchain_core.runnables.base import RunnableLambda,RunnableMap


class Chains_template:
    def __init__(self, llm, prompt_template_list,question):
        self.llm = llm
        self.prompt = prompt_template_list
        self.question = question

    # 顺序链
    def seq_chain(self):
        # 链一任务：翻译成中文
        first_chain = RunnableLambda(
            lambda content: self.llm.invoke(self.prompt[0].invoke(content)),
            afunc=lambda content: self.llm.ainvoke(self.prompt[0].invoke(content))  # 如果需要异步支持
        )

        # 链二任务：对翻译的中文进行总结
        second_chain = RunnableLambda(
            lambda chinese_review: self.llm.invoke(self.prompt[1].invoke(chinese_review)),
            afunc=lambda chinese_review: self.llm.ainvoke(self.prompt[1].invoke(chinese_review))
        )

        # 链三任务：智能识别语言
        third_chain = RunnableLambda(
            lambda summary: self.llm.invoke(self.prompt[2].invoke(summary)),
            afunc=lambda summary: self.llm.ainvoke(self.prompt[2].invoke(summary))
        )

        # 链四任务：针对摘要使用语言进行评论
        four_chain = RunnableLambda(
            lambda language: self.llm.invoke(self.prompt[3].invoke(language)),
            afunc=lambda language: self.llm.ainvoke(self.prompt[3].invoke(language))
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

