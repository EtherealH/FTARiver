from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
#示例选择器
#根据长度要求智能选择示例
class Selector_template:
    def __init__(self,select_key):
        self.select_key = select_key
    def base_length_choose_prompt(self,example_group,prompt):
        #example_group表示示例组，prompt表示提示词模板
        example_selector = LengthBasedExampleSelector(
            #传入提示词示例组
            examples = example_group,
            #传入提示词模板
            example_prompt = prompt,
            max_length = 25,
        )
        #使用小样本提示词模板来实现动态示例调用
        dynamic_group = FewShotPromptTemplate(
            example_selector = example_selector,
            example_prompt=prompt,
            prefix = "请根据给出的句诗取一个男孩和女孩名",
            suffix = "诗句:{adjective}\n男孩名:\n女孩名:",
            input_variables= ["adjective"]
        )
        return  dynamic_group
    #根据输入相似度选择示例（最大边际相关性）
    def base_mmr_prompt(self,example_group,prompt,model_name='all-MiniLM-L6-v2',k=2):
        #使用SentenceTransformer 加载模型
        #自定义嵌入搜索
        embeddings = CustomEmbeddings(model_name= model_name)
        # mmr
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            #传入示例组
            example_group,
            # 使用自定义嵌入搜索
            embeddings,
            # 设置使用的向量数据库
            FAISS,
            k=k,
        )
        # 使用小样本提示词模板来实现动态示例调用
        mmr_prompt = FewShotPromptTemplate(
            example_selector = example_selector,
            example_prompt= prompt,
            prefix="请根据给出的句诗取一个男孩和女孩名",
            suffix="诗句:{adjective}\n男孩名:\n女孩名:",
            input_variables=["adjective"]
        )
        return mmr_prompt
    # 根据输入相似度选择示例（最大余弦相似度）
    def base_chroma_prompt(self,example_group,prompt,model_name='all-MiniLM-L6-v2',k=2):
        embeddings = CustomEmbeddings(model_name=model_name)
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            example_group,
            embeddings,
            Chroma,
            k=k,
        )
        #小样本提示词模板
        similar_prompt = FewShotPromptTemplate(
            example_selector= example_selector,
            example_prompt = prompt,
            prefix="请根据给出的句诗取一个男孩和女孩名",
            suffix="诗句:{adjective}\n男孩名:\n女孩名:",
            input_variables=["adjective"]
        )
        return similar_prompt

    def process_input(self):
        # 返回通用模板函数
        if self.select_key == "base_len":
            return self.base_length_choose_prompt
        # 返回对话模板函数
        elif self.select_key == "base_mmr":
            return self.base_mmr_prompt
        # 返回直接对话模板
        elif self.select_key == "base_similarity":
            return self.base_chroma_prompt
        else:
            return None

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)

    def embed(self, texts):
        # 将文本转换为嵌入向量
        return self.embedding_model.encode(texts, convert_to_tensor=True).tolist()

    def embed_documents(self, documents):
        # 实现 embed_documents 方法
        return self.embed(documents)

    def embed_query(self, query):
        # 实现 embed_query 方法
        return self.embed([query])[0]