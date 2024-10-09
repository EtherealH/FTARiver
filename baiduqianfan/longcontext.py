from langchain_community.document_transformers import LongContextReorder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


from langchain_community.vectorstores import Chroma

class deal_context_template:

    def start(text):
        # 使用huggingface托管的开源LLM来做嵌入，MiniLM-L6-v2是一个较小的LLM
        embedings = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

        retrieval = Chroma.from_texts(text, embedings).as_retriever(
            search_kwargs={"k": 10}
        )
        query = "经济学是什么"

        # 根据相关性返回文本块
        docs = retrieval.get_relevant_documents(query)

        # 对检索的结果进行重新排序
        # 问题相关性越低的内容块放中间
        # 问题相关性越高的内容块放在头尾
        recordering = LongContextReorder()
        reo_docs = recordering.transform_documents(docs)

        return reo_docs
