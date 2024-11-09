from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer

class chat_doc_template:

    def __init__(self, doc):
        self.doc = doc
        self.splitText = []
        self.template = [
            ("system", "Here are the documents related to the issue: {context} \n"),
            ("user", "hello!"),
            ("ai", "hi"),
            ("user", "{question}"),
        ]
        self.prompts = ChatPromptTemplate.from_messages(self.template)

    # 向量化与向量存储
    def embeddingAndVectorDB(self):
        # 确保 self.splitText 不为空
        if not self.splitText or len(self.splitText) == 0:
            raise ValueError("Document list is empty. Please provide valid documents.")

        # 使用本地的预训练模型，如 'all-MiniLM-L6-v2'
        embedding_model = LocalEmbedding('sentence-transformers/all-MiniLM-L6-v2')
        page_texts = []
        for t in self.splitText:
            page_texts.append(t.page_content)
        print(f"page: {page_texts}")
        # 生成嵌入
        embeddings = embedding_model.embed_documents(page_texts)
        if not embeddings or len(embeddings) != len(page_texts):
            raise ValueError("Mismatch between document count and embedding count.")

        # 打印调试信息
        print(f"Documents: {self.splitText}")
        print(f"Embeddings: {embeddings}")

        # 使用 Chroma 存储这些向量
        db = Chroma.from_texts(
            texts=page_texts,  # 文档列表
            embedding=embedding_model  # 传入自定义的嵌入模型
        )

        return db

    def getFile(self):
        print(f"Attempting to load document: {self.doc}")
        doc = self.doc
        loaders = {
            "docx": Docx2txtLoader,
            "pdf": PyPDFLoader,
            "excel": UnstructuredExcelLoader
        }
        file_extension = doc.split(".")[-1]
        loader_class = loaders.get(file_extension)

        if loader_class:
            try:
                print(f"Using loader: {loader_class.__name__}")
                loader = loader_class(doc)
                text = loader.load()
                print("Loaded text:", text)

                # 处理返回的 text 是 list 但内容是 Document 类型的情况
                if isinstance(text, list):
                    processed_text = []
                    for t in text:
                        if isinstance(t, Document):
                            processed_text.append(t)
                        else:
                            print(f"Unknown object type in list: {type(t)}")
                    return processed_text
                elif isinstance(text, str):
                    return [text]
                else:
                    raise ValueError("Loaded content is neither string nor list of strings.")
            except Exception as e:
                print(f"Error loading {file_extension} file: {e}")
                print(f"Document path: {doc}")
                return None
        else:
            print(f"Unsupported file extension: {file_extension}")
            return None

    # 处理文档函数
    def splitSentences(self):
        text = self.getFile()  # 加载文档
        if text:  # 确保加载成功
            text_split = CharacterTextSplitter(
                chunk_size=150,
                chunk_overlap=20,
            )
            texts = text_split.split_documents(text)
            self.splitText = texts  # 将文本存储到 splitText
        else:
            raise ValueError("Failed to load document. Please check the file format.")

    # 提问并找到相关文本块（在向量存储时使用最大边际相似性和相似性打分）
    def askAndFindFiles(self, question):
        db = self.embeddingAndVectorDB()
        print(f"Vector DB created: {db}")  # 确保数据库创建成功
        retriever = db.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold": 0.1, "k": 1})
        context = retriever.invoke(input=question)
        print(f"Retrieved documents: {context}")  # 确保检索到内容
        return context


class LocalEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    # 实现 embed_documents 方法
    def embed_documents(self, texts):
        # 检查传入的 texts 是否为字符串列表
        # if not isinstance(texts, list) or not all(isinstance(text, Document) for text in texts):
        #     raise ValueError("Input must be a list of strings.")
        # 使用模型进行嵌入
        embeddings = []
        for text in texts:
            try:
                # 对文本进行嵌入
                embedding = self.model.encode(text)
                embeddings.append(embedding.tolist())
            except Exception as e:
                print(f"Error encoding text: {text}, Error: {e}")
                embeddings.append(None)  # 可以选择添加 None 或者处理错误
        return embeddings
    # 实现 embed_query 方法
    def embed_query(self, query):
        try:
            return self.model.encode(query).tolist()  # 转换为列表格式
        except Exception as e:
            print(f"Error encoding query: {query}, Error: {e}")
            return None  # 或者抛出异常
