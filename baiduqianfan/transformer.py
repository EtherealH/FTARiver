from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from doctran import Doctran


class TransformerTemplate:
    def __init__(self, **kwargs):
        self.file_path = kwargs.get("file")
        self.str_data = kwargs.get("str_data")
        self.code_doc = kwargs.get("code_doc")
        self.token_data = kwargs.get("token_data")

    # 文档分割方法
    def doc_segment(self, file):
        # 加载要分割的文档
        with open(file) as f:
            fragment = f.read()

        # 初始化分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,  # 切割文本块大小，一般通过长度函数计算
            chunk_overlap=20,  # 切分的文本块重叠大小，一般通过长度函数计算
            length_function=len,  # 长度函数，也可以传递tokensize函数
            add_start_index=True,  # 是否添加起始索引
        )
        text = text_splitter.create_documents([fragment])

        return text

    # 字符串分割
    def str_segment(self, str_data):
        # 加载要分割的文档
        with open(str_data) as f:
            fragment = f.read()
        # 初始化分割器
        text_splitter = CharacterTextSplitter(
            chunk_size=50,  # 切割文本块大小，一般通过长度函数计算
            chunk_overlap=20,  # 切分的文本块重叠大小，一般通过长度函数计算
            length_function=len,  # 长度函数，也可以传递tokensize函数
            add_start_index=True,  # 是否添加起始索引
            is_separator_regex=False,  # 是否使用正则表达式
        )
        text = text_splitter.create_documents([fragment])
        return text

    # 代码文档分割
    def code_segment(self, code_file):

        py_spliter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=50,
            chunk_overlap=10,
        )
        python_doc = py_spliter.create_documents([code_file])
        return python_doc

    # 按照token来分割文档

    def token_segment(self, token_data):
        # 要分割的文档
        with open(token_data) as f:
            frgment = f.read()
        text_spliter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="gpt-3.5-turbo", #指定 tiktoken 编码器
            chunk_size=4000,  # 切割文本块大小，一般通过长度函数计算
            chunk_overlap=30,  # 切分的文本块重叠大小，一般通过长度函数计算
        )
        text = text_spliter.create_documents([frgment])

        return text

    # 文档总结翻译
    def summ_trans_doc(self, file):
        doctrans = Doctran(
            openai_api_key="http://localhost:11343/api/chat",
            openai_model="llama3",
            openai_token_limit=8000
        )
        docments = doctrans.parse(content=file)
        # 总结文档&翻译&精练文档
        translated_summary = docments.summarize(token_limit=800).translate(language="chinese").refine(
            topics=["test1", "test2"]).execute()
        return translated_summary

    def process_input(self):

        if self.code_doc is not None:
            return self.code_segment
        elif self.str_data is not None:
            return self.str_segment
        elif self.token_data is not None:
            return self.token_segment
        elif self.file_path is not None:
            return self.doc_segment
