from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
class LoaderTemplate:
    def __init__(self,loader_key):
        self.loader_key = loader_key
    #使用loader加载markdown文档
    def testLoader(self,file):
        loader = TextLoader(file,encoding='utf-8')
        data = loader.load()
        return data

    #使用loader加载csv文件
    def csvLoader(self,file):
        loader = CSVLoader(file_path=file,source_column="name")
        data = loader.load()
        return data

    #某个目录下，有excel文件，需要将目录下所有xlxs文件加载进来
    def directoryLoader(self,file,suffix):
        loader = DirectoryLoader(path=file,glob=suffix)
        data = loader.load()
        return data

    #加载json文件
    def jsonLoader(self,file):

        loader = JSONLoader(file_path=file,jq_schema=".template",text_content=True)
        data = loader.load()
        return data

    # 加载pdf文件
    def pdfLoader(self,file):
        loader = PyPDFLoader(file)
        data = loader.load_and_split()
        return data

    #根据key返回对应的函数
    def process_input(self):

        if self.loader_key == "md":
            return self.testLoader

        elif self.loader_key == "csv":
            return self.csvLoader

        elif self.loader_key == "directory":
            return self.directoryLoader

        elif self.loader_key == "json":
            return self.jsonLoader

        elif self.loader_key == "pdf":
            return self.pdfLoader


