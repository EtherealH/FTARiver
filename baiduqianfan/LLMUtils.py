from langchain_core.runnables import Runnable
import requests
from typing import List,Dict,Any,Optional
import json
class LocalLLM(Runnable):
    def __init__(self, api_url="http://localhost:11343/api/chat", model="llama3", temperature=0.7):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature

    def invoke(self, input: str, **kwargs) -> str:
        headers = {
            "Content-Type": "application/json"
        }

        # 从 kwargs 中获取 max_tokens，默认 100
        max_tokens = kwargs.get("max_tokens", 100)

        data = {
            "model": self.model,
            "options": {
                "temperature": self.temperature,
                "max_tokens": max_tokens
            },
            "stream": False,
            "messages": [{
                "role": "user",
                "content": input  # 用户输入的 prompt
            }]
        }

        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))

        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            message = result.get("message","")
            content = message.get("content")
            return content
        else:
            raise Exception(f"请求模型服务器失败: {response.status_code} - {response.text}")

    def generate_prompt(self, prompts: List[str], **kwargs) -> dict[str, list[list[dict[str, str]]]]:
        """生成的 prompt 返回兼容格式"""
        generations = []
        for prompt in prompts:
            # 使用 invoke 生成每个 prompt 的响应
            response_text = self.invoke(prompt, **kwargs)
            generations.append([{"text": response_text}])

        return {"generations": generations}