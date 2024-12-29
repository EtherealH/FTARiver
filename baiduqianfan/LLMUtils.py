from langchain_core.runnables import Runnable
import requests
from typing import List,Dict,Any,Optional
import json
class LocalLLM(Runnable):
    def __init__(self, api_url="http://localhost:11343/api/chat", model="llama3", temperature=0.7):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature

    def invoke(self, input: str, *args, **kwargs) -> str:
        import requests, json
        # 如果 input 不是字符串，尝试转换
        if not isinstance(input, str):
            try:
                input = str(input)
            except Exception as e:
                raise TypeError(f"input 参数必须是字符串，但得到了 {type(input)}: {e}")

        headers = {"Content-Type": "application/json"}
        max_tokens = kwargs.get("max_tokens", 100)

        data = {
            "model": self.model,
            "options": {"temperature": self.temperature, "max_tokens": max_tokens},
            "stream": False,
            "messages": [{"role": "user", "content": input}],
        }

        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            result = response.json()
            message_content = result.get("message", {}).get("content", "")
            if message_content:
                # 可以加入对 message_content 的后处理，例如截取或清理多余的信息
                return message_content.strip()
            else:
                return "未返回有效的内容"
        else:
            raise Exception(f"请求模型服务器失败: {response.status_code} - {response.text}")

    def __call__(self, input: str, *args, **kwargs):
        return self.invoke(input, *args, **kwargs)

    def generate_prompt(self, prompts: List[str], **kwargs) -> dict[str, list[list[dict[str, str]]]]:
        """生成的 prompt 返回兼容格式"""
        generations = []
        for prompt in prompts:
            # 使用 invoke 生成每个 prompt 的响应
            response_text = self.invoke(prompt, **kwargs)
            generations.append([{"text": response_text}])

        return {"generations": generations}