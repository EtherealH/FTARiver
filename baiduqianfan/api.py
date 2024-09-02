import requests
import json
def get_access_token():
    url =  "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=q9IGmb6ZjDxbQjyGvXtwRBYn&client_secret=35EQhSIKW5nS1f02VVGsk0LZKGGFdt6r"
    payload = json.dumps("")
    headers = {
        'Content-Type':'application/json',
        'Accept':'application/json'
    }
    response = requests.request("POST",url,headers=headers,data=payload)

    return response.json().get("access_token")
def get_baidu_api_url():
    api_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_8b?access_token=" + get_access_token()
    return api_url

if __name__ == "__main__":
    api_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_8b?access_token=" + get_access_token()
    payload = json.dumps({
        "messages":[
            {
                "role":"user",
                "content":"how are you"
            }
        ]
    })
    headers = {
        'Content-Type':'application/json'
    }
    response = requests.request("POST",api_url,headers=headers,data=payload)
    print(response.text)
