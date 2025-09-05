"""
the official API usage of qwen3-30b-a3b-instruct-2507 model
"""

import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

system_prompt = """
You're a helpful research assistant, who answers questions based on provided research documents in a clear way and easy-to-understand way.
If there are no research documents, or the research documents are irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. 
If you're unable to answer the question, do not list sources.
"""
human_prompt = """
## Research:
Source Document: Research\\pku.pdf, Page 0:
PKU is the abbreviation of Peking University. The university's software engineering discipline is very strong and has high
academic influence worldwide.

Source Document: Research\\2305.14325.pdf, Page 24:
Round 2
✓ ✓
Figure 26: Example of MMLU Debate.
25

Source Document: Research\\ecnu.pdf, Page 0:
ECNU is a school in China, its total name is “East China Normal University”. And the
software engineering of this school is good.

## Question:
What does PKU stand for? How about its software engineering discipline?
"""
# 1. 用OpenAI接口
# try:
#     client = OpenAI(
#         # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
#         api_key=os.getenv("DASHSCOPE_API_KEY"),
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     )

#     completion = client.chat.completions.create(
#         model="qwen3-30b-a3b-instruct-2507",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#         messages=[
#             # {'role': 'system', 'content': 'You are a helpful assistant.'},
#             # {'role': 'user', 'content': 'What does PKU stand for?'}
#             {'role': 'system', 'content': system_prompt},
#             {'role': 'user', 'content': human_prompt}
#             ]
#     )
#     print(completion.choices[0].message.content)
# except Exception as e:
#     print(f"错误信息：{e}")
#     print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

# 2. 用LangChain
try:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    client = ChatOpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_key=SecretStr(api_key) if api_key else None,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3-30b-a3b-instruct-2507"
    )

    
    messages=[
        # {'role': 'system', 'content': 'You are a helpful assistant.'},
        # {'role': 'user', 'content': 'What does PKU stand for?'}
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': human_prompt}
        ]
    
    response = client.invoke(messages)
    # print(response.model_dump_json()["content"])
    print(response.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
# # print(os.getenv("DASHSCOPE_API_KEY"))