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
Source Document: E:\Work\LM_RAG_Projects\open-rag-bench\data\raw\pdf\2410.14077v2.pdf, Page 0:
significantly due to changes in the power network, such as
fluctuations in electrical loads, the addition or removal of
power sources, and environmental factors like temperature.
Consequently, real-time impedance estimation is crucial for
optimizing inverter performance and reliability. By continu-
ously adapting to dynamic grid conditions, it enables stable
power injection and ensures that control strategies remain
effective.
The main challenges in accurately estimating output line
impedance stem from several factors. (i) First, inverters
typically lack access to global measurements or network-
wide data, which makes it difficult to estimate the effective
grid voltage. (ii) Additionally, measured signals often lack
the necessary persistence of excitation, which is crucial for
accurate impedance estimation. (iii) Since inverters usually
operate at a steady state, only local output voltage and
current are measurable, while both line impedance and grid

Source Document: E:\Work\LM_RAG_Projects\open-rag-bench\data\raw\pdf\2410.14077v2.pdf, Page 0:
As inverter-based resources continue to expand, accurate
monitoring of these factors is essential for maintaining grid
stability.
Output impedance is commonly defined as the impedance
between the inverter and the grid, where the grid, in this
context, represents an abstraction of the remaining network.
1 Department of Mechanical Science and Engineering,
University of Illinois at Urbana-Champaign, 61801 IL, USA
ajaesang4@illinois.edu,baskaria2@illinois.edu,csalapaka@illinois.edu
(a)
 (b)
Fig. 1: Grid model: (a) Grid with a complex structure, and
(b) Thevenin equivalent model from the inverter, where R
and L represent the equivalent resistance and inductance of
the Thevenin equivalent model.
In a complex power system, we model the grid as perceived
by the inverter using Thevenin’s theorem as an equivalent
grid voltage source in series with the output line impedance
(see Fig. 1). This Thevenin-equivalent impedance can vary
significantly due to changes in the power network, such as

Source Document: E:\Work\LM_RAG_Projects\open-rag-bench\data\raw\pdf\2410.14077v2.pdf, Page 0:
Inverter Output Impedance Estimation in Power Networks: A Variable
Direction Forgetting Recursive-Least-Square Algorithm Based Approach
Jaesang Park1,a, Alireza Askarian 1,b, and Srinivasa Salapaka 1,c
Abstract— As inverter-based loads and energy sources be-
come increasingly prevalent, accurate estimation of line
impedance between inverters and the grid is essential for
optimizing performance and enhancing control strategies. This
paper presents a non-invasive method for estimating output-
line impedance using measurements local to the inverter. It
provides a specific method for signal conditioning of signals
measured at the inverter, which makes the measured data better
suited to estimation algorithms. An algorithm based on the Vari-
able Direction Forgetting Recursive Least Squares (VDF-RLS)
method is introduced, which leverages these conditioned signals
for precise impedance estimation. The signal conditioning
process transforms measurements into the direct-quadrature

Source Document: E:\Work\LM_RAG_Projects\open-rag-bench\data\raw\pdf\2410.14077v2.pdf, Page 1:
noise induced by inverter activity.
II. G RID MODELING AND PRECONDITIONING FOR
IMPEDANCE ESTIMATION
A. Thevenin-Based Grid Representation
The grid around an inverter is simplified using Thevenin’s
theorem as a single voltage source and output line
impedance. The stiff voltage source, unaffected by the in-
verter’s operation, is treated as the grid voltage, while the
impedance represents the output line impedance perceived
by the inverter. For simplicity, we approximate the Thevenin
impedance with a first-order model consisting of resistance
R and inductance L. The dynamics that describe this system
in Fig. 1 are given by
− →Vc = R− →i + Ld− →i
dt + ⃗Vg, (1)
where − →Vc, − →Vg, − →i , and R and L respectively represent phasors
of the voltage across the inverter’s output capacitance, the
grid voltage, the inverter current, and the line impedance
parameters.

Source Document: E:\Work\LM_RAG_Projects\open-rag-bench\data\raw\pdf\2410.14077v2.pdf, Page 0:
conditions, where the VDF-RLS method achieves more than
three time lower error compared to existing approaches such
as constant forgetting RLS and the Kalman filter.
I. INTRODUCTION
The rapid expansion of distributed energy resources
(DERs) and inverter-based loads has made inverter-based
grids increasingly common, driving the need for precise
power regulation and deeper insights into grid interactions. In
this context, output line impedance—the impedance between
the inverter and the grid—plays a crucial role in determining
inverter performance, affecting power injection limits and
droop control characteristics [1], [2]. Improved impedance
estimation, as highlighted in [3], enhances controller band-
width, while impedance also serves as an indicator of grid
stiffness and assists in islanding detection [4]. This capability
supports smooth operational transitions for inverters [5].
As inverter-based resources continue to expand, accurate

## Question:
What are the challenges in estimating output impedance in inverter-based grids?
"""
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
    
    # # 使用 get_openai_callback() 上下文管理器来捕获 token 消耗
    # from langchain.callbacks import get_openai_callback
    # with get_openai_callback() as cb:
    #     response = client.invoke(messages)
    #     # print(response.model_dump_json()["content"])
    #     print(response.content)
    #     print("\n--- Token 消耗详情 ---")
    #     print(f"总 token: {cb.total_tokens}")
    #     print(f"提示 token: {cb.prompt_tokens}")
    #     print(f"补全 token: {cb.completion_tokens}")
    """
    two time exps for token cost of qwen:
    exp1 : total tokens: 1712, prompt_token: 1355, completion_tokens:  357
    exp2 : total tokens: 1686, prompt_token: 1355, completion_tokens:  331
    """
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
