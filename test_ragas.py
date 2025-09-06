"""
test ragas for evaluation
"""
import os
from pydantic import SecretStr
import time
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# the test case is from openragbench and the rag based on qwen3-30b-a3b-instruct-2507
query = "What are the challenges in estimating output impedance in inverter-based grids?"
relevant_docs = [
"""
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
""",
"""
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
""",
"""
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
""",
"""
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
""",
"""
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
"""
]
# response = """
# The challenges in estimating output impedance in inverter-based grids, as outlined in the provided research document, are:

# 1. **Lack of Global or Network-Wide Measurements**: Inverters typically do not have access to global system data or measurements from other parts of the grid. This limits their ability to directly measure or estimate the effective grid voltage, which is essential for accurate impedance estimation.

# 2. **Insufficient Persistence of Excitation in Measured Signals**: Accurate impedance estimation requires input signals that sufficiently excite all dynamic modes of the system (i.e., persistent excitation). However, measured signals at the inverter often lack this property, especially under steady-state operation, making it difficult to reliably identify the impedance parameters.

# 3. **Limited Measurable Variables**: Inverters operate primarily in steady state and can only measure local quantities—specifically, the output voltage and current at their terminals. The line impedance and grid voltage are not directly measurable, and their indirect estimation is complicated by the absence of external data and signal richness.

# These challenges are compounded by dynamic changes in the power network such as load fluctuations, addition/removal of power sources, and environmental factors like temperature, which cause the actual grid impedance to vary significantly over time. Therefore, real-time, accurate, and non-invasive impedance estimation methods are crucial for maintaining stable power injection and ensuring effective control strategies in modern inverter-dominated grids.

# Source: Research Document — *Inverter Output Impedance Estimation in Power Networks: A Variable Direction Forgetting Recursive-Least-Square Algorithm Based Approach* (Pages 0–1)
# """
response = """
The main challenges are:  

1. **Limited measurement scope** – Inverters only see local voltage and current; they lack global or network‑wide data needed to estimate the effective grid voltage.
2. **Insufficient excitation** – The signals measured at the inverter often do not contain enough persistence of excitation, which is essential for accurate impedance estimation.
3. **Steady‑state operation** – Inverters normally operate at steady state, so only local output voltage and current are measurable, while the line impedance and grid voltage must be inferred indirectly.
4. **Dynamic grid variations** – The Thevenin‑equivalent impedance can change rapidly due to load fluctuations, source additions/removals, and environmental factors, adding uncertainty to the estimation.
"""
reference = "Estimating output impedance in inverter-based grids is challenging due to dynamic grid conditions, which require real-time estimation. Traditional methods like signal injection and historical data analysis have limitations, such as sensitivity to noise and complexity."
try:
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        api_key=SecretStr(api_key) if api_key else None,
        base_url="https://sg.uiuiapi.com/v1", # uiuiapi
        model="gpt-4o" # the evaluation llm is gpt-4o
    )
    dataset = []
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(llm)
    start_time = time.time()
    result = evaluate(dataset=evaluation_dataset,metrics=\
        [LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
    print(result)
    # first exp:{'context_recall': 0.5000, 'faithfulness': 0.9231, 'factual_correctness(mode=f1)': 0.1300}
    # second exp: {'context_recall': 0.5000, 'faithfulness': 0.8462, 'factual_correctness(mode=f1)': 0.2700}
    end_time = time.time()
    cost = end_time - start_time
    print(f"evaluation costs {cost:.2f} s") # 17.96 s
    
    # # use call back of Langchain to compute the token cost
    # from langchain.callbacks import get_openai_callback # 导入回调函数
    # with get_openai_callback() as cb:
    #     start_time = time.time()
    #     result = evaluate(
    #         dataset=evaluation_dataset,
    #         metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    #         llm=evaluator_llm
    #     )
    #     end_time = time.time()

    #     print(result)
    #     print(f"evaluation costs {end_time - start_time:.2f} s")
    #     print("--- Token 消耗详情 ---")
    #     print(f"总 token: {cb.total_tokens}")
    #     print(f"提示 token: {cb.prompt_tokens}")
    #     print(f"补全 token: {cb.completion_tokens}")
    #     print(f"总花费: ${cb.total_cost:.4f}")

    """
    two time exps for token cost of rags:
    exp1 : total tokens: 10913, prompt_token: 8343, completion_tokens:  2570
    exp2 : total tokens: 11431, prompt_token: 8445, completion_tokens:  2986
    """
except Exception as e:
    print(f"错误信息：{e}")
