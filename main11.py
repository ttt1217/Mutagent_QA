import os
import json
import pickle
import numpy as np
import argparse
import time
from typing import List, Optional
from pydantic import BaseModel

import asyncio

from langgraph.graph import MessagesState, StateGraph, START, END
from corpus.retrieve import Retriever
from src.utils import load_dataset, load_corpus, parse_args
from src.utils import GraphState, RetrieveTopChunk
from agents.plan_executor import build_plan_executor
from agents.plan import plan_agent
from corpus.embed_corpus import build_local_corpus

from src.retrieval_client import RetrievalClient

from dotenv import load_dotenv
load_dotenv()

#  LLM 配置
LLM_API_KEY = "ailab_zqeUaTfVYu3cbR44jTE73Sw9doTam4YgGX/9uzRQEAs7PA8hKEYJ2N77GpZbOkJPFBhxcVpEez317gwuK8D29Ssa99p4GcX2ZN71jnMGi1AlUC3+UXAFIuc="
#LLM_API_KEY = "ailab_B5YNZRu+Zl3P0/RDPLCJkuGel3PyDlgYID2kYibg8xGjGdfXIETE/nSBSNynMNiyjJq8ERd33wSLpZSlMnY+/O3CCE8ZoUYrW5aj3wUu4fZo21AIYzNXJqM="
LLM_URL = "https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1"
LLM_MODEL = "Qwen/Qwen3-235B-A22B"

class Message(BaseModel):
    role: Optional[str] = "user"
    content: Optional[str] = ""

def call_llm(message: Message, **kwargs) -> List[str]:

    from openai import OpenAI, OpenAIError

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_URL)
    try:
        chat_response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[message.model_dump()],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        response = [choice.message.content.strip() for choice in chat_response.choices]
    except OpenAIError as e:
        print(f" LLM 请求失败: {str(e)}")
        return ["__LLM_CALL_FAILED__"]

    return response

#  计划执行节点
def plan_executor_node(state: GraphState):
    input = {
        "original_question": state["original_question"],
        "plan": state.get("plan", []),
        "stop": False
    }
    print("\n [plan_executor_node] 输入计划：")
    print(json.dumps(input, ensure_ascii=False, indent=2))

    output = asyncio.run(plan_executor_agent.ainvoke(input))

    #output = plan_executor_agent.invoke(input)

    print("\n [plan_executor_node] 输出结果：")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return {"past_exp": [output]}


if __name__ == "__main__":
    args = parse_args()

    # === Debug print args to ensure start/end indices ===
    # print("\n--- RUN ARGS ---")
    # # print all fields available in args for clarity
    # try:
    #     print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    # except Exception:
    #     print("args:", args)

    # #  构建语料
    # print("\n=== Step 1: 构建本地语料 ===")
    # doc_ids, docs = build_local_corpus()
    # print(f" 已加载 {len(docs)} 条语料:")
    # for i, text in enumerate(docs):
    #     print(f"  - {i}: {text}")

    #  动态生成嵌入
    # print("\n=== Step 2: 动态生成语料嵌入 ===")
    # retriever_tool = RetrieveTopChunk(retrieval_model=None, corpus=None, top_k=5)
    # # debug wrapper: print before/after embed call
    # print("[DEBUG] 调用 retriever_tool.embed_text(...) 生成语料嵌入")
    # corpus_embeddings = retriever_tool.embed_text(docs)
    # corpus_embeddings = np.array(corpus_embeddings, dtype="float32")
    # print(f" 已生成 {len(corpus_embeddings)} 条嵌入，维度: {corpus_embeddings.shape[1]}")


    # 构建检索器
    # print("\n=== Step 3: 构建 Retriever 检索器 ===")
    # retriever = Retriever()
    # retriever.load_emb_data({
    #     "doc_ids": doc_ids,
    #     "embeddings": corpus_embeddings,
    #     "texts": docs
    # })
    #
    # corpus = {doc_id: {"text": text} for doc_id, text in zip(doc_ids, docs)}
    #
    # retriever_tool.retrieval_model = retriever
    # retriever_tool.corpus = corpus
    # print(" Retriever 初始化完成")

    # # 构建 Plan Executor
    # print("\n=== Step 4: 构建 Plan Executor ===")
    # global plan_executor_agent
    # plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)
    # print(" plan_executor_agent 初始化成功")

    #print("\n=== Step 1: 初始化 Retrieval Client ===")
    # ⚠️ 这里填写 retrieval_service.py 启动的地址
    retriever_tool = RetrievalClient(base_url="http://localhost:7001")
    print(" 已连接到远程 MA-RAG Retrieval API")

    #print("\n=== Step 2: 构建 Plan Executor ===")
    global plan_executor_agent
    plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)
    #print(" plan_executor_agent 初始化成功")


    # 构建 LangGraph 图
    #print("\n=== Step 5: 构建 LangGraph 流程图 ===")
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("planer_node", plan_agent)
    graph_builder.add_node("plan_executor_node", plan_executor_node)
    graph_builder.add_edge(START, "planer_node")
    graph_builder.add_edge("planer_node", "plan_executor_node")
    graph = graph_builder.compile()
    #print(" LangGraph 构建完成")

    #  数据集
    exp_name = getattr(args, "exp", "exp")
    model = getattr(args, "model", "model")
    dataset_name = getattr(args, "dataset", "dataset")

    dataset = [
        #{"id": "q1", "input": "The arena where the Lewiston Maineiacs played their home games can seat how many people?"},
        # {"id": "q2", "input": "2019年浙江海翔药业股份有限公司主要前五大销售客户里，销售额最大的为多少？"},
        # {"id": "q3", "input": "2019年浙江海翔药业股份有限公司医药行业的销售、生成和库存量分别是多少？"},
        # {"id": "q4", "input": "2019年浙江海翔药业股份有限公司的营业收入为多少？"},
        # {"id": "q5", "input": "浙江海翔药业股份有限公司董事长的背景？"},
        # {"id": "q6", "input": "2019年江苏安靠智能输电工程科技股份有限公司的华南地区营业收入为多少？占营业收入比例为多少？"},
        # {"id": "q7", "input": "2019年江苏安靠智能输电工程科技股份有限公司的营业收入为多少？"},
        # {"id": "q8", "input": "2019年江苏安靠智能输电工程科技股份有限公司募投项目的情况？"},
        # {"id": "q9", "input": "江苏安靠智能输电工程科技股份有限公司的主要产品及其用途是什么？"},
        # {"id": "q10", "input": "2019年末焦点科技总资产？"},
        {"id": "q11", "input": "2019年江苏安靠智能输电工程科技股份有限公司的华南地区营业收入为多少？占营业收入比例为多少？"},
    ]

    total = len(dataset)

    start_index = getattr(args, "start_index", None)
    end_index = getattr(args, "end_index", None)

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = total - 1

    # normalize and clamp
    try:
        start_index = int(start_index)
    except Exception:
        start_index = 0
    try:
        end_index = int(end_index)
    except Exception:
        end_index = total - 1

    if start_index < 0:
        start_index = 0
    if end_index >= total:
        end_index = total - 1

    #print(f"\n[DEBUG] dataset length={total}, start_index={start_index}, end_index={end_index}")

    save_dir = f"{exp_name}_{model}_{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    # main loop
    for idx, item in enumerate(dataset):
        # This will now follow the explicit bounds above
        if idx < start_index or idx > end_index:
            print(f"[DEBUG] skip idx={idx} (outside bounds)")
            continue

        question_id = item["id"]
        save_file = os.path.join(save_dir, f"{question_id}.json")

        if os.path.exists(save_file):
            print(f"[DEBUG] result exists, skip {save_file}")
            continue

        question = item["input"]
        #print(f"\n=== Step 6.{idx+1}: 处理问题 [{question}] ===")
        inputs = {
            "original_question": question,
            "plan": [],
            "past_exp": [],
            "final_answer": ""
        }
        start_time = time.time()
        try:
            #print("调用 graph.invoke() ...")

            output = asyncio.run(graph.ainvoke(inputs))

            #output = graph.invoke(inputs)
            print("\n 最终输出：")
            print(json.dumps(output, ensure_ascii=False, indent=2))

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f" 已保存结果到 {save_file}")
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"🕒 [总耗时] {elapsed:.2f} 秒")
        except Exception as e:
            import traceback
            print("===  Exception when calling graph.invoke ===")
            traceback.print_exc()
            print("inputs repr:", repr(inputs))
            try:
                print("type(plan_executor_agent):", type(plan_executor_agent))
            except NameError:
                pass
            try:
                print("type(retriever_tool):", type(retriever_tool))
            except NameError:
                pass
            break
