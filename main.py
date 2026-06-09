import os
import re
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
#LLM_API_KEY = "ailab_zqeUaTfVYu3cbR44jTE73Sw9doTam4YgGX/9uzRQEAs7PA8hKEYJ2N77GpZbOkJPFBhxcVpEez317gwuK8D29Ssa99p4GcX2ZN71jnMGi1AlUC3+UXAFIuc="
LLM_API_KEY = "ailab_B5YNZRu+Zl3P0/RDPLCJkuGel3PyDlgYID2kYibg8xGjGdfXIETE/nSBSNynMNiyjJq8ERd33wSLpZSlMnY+/O3CCE8ZoUYrW5aj3wUu4fZo21AIYzNXJqM="
# LLM_API_KEY = "ailab_R9G08Jakd2KMLtJYIdyculXyHxYSUMPqa1sNVWmm0dJ2sbwTM41G0EJeG73X4J4NCGng39D3A96RKxglC0XB4ni0r0iXLXnqjQS6TYD2UwY4qf5shD8ew64="
LLM_URL = "https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1"
LLM_MODEL = "Qwen/Qwen3-235B-A22B"

# class Message(BaseModel):
#     role: Optional[str] = "user"
#     content: Optional[str] = ""
#
# def call_llm(message: Message, **kwargs) -> List[str]:
#
#     from openai import OpenAI, OpenAIError
#
#     client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_URL)
#     try:
#         chat_response = client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[message.model_dump()],
#             temperature=kwargs.get("temperature", 0.0),
#             max_tokens=kwargs.get("max_tokens", 1024),
#         )
#         response = [choice.message.content.strip() for choice in chat_response.choices]
#     except OpenAIError as e:
#         print(f" LLM 请求失败: {str(e)}")
#         return ["__LLM_CALL_FAILED__"]
#
#     return response

# def extract_parent_semantic_anchors(question: str) -> dict:
#     prompt = f"""
# 请从下面问题中，抽取【用于文档检索的核心语义锚点】，
# 这些锚点应该是：
# - 能唯一或强烈约束检索范围的词或短语
# - 通常来自实体、时间、主题、限定条件
# - 不要求固定字段名
#
# 以 JSON 返回，格式如下：
# {{
#   "anchors": ["锚点1", "锚点2", "..."]
# }}
#
# 问题：
# {question}
#
# 要求：
# - anchors 不超过 6 个
# - 不要解释
# """
#
#     resp = call_llm(
#         Message(content=prompt),
#         temperature=0.0,
#         max_tokens=256
#     )[0]
#
#     print("🔍 raw anchor llm resp:\n", resp)
#
#     # 关键修复点：抽取 JSON 块
#     match = re.search(r"\{[\s\S]*\}", resp)
#     if not match:
#         print("⚠️ no json object found in anchor response")
#         return {"anchors": []}
#
#     json_str = match.group(0)
#
#     try:
#         data = json.loads(json_str)
#         anchors = data.get("anchors", [])
#     except Exception as e:
#         print("⚠️ anchor json parse failed:", e)
#         anchors = []
#
#     print("✅ extracted parent anchors:", anchors)
#     return {"anchors": anchors}

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
        # {"id": "q1", "input": "The arena where the Lewiston Maineiacs played their home games can seat how many people?"},
        # {"id": "q2", "input": "2019年浙江海翔药业股份有限公司主要前五大销售客户里，销售额最大的为多少？"},
        # {"id": "q3", "input": "2019年浙江海翔药业股份有限公司医药行业的销售量、生成量和库存量分别是多少？"},
        # {"id": "q4", "input": "2019年浙江海翔药业股份有限公司的营业收入为多少？"},
        {"id": "q5", "input": "浙江海翔药业股份有限公司董事长的背景？"},
        #{"id": "q6", "input": "2019年江苏安靠智能输电工程科技股份有限公司的华南地区营业收入为多少？占营业收入比例为多少？"},
        # {"id": "q7", "input": "2019年江苏安靠智能输电工程科技股份有限公司的营业收入为多少？"},
        # {"id": "q8", "input": "2019年江苏安靠智能输电工程科技股份有限公司募投项目的情况？"},
        # {"id": "q9", "input": "江苏安靠智能输电工程科技股份有限公司的主要产品及其用途是什么？"},
        # {"id": "q10", "input": "2019年末焦点科技总资产？"},
        # {"id": "q11", "input": "2019年江苏安靠智能输电工程科技股份有限公司的华南地区营业收入为多少？占营业收入比例为多少？"},
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

            # anchors_info = extract_parent_semantic_anchors(question)
            # retriever_tool.set_parent_semantic_anchors(anchors_info["anchors"])

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
