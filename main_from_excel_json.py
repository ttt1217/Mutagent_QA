import os
import json
import time
import asyncio
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from corpus.retrieve import Retriever
from src.utils import parse_args, GraphState
from agents.plan_executor import build_plan_executor
from agents.plan import plan_agent
from src.retrieval_client import RetrievalClient

load_dotenv()

# -------------------- 模型配置 --------------------
#LLM_API_KEY = "ailab_XfFgDfq4a26+HiFjKDzuOY0A2Fx7xjSyuJQDShHbACxhoEcJ0rxpaPFKYeQ9rC58DIKPN7tryyVzzkvlkeC3qrkD4ibjJFVpjzLrwoNkUANKbJ8bcedxrgA="
LLM_API_KEY = "ailab_zqeUaTfVYu3cbR44jTE73Sw9doTam4YgGX/9uzRQEAs7PA8hKEYJ2N77GpZbOkJPFBhxcVpEez317gwuK8D29Ssa99p4GcX2ZN71jnMGi1AlUC3+UXAFIuc="
#LLM_API_KEY = "ailab_B5YNZRu+Zl3P0/RDPLCJkuGel3PyDlgYID2kYibg8xGjGdfXIETE/nSBSNynMNiyjJq8ERd33wSLpZSlMnY+/O3CCE8ZoUYrW5aj3wUu4fZo21AIYzNXJqM="
LLM_URL = "https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1"
LLM_MODEL = "Qwen/Qwen3-235B-A22B"

# -------------------- LLM 消息结构 --------------------
class Message(BaseModel):
    role: Optional[str] = "user"
    content: Optional[str] = ""

# -------------------- LLM 调用封装 --------------------
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
        print(f"❌ LLM 请求失败: {str(e)}")
        return ["__LLM_CALL_FAILED__"]
    return response

# -------------------- 计划执行节点 --------------------
def plan_executor_node(state: GraphState):
    input_data = {
        "original_question": state["original_question"],
        "plan": state.get("plan", []),
        "stop": False
    }
    print("\n[plan_executor_node] 输入计划：")
    print(json.dumps(input_data, ensure_ascii=False, indent=2))

    output = asyncio.run(plan_executor_agent.ainvoke(input_data))

    print("\n✅ [plan_executor_node] 输出结果：")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return {"past_exp": [output]}

# -------------------- 主函数 --------------------
if __name__ == "__main__":
    args = parse_args()

    # === Step 1: 初始化 Retrieval Client ===
    retriever_tool = RetrievalClient(base_url="http://localhost:7001")
    print("已连接到远程 MA-RAG Retrieval API")

    # === Step 2: 构建 Plan Executor ===
    global plan_executor_agent
    plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)
    print("plan_executor_agent 初始化成功")

    # === Step 3: 构建 LangGraph 流程 ===
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("planer_node", plan_agent)
    graph_builder.add_node("plan_executor_node", plan_executor_node)
    graph_builder.add_edge(START, "planer_node")
    graph_builder.add_edge("planer_node", "plan_executor_node")
    graph = graph_builder.compile()
    print("LangGraph 构建完成")

    # === Step 4: 读取 Excel 文件 ===
    input_excel = "Finance_query.xlsx"
    df = pd.read_excel(input_excel)

    if "query" not in df.columns:
        raise ValueError(" Excel 文件必须包含列名为 'query' 的列")

    queries = df["query"].dropna().tolist()
    total = len(queries)
    print(f"共加载 {total} 个问题。")

    save_dir = "results_from_excel"
    os.makedirs(save_dir, exist_ok=True)

    results = []
    total_time = 0

    # === Step 5: 批量运行 ===
    for idx, question in enumerate(queries, start=1):
        print(f"\n=== 🔹 问题 {idx}/{total}: {question} ===")

        inputs = {
            "original_question": question,
            "plan": [],
            "past_exp": [],
            "final_answer": ""
        }

        start_time = time.time()
        try:
            output = asyncio.run(graph.ainvoke(inputs))
            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed

            print(f"输出结果：")
            print(json.dumps(output, ensure_ascii=False, indent=2))
            print(f"耗时: {elapsed:.2f} 秒")

            # 保存单条结果
            single_result_path = os.path.join(save_dir, f"q{idx}.json")
            with open(single_result_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"已保存结果到 {single_result_path}")
            final_answer = output["past_exp"][-1]["plan_summary"]["output"]
            results.append({
                "question": question,
                "answer": final_answer,
                "elapsed_time": round(elapsed, 2)
            })

            summary_data = {
                "total_questions": len(queries),
                "completed": len(results),
                "results": results
            }
            summary_path = os.path.join(save_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            import traceback
            print("异常发生：")
            traceback.print_exc()
            results.append({
                "question": question,
                "answer": "__ERROR__",
                "elapsed_time": 0
            })
            continue

    # === Step 6: 统计平均耗时 ===
    valid_times = [r["elapsed_time"] for r in results if r["elapsed_time"] > 0]
    avg_time = np.mean(valid_times) if valid_times else 0
    print("\n=== 批处理统计结果 ===")
    print(f"共完成: {len(valid_times)} 个问题")
    print(f"平均耗时: {avg_time:.2f} 秒")

    # === Step 7: 汇总结果输出为 JSON ===
    summary_data = {
        "total_questions": len(queries),
        "completed": len(valid_times),
        "average_time": round(avg_time, 2),
        "results": results
    }

    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"汇总结果已保存至 {summary_path}")
