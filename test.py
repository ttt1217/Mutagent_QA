import os
import random
import json
import numpy as np
import argparse

from langgraph.graph import MessagesState, StateGraph, START, END
from transformers import AutoTokenizer, AutoModel
from corpus.retrieve import Retriever
from src.utils import load_dataset, load_corpus, parse_args
from src.utils import GraphState
from src.utils import RetrieveTopChunk
from agents.plan_executor import build_plan_executor
from agents.plan import plan_agent

from dotenv import load_dotenv
load_dotenv()

import time

def dynamic_planner_node(state: GraphState):
    """
    动态规划执行节点：循环进行（planner -> retrieve/execute）直至 finish 或达到最大步数。
    依赖外部全局变量：
      - plan_agent: 负责生成下一步 action（返回 JSON）
      - plan_executor_agent: 执行器（接受 {'original_question','plan','context'}）
      - retriever_tool: retriever 对象，需有 retrieve(query, top_k) 或 get_topk(...) 方法
    """
    original_q = state["original_question"]
    max_steps = state.get("max_steps", 6)   # 上限，避免无限循环
    top_k = state.get("top_k", 5)

    # 内部状态
    retrieved_docs = []   # list[str] 累积的证据文本
    evidence_ids = set()  # 去重用
    execution_history = []  # executor 历史
    plan_history = []       # planner 历史

    # initial prompt query can be the original question
    current_query = original_q

    for step in range(max_steps):
        # 1) call planner with current context
        planner_input = {
            "question": original_q,
            "current_query": current_query,
            "retrieved_docs": retrieved_docs[-50:],   # 截断避免太长，保留最近 50 段
            "execution_history": execution_history,
            "plan_history": plan_history,
            "step": step,
            "max_steps": max_steps
        }

        # planner returns a structured dict. If your plan_agent returns text,
        # you should parse it into the schema below (e.g., via json.loads or simple regex).
        planner_out = plan_agent.invoke(planner_input)  # 你现有接口
        # try normalize to dict
        if isinstance(planner_out, str):
            try:
                planner_dict = json.loads(planner_out)
            except Exception:
                # 如果 planner 返回自由文本，做最简单的解析（fallback）
                # 解析出 next_action: look for keywords
                text = planner_out.lower()
                if "retrieve" in text or "search" in text:
                    planner_dict = {"next_action": "retrieve", "action_args": {"query": current_query}}
                elif "finish" in text or "final" in text or "answer" in text:
                    planner_dict = {"next_action": "finish", "action_args": {"answer": planner_out}}
                else:
                    # 默认尝试执行
                    planner_dict = {"next_action": "execute", "action_args": {"plan": planner_out}}
        else:
            planner_dict = planner_out

        plan_history.append(planner_dict)

        next_action = planner_dict.get("next_action", "retrieve")
        args = planner_dict.get("action_args", {})

        # 2) Branch on action
        if next_action == "retrieve":
            q_for_retrieval = args.get("query", current_query)
            # call retriever_tool - adjust if your API name differs
            hits = retriever_tool.retrieve(q_for_retrieval, top_k=top_k)
            # hits expected to be list of dicts: {'id':..., 'text':...}
            new_added = 0
            for h in hits:
                hid = h.get("id", None) or h.get("doc_id", None) or h.get("cursor", None)
                text = h.get("text") if "text" in h else h.get("content", h)
                if hid is None:
                    hid = text[:120]  # fallback id
                if hid not in evidence_ids:
                    evidence_ids.add(hid)
                    retrieved_docs.append(text)
                    new_added += 1
            # update current_query if planner suggested a refined query
            current_query = args.get("refined_query", q_for_retrieval)
            # if nothing new found, consider either finishing or broadening query
            if new_added == 0:
                # mark planner to refine or stop
                # fallback: try broaden by using original question once
                if current_query != original_q:
                    current_query = original_q
                    continue
                else:
                    # nothing new after trying original question -> ask planner to finish
                    # call planner one more time with 'no_new_evidence' flag
                    # NOTE: to avoid cycles, set next_action to finish
                    return {"final_result": {"status":"no_more_evidence","retrieved_docs": retrieved_docs, "plan_history": plan_history, "execution_history": execution_history}}
            else:
                # go next iteration so planner can use new retrieved docs
                continue

        elif next_action == "execute":
            plan_text = args.get("plan") or planner_dict.get("plan")
            exec_input = {
                "original_question": original_q,
                "plan": plan_text,
                "context": retrieved_docs[-top_k:],   # send recent top-k as context
            }
            exec_out = plan_executor_agent.invoke(exec_input)
            # normalize executor output
            if isinstance(exec_out, str):
                exec_dict = {"result": exec_out, "evidence": retrieved_docs[-top_k:], "stop": False}
            else:
                exec_dict = exec_out
            execution_history.append(exec_dict)
            # If executor requests stop or planner indicated stop, we finish
            if exec_dict.get("stop", False):
                return {"final_result": {"status":"finished_by_executor", "retrieved_docs": retrieved_docs, "plan_history": plan_history, "execution_history": execution_history}}
            # otherwise continue loop to let planner refine next step
            # optionally update current_query from exec result
            current_query = args.get("followup_query", current_query)
            continue

        elif next_action == "finish" or next_action == "finalize":
            answer = args.get("answer", planner_dict.get("reason", "No answer provided"))
            return {"final_result": {"status":"finished_by_planner", "answer": answer, "retrieved_docs": retrieved_docs, "plan_history": plan_history, "execution_history": execution_history}}

        else:
            # unknown action: treat as execute
            plan_text = args.get("plan") or planner_dict.get("plan", "")
            exec_input = {"original_question": original_q, "plan": plan_text, "context": retrieved_docs[-top_k:]}
            exec_out = plan_executor_agent.invoke(exec_input)
            execution_history.append(exec_out if isinstance(exec_out, dict) else {"result": str(exec_out)})
            continue

    # reached max steps -> aggregate and return partial result
    return {"final_result": {"status":"max_steps_reached","retrieved_docs": retrieved_docs, "plan_history": plan_history, "execution_history": execution_history}}


if __name__ == "__main__":
    args = parse_args()

    model_name_or_path = 'hf/gte-multilingual-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.to(f"cuda:{args.gpus[0]}")

    retrieve = Retriever(gpu_ids=args.gpus)
    retrieve.init_index_and_add(root_dir="save_embs/gte-ml-base", dataset_name="dpr100")

    dpr100_corpus = load_corpus()
    retriever_tool = RetrieveTopChunk(tokenizer=tokenizer, embedding_model=model, retrieval_model=retrieve,
                                      corpus=dpr100_corpus)

    global plan_executor_agent

    plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("dynamic_planner_node", dynamic_planner_node)
    graph_builder.add_edge(START, "dynamic_planner_node")
    graph_builder.add_edge("dynamic_planner_node", END)
    graph = graph_builder.compile()

    exp_name = args.exp
    model = args.model
    dataset_name = args.dataset
    dataset = load_dataset(name=dataset_name)

    c = 0
    save_dir = f"{exp_name}_{model}_{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    for id, item in enumerate(dataset):
        if id < args.start_index or id > args.end_index:
            continue
        question_id = item["id"]
        print(question_id)
        save_file = os.path.join(save_dir, f"{question_id}.json")
        if os.path.exists(save_file):
            continue
        if dataset_name == "fever":
            question = f"Verify this claim, answer SUPPORTS or REFUTES\n{item['input']}"
        else:
            question = item["input"]
        inputs = {
            "original_question": f"{question}?"
        }
        try:
            output = graph.invoke(inputs)
            print(output)
            print()

            with open(save_file, "w") as f:
                json.dump(output, f)
        except Exception as e:
            print(e)
            pass