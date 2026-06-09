import os
import re

from agents.step_definer import task_define
from agents.rag import build_rag_agent
from src.utils import GraphState, PlanExecState, RagState, StepTaskFormat, StepTaskState, PlanFormat, PlanSummaryFormat, PlanSummaryState, QAAnswerFormat, QAAnswerState
from src.prompt_template import extract_system_messgage, extract_human_message, extract_input_variables
from src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from src.prompt_template import aggregate_human_message, aggregate_input_variables, aggregate_system_message
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.whale_llm import WhaleLLM
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState, StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

TASK_TYPE_FIELD_PATTERN = "\u4efb\u52a1\u7c7b\u578b[:\uff1a]\s*(\S+)"
TASK_TYPE_PROCESS = "\u5904\u7406"


def build_plan_executor(retriever_tool=None):

    rag_agent = build_rag_agent(retriever_tool=retriever_tool)

    async def single_task_execute(state: PlanExecState):

        cur_task = state["step_question"][-1]
        raw_query = cur_task["task"]

        parent_anchors = state.get("parent_anchors", [])

        if parent_anchors:
            anchor_prefix = " ".join(parent_anchors)
            query = f"{anchor_prefix}\n{raw_query}"
        else:
            query = raw_query

        type_match = re.search(TASK_TYPE_FIELD_PATTERN, query)

        if type_match:
            task_type = type_match.group(1).strip()
        else:
            task_type = TASK_TYPE_PROCESS
        step_doc_ids = []
        step_notes = []

        if task_type == TASK_TYPE_PROCESS:
            try:
                llm = WhaleLLM(model_name=os.getenv("MODEL_NAME"), temperature=0.2)

                memory_parts = []
                for idx, item in enumerate(state["step_output"]):
                    memory_parts.append(
                        f"Step {idx + 1}: {state['plan'][idx]}\nAnswer: {item['answer']}\n"
                    )
                memory = "\n".join(memory_parts)

                user_prompt = f"Complete the current task based on the following prior step outputs.\n{memory}\n\nCurrent task: {query}"
                system_prompt = aggregate_system_message
                reply = llm.chat(user_prompt=user_prompt, system_prompt=system_prompt)
            except Exception:
                import traceback
                print("[single_task_execute][process] WhaleLLM.chat failed")
                traceback.print_exc()
                raise

            response = QAAnswerState(
                answer=reply,
                analysis="",
                success="yes",
                rating=1.0,
            )

        else:
            step_index = len(state.get("step_output", [])) + 1
            try:
                response = await rag_agent.ainvoke({"question": query, "step_index": step_index})
            except Exception:
                import traceback
                print("[single_task_execute] rag_agent.invoke failed")
                traceback.print_exc()
                raise
            print("[single_task_execute] rag_agent.invoke returned type:", type(response))
            print("[single_task_execute] rag_agent.invoke :", response)
            step_doc_ids = [response.get("doc_ids") if isinstance(response, dict) else None]
            step_notes = [response.get("notes") if isinstance(response, dict) else None]
            response = response.get("final_raw_answer") if isinstance(response, dict) else response

            if isinstance(response, dict) and "success" not in response:
                response["success"] = "yes"

        return {"step_output": [response], "step_docs_ids": step_doc_ids, "step_notes": step_notes}

    def task_definer_out(state: PlanExecState):
        if state["stop"] is True:
            return END
        return "single_task_execute"

    graph_builder = StateGraph(PlanExecState)

    graph_builder.add_node("task_definer", task_define)
    graph_builder.add_node("single_task_execute", single_task_execute)
    graph_builder.add_edge(START, "task_definer")
    graph_builder.add_edge("single_task_execute", "task_definer")
    graph_builder.add_conditional_edges("task_definer", task_definer_out)
    graph = graph_builder.compile()
    return graph
