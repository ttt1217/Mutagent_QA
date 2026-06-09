import os
import re

from langchain_core.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
)
from src.utils import PlanExecState, PlanSummaryState, StepTaskState
from src.prompt_template import (
    step_system_message, step_human_message, step_input_variables,
    summary_system_message, summary_human_message, summary_input_variables
)
from src.whale_llm import WhaleLLM
from dotenv import load_dotenv

load_dotenv()


CITATION_PATTERN = re.compile(r"\[(\d+(?:-\d+)?)\]")
UNSUPPORTED_REGEXES = [
    re.compile(r"insufficient evidence", re.IGNORECASE),
    re.compile(r"cannot answer.{0,24}(?:provided|reference|chunk|context)", re.IGNORECASE),
    re.compile(r"cannot be answered.{0,24}(?:provided|reference|chunk|context)", re.IGNORECASE),
]
UNSUPPORTED_SUBSTRINGS = [
    "该问题在参考资料中无法回答",
    "该问题无法根据参考资料回答",
    "根据参考资料无法回答",
    "根据提供的资料无法回答",
    "根据提供的参考资料无法回答",
    "参考资料中无法回答",
    "参考资料不足以回答",
    "无法从参考资料中回答",
    "无法基于参考资料回答",
]
REFERENCE_KEYWORDS = [
    "参考资料",
    "资料",
    "材料",
    "文档",
    "文中",
    "提供的资料",
    "提供的参考资料",
    "上述资料",
]
UNSUPPORTED_VERBS = [
    "无法回答",
    "不能回答",
    "无法作答",
    "不能作答",
    "无法确定",
]
MISSING_INFO_KEYWORDS = [
    "资料不足",
    "信息不足",
    "证据不足",
    "依据不足",
    "未找到",
    "未提供",
    "未提及",
    "未说明",
]


def _is_unsupported_sentence(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False

    lowered = normalized.lower()
    if any(keyword in lowered for keyword in ("insufficient evidence", "cannot answer based on", "cannot be answered based on")):
        return True

    if any(keyword in normalized for keyword in UNSUPPORTED_SUBSTRINGS):
        return True

    has_reference = any(keyword in normalized for keyword in REFERENCE_KEYWORDS)
    has_unsupported_verb = any(keyword in normalized for keyword in UNSUPPORTED_VERBS)
    has_missing_info = any(keyword in normalized for keyword in MISSING_INFO_KEYWORDS)

    if has_missing_info:
        return True
    if has_reference and has_unsupported_verb:
        return True

    return any(pattern.search(normalized) for pattern in UNSUPPORTED_REGEXES)


def _normalize_summary_text(summary_text: str) -> str:
    normalized_lines = []
    for raw_line in str(summary_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        plain_line = CITATION_PATTERN.sub("", line).strip()
        if re.search(r"insufficient evidence from the provided (?:chunks|steps)\.?", plain_line, re.IGNORECASE):
            normalized_lines.append("该问题在参考资料中无法回答。")
            continue

        if _is_unsupported_sentence(plain_line):
            cleaned = plain_line.strip()
            if cleaned and cleaned[-1] not in "。！？?!":
                cleaned += "。"
            normalized_lines.append(cleaned)
            continue

        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def task_define(state: PlanExecState):

    step_prompt = ChatPromptTemplate(
        input_variables=step_input_variables,
        messages=[
            SystemMessagePromptTemplate.from_template(step_system_message),
            HumanMessagePromptTemplate.from_template(step_human_message),
        ]
    )

    llm = WhaleLLM(model_name=os.getenv("MODEL_NAME"), temperature=0.3)

    should_stop = (
        len(state["step_output"]) == len(state["plan"])
        or (len(state["step_output"]) > 0 and state["step_output"][-1]["success"].lower() == "no")
    )

    if should_stop:
        summary_prompt = ChatPromptTemplate(
            input_variables=summary_input_variables,
            messages=[
                SystemMessagePromptTemplate.from_template(summary_system_message),
                HumanMessagePromptTemplate.from_template(summary_human_message),
            ]
        )

        llm_summary = WhaleLLM(model_name=os.getenv("MODEL_NAME"), temperature=0.0)

        question = state["original_question"]
        plan = f"[{', '.join(state['plan'])}]"

        memory_parts = []
        for idx, item in enumerate(state["step_output"]):
            used_refs = item.get("used_chunk_refs", []) if isinstance(item, dict) else []
            refs_text = " ".join([f"[{ref}]" for ref in used_refs]) if used_refs else "none"
            cited_answer = item.get("answer", "") if isinstance(item, dict) else ""
            memory_parts.append(
                f"Step {idx + 1}: {state['plan'][idx]}\n"
                f"Question: {state['step_question'][idx]['task']}\n"
                f"Answer with citations:\n{cited_answer}\n"
                f"Available citations in this step: {refs_text}\n"
                f"Confident score: {item.get('rating', '') if isinstance(item, dict) else ''}\n"
            )
        memory = "\n".join(memory_parts)

        sys_msg = summary_system_message
        user_msg = summary_human_message.format(question=question, plan=plan, memory=memory)
        raw_summary_text = llm_summary.chat(user_prompt=user_msg, system_prompt=sys_msg)
        summary_text = _normalize_summary_text(raw_summary_text)

        output = PlanSummaryState(
            output=summary_text,
            score=1.0
        )
        return {"plan_summary": output, "stop": True}

    else:
        plan = f"[{', '.join(state['plan'])}]"
        cur_step = state["plan"][len(state["step_output"])]

        memory_parts = []
        for idx in range(len(state["step_output"])):
            memory_parts.append(
                f"Task: {state['plan'][idx]}\nAnswer: {state['step_output'][idx]['answer']}\n"
            )
        memory = "\n".join(memory_parts)

        sys_msg = step_system_message
        user_msg = step_human_message.format(plan=plan, cur_step=cur_step, memory=memory)

        try:
            step_text = llm.chat(user_prompt=user_msg, system_prompt=sys_msg)
        except Exception as e:
            import traceback
            print("[step_definer] llm.chat failed")
            traceback.print_exc()
            raise
        print("[step_definer] step_text :", step_text)

        response = StepTaskState(
            task=step_text,
            type="normal",
            success="yes",
            reason="WhaleLLM generated step"
        )
        return {"step_question": [response]}
