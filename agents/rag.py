import os
import re

from src.utils import RagState, QAAnswerState
from src.prompt_template import extract_system_messgage, extract_human_message
from src.prompt_template import qa_human_message, qa_system_message
from src.whale_llm import WhaleLLM
from langgraph.graph import StateGraph, START, END

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


def _normalize_answer_text(answer_text: str, fallback_message: str) -> str:
    normalized_lines = []
    for raw_line in str(answer_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        plain_line = CITATION_PATTERN.sub("", line).strip()
        if re.search(r"insufficient evidence from the provided (?:chunks|steps)\.?", plain_line, re.IGNORECASE):
            normalized_lines.append(fallback_message)
            continue

        if _is_unsupported_sentence(plain_line):
            cleaned = plain_line.strip()
            if cleaned and cleaned[-1] not in "。！？?!":
                cleaned += "。"
            normalized_lines.append(cleaned)
            continue

        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def build_rag_agent(retriever_tool=None):

    def _parse_cited_answer(answer_text: str, chunk_map: dict):
        lines = [line.strip() for line in str(answer_text).splitlines() if line.strip()]
        citation_items = []
        used_chunk_refs = []

        for line in lines:
            refs = CITATION_PATTERN.findall(line)
            if not refs:
                continue

            sentence_text = CITATION_PATTERN.sub("", line).strip()
            dedup_refs = []
            for ref in refs:
                ref_key = str(ref)
                if ref_key not in dedup_refs:
                    dedup_refs.append(ref_key)

            resolved_chunks = [chunk_map.get(ref_key) for ref_key in dedup_refs if chunk_map.get(ref_key) is not None]
            citation_items.append(
                {
                    "sentence": sentence_text,
                    "ref_ids": dedup_refs,
                    "chunks": resolved_chunks,
                    "raw_text": line,
                }
            )
            used_chunk_refs.extend(dedup_refs)

        used_chunk_refs = sorted(set(used_chunk_refs))
        return citation_items, used_chunk_refs

    async def retrieve(state: RagState):
        user_question = state["question"]
        step_index = state.get("step_index", 1)

        retrieval_result = await retriever_tool.retrieve(user_question, step_index=step_index)
        if isinstance(retrieval_result, tuple):
            list_docs, list_doc_ids = retrieval_result
            chunks = []
            chunk_map = {}
            wrapped_documents = ""
        else:
            list_docs = retrieval_result.get("documents", [])
            list_doc_ids = retrieval_result.get("doc_ids", [])
            chunks = retrieval_result.get("chunks", [])
            chunk_map = retrieval_result.get("chunk_map", {})
            wrapped_documents = retrieval_result.get("wrapped_documents", "")

        state["documents"] = list_docs
        state["doc_ids"] = list_doc_ids
        state["chunks"] = chunks
        state["chunk_map"] = chunk_map
        state["wrapped_documents"] = wrapped_documents
        print("[retrieve] user_question:", user_question)
        print("[retrieve] step_index:", step_index)
        print("[retrieve] doc_ids:", list_doc_ids)
        print("[retrieve] chunk_map:", chunk_map)
        return state

    def extract(state: RagState):
        list_docs = state.get("documents", [])
        user_question = state["question"]
        sys_msg = extract_system_messgage
        llm = WhaleLLM(model_name=os.getenv("MODEL_NAME"), temperature=0.0)
        list_notes = []
        for idx, doc in enumerate(list_docs):
            user_msg = extract_human_message.format(passage=doc, question=user_question)
            try:
                note = llm.chat(user_prompt=user_msg, system_prompt=sys_msg)
            except Exception:
                import traceback
                print(f"[extract] LLM.chat failed on doc idx={idx}, doc_id/info maybe missing")
                traceback.print_exc()
                raise
            print(f"[extract] got note type={type(note)}, len={len(note) if note is not None else 'None'}")
            list_notes.append(f"[{note}]")
        state["notes"] = list_notes
        return state

    def generate(state: RagState):
        question = state["question"]
        docs = state.get("wrapped_documents", "")
        if not docs:
            tmps = []
            for doc_id, note in zip(state.get("doc_ids", []), state.get("notes", [])):
                tmps.append(f"doc_{doc_id}: {note}")
            docs = "\n\n".join(tmps)

        sys_msg = qa_system_message
        user_msg = qa_human_message.format(context=docs, question=question)

        llm = WhaleLLM(model_name=os.getenv("MODEL_NAME"), temperature=0.1)
        try:
            raw_answer_text = llm.chat(user_prompt=user_msg, system_prompt=sys_msg)
        except Exception:
            import traceback
            print("[generate] LLM.chat raised exception")
            traceback.print_exc()
            raise

        answer_text = _normalize_answer_text(
            raw_answer_text,
            fallback_message="??????????????",
        )

        chunk_map = state.get("chunk_map", {})
        citation_items, used_chunk_refs = _parse_cited_answer(answer_text, chunk_map)

        print(f"[generate] answer_text type={type(answer_text)} len={len(answer_text) if answer_text is not None else 'None'}")
        print("[generate] raw_answer_text:", raw_answer_text)
        print("[generate] normalized_answer_text:", answer_text)
        print("[generate] citation_items:", citation_items)

        try:
            response = QAAnswerState(
                answer=answer_text,
                analysis="RAG with citations",
                success="yes",
                rating=1.0,
            )
        except Exception as e:
            print("[generate] QAAnswerState construction failed:", e)
            import traceback
            traceback.print_exc()
            response = {
                "answer": answer_text,
                "analysis": "RAG with citations",
                "success": "yes",
                "rating": 1.0,
            }

        response["chunk_map"] = chunk_map
        response["chunks"] = state.get("chunks", [])
        response["citation_items"] = citation_items
        response["used_chunk_refs"] = used_chunk_refs
        return {
            "final_raw_answer": response,
            "doc_ids": state.get("doc_ids", []),
            "notes": state.get("notes", []),
            "chunk_map": chunk_map,
            "chunks": state.get("chunks", []),
        }

    rag_graph_builder = StateGraph(RagState)
    rag_graph_builder.add_node("retrieve", retrieve)
    rag_graph_builder.add_node("extract", extract)
    rag_graph_builder.add_node("generate", generate)

    rag_graph_builder.add_edge(START, "retrieve")
    rag_graph_builder.add_edge("retrieve", "extract")
    rag_graph_builder.add_edge("extract", "generate")
    rag_graph_builder.add_edge("generate", END)
    rag_graph = rag_graph_builder.compile()
    return rag_graph
