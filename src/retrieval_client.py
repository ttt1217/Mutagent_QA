# src/retrieval_client.py
import aiohttp
import json


class RetrievalClient:
    """
    Remote retrieval client for MA-RAG.
    """
    def __init__(self, base_url: str, default_dataset_ids=None):
        self.base_url = base_url.rstrip("/")
        self.default_dataset_ids = default_dataset_ids or ["1485090178244919296"]
        self.parent_anchors = []

    def set_parent_semantic_anchors(self, anchors: list[str]):
        self.parent_anchors = anchors

    def _build_retrieval_query(self, sub_query: str) -> str:
        parts = []
        parts.extend(self.parent_anchors)
        parts.append(sub_query)
        return " ".join(parts)

    def _format_chunk_for_model(self, ref_id: str, doc_id, document_name: str, content: str) -> str:
        return (
            f"[Chunk {ref_id}]\n"
            f"doc_id: {doc_id}\n"
            f"documentName: {document_name}\n"
            f"content: {content}\n"
            f"[/Chunk {ref_id}]"
        )

    async def retrieve(self, query: str, dataset_ids=None, size=20, threshold_score=0.6, step_index=1):
        """
        Call remote /api/retrieve and return wrapped chunks plus a stable chunk map.
        Citation ids follow the [step-chunk] format, e.g. [1-1], [1-2], [2-1].
        """
        dataset_ids = dataset_ids or self.default_dataset_ids
        url = f"{self.base_url}/api/retrieve"

        final_query = self._build_retrieval_query(query)
        print("[retrieve] FINAL RETRIEVAL QUERY =", final_query)

        payload = {
            "query": final_query,
            "query_type": "mixedRecall",
            "dataset_ids": dataset_ids,
            "size": size,
            "threshold_score": threshold_score,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Retrieval service error ({resp.status}): {text}")
                data = await resp.json()

        print(json.dumps(data, ensure_ascii=False, indent=2))

        text_items = data.get("text", [])
        documents, doc_ids, chunks = [], [], []
        chunk_map = {}
        wrapped_chunks = []

        step_prefix = str(step_index)
        next_chunk_no = 1
        for item in text_items:
            data_item = item.get("data", {})
            chunk_id = data_item.get("chunk_id")
            doc_id = data_item.get("doc_id") or data_item.get("document_id")
            document_name = (data_item.get("documentName") or "").strip()
            content = (data_item.get("content") or "").strip()
            if doc_id is None or not content:
                continue

            ref_id = f"{step_prefix}-{next_chunk_no}"
            chunk_entry = {
                "ref_id": ref_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "documentName": document_name,
                "content": content,
            }
            chunks.append(chunk_entry)
            chunk_map[ref_id] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "documentName": document_name,
                "content": content,
            }
            wrapped_chunks.append(
                self._format_chunk_for_model(
                    ref_id=ref_id,
                    doc_id=doc_id,
                    document_name=document_name,
                    content=content,
                )
            )
            documents.append(content)
            doc_ids.append(doc_id)
            next_chunk_no += 1

        wrapped_documents = "\n\n".join(wrapped_chunks)
        print("[retrieve] chunk_map:", json.dumps(chunk_map, ensure_ascii=False, indent=2))

        return {
            "documents": documents,
            "doc_ids": doc_ids,
            "chunks": chunks,
            "chunk_map": chunk_map,
            "wrapped_documents": wrapped_documents,
        }
