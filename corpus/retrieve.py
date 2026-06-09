import faiss 
import logging
import pickle
import glob
import os
import numpy as np

from accelerate.logging import get_logger

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, gpu_ids=None):
        logger.info("Initializing retriever")
        self.doc_lookup = []
        self.gpu_ids = gpu_ids or []  # 支持不传 GPU
        self.index = None
        self.index_on_gpu = False
        self.embeddings = None
        self.doc_ids = None
        self.texts = None

    def _initialize_faiss_index(self, dim: int):
        """初始化 CPU FAISS index"""
        self.index = faiss.IndexFlatL2(dim)

    def _move_index_to_gpu(self):
        if not self.gpu_ids:
            logger.info("No GPUs specified, keep index on CPU")
            return
        ngpus = len(self.gpu_ids)
        total = faiss.get_num_gpus()
        assert ngpus <= total, "Number of gpus must <= total gpus"
        logger.info(f"Moving index to GPU(s) {self.gpu_ids}, total GPUs: {total}")
        gpu_resources = [faiss.StandardGpuResources() for _ in range(total)]
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(ngpus):
            vdev.push_back(i)
            vres.push_back(gpu_resources[self.gpu_ids[i]])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)
        self.index_on_gpu = True

    def load_emb_data(self, emb_data):
        """
        emb_data expected:
          {
            "doc_ids": [...],
            "embeddings": np.ndarray (N, dim),
            "texts": [...]
          }
        This will normalize doc_ids to Python native types (int or str).
        """
        self.embeddings = emb_data["embeddings"].astype("float32")
        # normalize doc_ids -> python native list (prefer int when possible)
        raw_doc_ids = list(emb_data["doc_ids"])
        norm_ids = []
        for x in raw_doc_ids:
            try:
                # prefer int if it is integer-like
                if isinstance(x, (np.integer,)) or (isinstance(x, (str,)) and str(x).isdigit()):
                    norm_ids.append(int(x))
                else:
                    norm_ids.append(x)
            except Exception:
                norm_ids.append(x)
        self.doc_ids = norm_ids
        self.texts = emb_data.get("texts", [""] * len(self.doc_ids))
        dim = self.embeddings.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(self.embeddings)
        print(f"✅ 已加载 {len(self.doc_ids)} 条文档嵌入 (dim={dim})")
        self._move_index_to_gpu()

    def search(self, query_ids, query_embs, top_k):
        """
        返回格式:
            { query_id: { doc_id: score, ... }, ... }
        doc_id 使用 self.doc_ids 对应的原始 id（Python int 或 str）
        """
        if self.index is None:
            raise ValueError("Faiss index is not initialized!")

        D, I = self.index.search(query_embs, top_k)  # I shape: (nq, top_k)
        # map positions (0..N-1) -> original doc ids
        if self.doc_ids is None:
            mapped = I  # fallback: use positions
        else:
            docid_arr = np.array(self.doc_ids, dtype=object)  # keep as object to preserve types
            mapped = docid_arr[I]  # shape (nq, top_k)

        return_dict = {}
        for q_idx, (scores_per_q, doc_positions_per_q) in enumerate(zip(D, mapped)):
            qid = query_ids[q_idx]
            # normalize qid to python int if numeric numpy scalar
            try:
                if isinstance(qid, (np.integer,)):
                    qid_key = int(qid)
                else:
                    qid_key = qid
            except Exception:
                qid_key = qid

            return_dict[qid_key] = {}
            for doc_id, score in zip(doc_positions_per_q, scores_per_q):
                # doc_id may be numpy scalar; convert to python type
                try:
                    if isinstance(doc_id, (np.integer,)):
                        doc_id_key = int(doc_id)
                    else:
                        doc_id_key = doc_id
                except Exception:
                    doc_id_key = doc_id
                return_dict[qid_key][doc_id_key] = float(score)

        # debug: print brief summary
        logger.debug(
            f"Retriever.search returned keys: {list(return_dict.keys())} ; sample: { {k: list(v.items())[:3] for k, v in list(return_dict.items())[:1]} }")
        return return_dict
    # def load_emb_data(self, emb_data):
    #     self.embeddings = emb_data["embeddings"].astype("float32")
    #     self.doc_ids = emb_data["doc_ids"]
    #     self.texts = emb_data["texts"]
    #     dim = self.embeddings.shape[1]
    #     self._initialize_faiss_index(dim)
    #     self.index.add(self.embeddings)
    #     print(f"✅ 已加载 {len(self.doc_ids)} 条文档嵌入 (dim={dim})")
    #     self._move_index_to_gpu()  # 仅在有 GPU 时移动
    #
    # def search(self, query_ids, query_embs, top_k):
    #     """
    #     返回格式: { query_id: { doc_id: score, ... }, ... }
    #     doc_id 使用与 self.doc_ids 中相同的值（通常为 Python int 或字符串）
    #     """
    #     if self.index is None:
    #         raise ValueError("Faiss index is not initialized!")
    #
    #     D, I = self.index.search(query_embs, top_k)  # I shape: (nq, top_k)
    #     # I contains positions (0..N-1). map them to original doc_ids if available.
    #     # self.doc_ids 应该是与 embeddings 行对齐的数组/list
    #     if self.doc_ids is None:
    #         # 如果没有 doc_ids，直接返回索引（作为 int）
    #         mapped = I
    #     else:
    #         # self.doc_ids 可能是 numpy array，确保转为 list 且用 int/str 一致性
    #         docid_arr = np.array(self.doc_ids)
    #         # docid_arr[ I ] -> shape (nq, top_k) of original doc ids
    #         mapped = docid_arr[I]
    #
    #     return_dict = {}
    #     for q_idx, (scores_per_q, doc_positions_per_q) in enumerate(zip(D, mapped)):
    #         qid = query_ids[q_idx]
    #         return_dict[qid] = {}
    #         for doc_id, score in zip(doc_positions_per_q, scores_per_q):
    #             # 确保 key 是 Python 原始类型（int 或 str），这里使用 int 如果可能
    #             try:
    #                 # 如果是 numpy scalar，转为 Python int
    #                 doc_id_key = int(doc_id)
    #             except Exception:
    #                 doc_id_key = doc_id
    #             return_dict[qid][doc_id_key] = float(score)
    #
    #     return return_dict


if __name__ == "__main__":
    retrieve = Retriever(gpu_ids=[])  # CPU 模式
    os.makedirs("save_embs", exist_ok=True)

    # 测试用embeddings dict
    emb_data = {
        "embeddings": np.random.rand(5, 1024).astype("float32"),
        "doc_ids": [0, 1, 2, 3, 4],
        "texts": ["doc0", "doc1", "doc2", "doc3", "doc4"]
    }

    retrieve.load_emb_data(emb_data)
    query_embs = np.random.rand(1, 1024).astype("float32")
    result = retrieve.search([0], query_embs, top_k=3)
    print(result)
