import os
import json
import time
import pickle
import logging
import numpy as np
import requests
import torch
import pdfplumber
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
import torch.nn.functional as F


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")

def build_local_corpus(pdf_dir: str = "./pdf_docs", chunk_size: int = 500, overlap: int = 100):
    ids, docs = [], []

    if not os.path.exists(pdf_dir):
        print(f" 目录 {pdf_dir} 不存在，自动创建。")
        os.makedirs(pdf_dir, exist_ok=True)
        return ids, docs

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f" 未在 {pdf_dir} 中找到 PDF 文件。")
        return ids, docs

    print(f" 在 {pdf_dir} 中找到 {len(pdf_files)} 个 PDF 文件，开始解析...")

    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]

        try:
            # 提取全文文本
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            text = text.strip()
            if not text:
                print(f"️ 文件 {pdf_file} 无法提取文本，跳过。")
                continue

            # chunk
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start += chunk_size - overlap

            for i, chunk in enumerate(chunks):
                chunk_id = f"{base_name}_{i}"
                ids.append(chunk_id)
                docs.append(chunk)

            print(f" 解析完成：{pdf_file} -> {len(chunks)} 个 chunk。")

        except Exception as e:
            print(f" 解析文件 {pdf_file} 时出错: {e}")

    print(f" 共加载 {len(docs)} 个 chunk。")
    return ids, docs


# ========== 模拟本地小型语料 ==========
# def build_local_corpus():
#     docs = [
#         "人工智能是一门研究如何让计算机模仿人类智能的科学。",
#         "自然语言处理是人工智能的重要分支，用于理解和生成人类语言。",
#         "向量检索通过语义相似度快速查找相关文档。",
#         "机器学习使用数据和算法自动改进模型性能。",
#         "硅基流动提供了高性能的企业级大模型服务。"
#     ]
#     ids = list(range(len(docs)))
#     return ids, docs


# ========== 嵌入API调用 ==========
def get_embeddings_from_api(text_list, api_key, model="BAAI/bge-m3"):
    url = "https://lab.iwhalecloud.com/gpt-proxy/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }

    payload = {"model": model, "input": text_list}
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

    if response.status_code != 200:
        logger.error(f"API 调用失败: {response.status_code} - {response.text}")
        raise RuntimeError(response.text)

    data = response.json()
    embs = [item["embedding"] for item in data["data"]]
    return np.array(embs, dtype=np.float32)


# ========== 主流程 ==========
def main():
    accelerator = Accelerator()
    API_KEY = "ailab_zqeUaTfVYu3cbR44jTE73Sw9doTam4YgGX/9uzRQEAs7PA8hKEYJ2N77GpZbOkJPFBhxcVpEez317gwuK8D29Ssa99p4GcX2ZN71jnMGi1AlUC3+UXAFIuc="

    ids, docs = build_local_corpus()
    logger.info(f"共加载 {len(docs)} 条文档")

    model_name = "BAAI/bge-m3"
    output_dir = "emb_corpus/demo"
    os.makedirs(output_dir, exist_ok=True)
    dimension = 1024

    # === 生成嵌入 ===
    embeddings = get_embeddings_from_api(docs, API_KEY, model=model_name)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # === 保存 ===
    all_ids = np.array(ids)
    all_embs = embeddings.cpu().numpy()

    with open(f"{output_dir}/demo_emb.pkl", "wb") as f:
        pickle.dump((all_ids, all_embs), f)

    logger.info(f" 嵌入生成完成！保存到 {output_dir}/demo_emb.pkl")


if __name__ == "__main__":
    main()
