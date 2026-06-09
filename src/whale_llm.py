# src/whale_llm.py
import os
from openai import OpenAI

class WhaleLLM:
    """兼容 LangChain 风格的简易 LLM 封装（使用阿里鲸云 OpenAI 接口）"""
    def __init__(self, model_name=None, temperature=0.3, api_key=None, base_url=None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "Qwen/Qwen3-235B-A22B")
        self.temperature = temperature
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv(
            "BASE_URL", "https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1"
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # def invoke(self, messages):
    #     """兼容 chain.invoke() 调用"""
    #     if isinstance(messages, list):
    #         msgs = messages
    #     elif isinstance(messages, dict):
    #         msgs = [
    #             {"role": "system", "content": messages.get("system", "")},
    #             {"role": "user", "content": messages.get("question", "")}
    #         ]
    #     else:
    #         msgs = [{"role": "user", "content": str(messages)}]
    #
    #     allowed_roles = {"system", "user", "assistant", "function"}
    #     for m in msgs:
    #         if m.get("role") not in allowed_roles:
    #             print(f"[WhaleLLM] 替换非法 role={m.get('role')} → 'user'")
    #             m["role"] = "user"
    #
    #     resp = self.client.chat.completions.create(
    #         model=self.model_name,
    #         messages=msgs,
    #         temperature=self.temperature,
    #     )
    #     return resp.choices[0].message.content

    def invoke(self, messages):
        import time
        start_time = time.time()

        # === 统一 messages 格式 ===
        if isinstance(messages, list):
            msgs = []
            for m in messages:
                # 如果是 LangChain 的 Message 对象
                if hasattr(m, "role") and hasattr(m, "content"):
                    msgs.append({"role": m.role, "content": m.content})
                # 如果是 dict
                elif isinstance(m, dict):
                    msgs.append({
                        "role": m.get("role", "user"),
                        "content": m.get("content", str(m))
                    })
                else:
                    msgs.append({"role": "user", "content": str(m)})
        elif isinstance(messages, dict):
            msgs = [
                {"role": "system", "content": messages.get("system", "")},
                {"role": "user", "content": messages.get("question", "")}
            ]
        else:
            msgs = [{"role": "user", "content": str(messages)}]

        # === 过滤非法角色 ===
        allowed_roles = {"system", "user", "assistant", "function"}
        for m in msgs:
            role = m.get("role", "user")
            if role not in allowed_roles:
                print(f"[WhaleLLM] 检测到非法角色 role='{role}'，自动替换为 'user'")
                m["role"] = "user"

        # === 调用模型 ===
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            temperature=self.temperature,
        )

        end_time = time.time()
        print(f"[WhaleLLM] 模型调用耗时 {end_time - start_time:.2f}s")
        return resp.choices[0].message.content

    def chat(self, user_prompt, system_prompt=""):
        """简单模式"""
        return self.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
