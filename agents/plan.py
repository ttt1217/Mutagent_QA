import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from src.utils import GraphState
from src.prompt_template import planing_system_message, planing_human_message

load_dotenv()

def plan_agent(state: GraphState):
    API_KEY = os.getenv("API_KEY") or "ailab_XfFgDfq4a26+HiFjKDzuOY0A2Fx7xjSyuJQDShHbACxhoEcJ0rxpaPFKYeQ9rC58DIKPN7tryyVzzkvlkeC3qrkD4ibjJFVpjzLrwoNkUANKbJ8bcedxrgA="
    BASE_URL = os.getenv("BASE_URL") or "https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1"
    MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3-235B-A22B"

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    original_question = state["original_question"]

    #  memory
    all_mem = []
    for past_exp in state.get("past_exp", []):
        memory = ""
        plan = ', '.join(past_exp.get("plan", []))
        summary = past_exp.get("plan_summary", {}).get("output", "")
        score = past_exp.get("plan_summary", {}).get("score", "")
        memory += f"Plan: [{plan}]\nStatus: {summary}\nScore: {score}\n"
        all_mem.append(memory)
    memory = "\n".join([f"Trial {i}:\n{m}" for i, m in enumerate(all_mem)]) if all_mem else "empty"

    # Prompt
    system_prompt = planing_system_message
    user_prompt = planing_human_message.format(question=original_question, memory=memory)

    #  LLM
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            timeout=60,
        )

        reply = resp.choices[0].message.content.strip()
        #print("\n[plan_agent] LLM 原始输出:\n", reply)
        steps = []

        # 优先解析 JSON
        try:
            parsed = json.loads(reply)
            if isinstance(parsed, dict) and "steps" in parsed:
                steps = parsed["steps"]
            elif isinstance(parsed, list):
                steps = parsed
            elif isinstance(parsed, dict) and "plan" in parsed:
                steps = parsed["plan"]
        except Exception:

            matches = re.findall(r"(?:^|\n)[\-\*\d\.]+\s*(.+)", reply)
            if matches:
                steps = [s.strip() for s in matches]
            else:

                raw_lines = [l.strip() for l in reply.splitlines() if l.strip()]
                for l in raw_lines:
                    if l.lower().startswith(("step", "-", "*")):
                        cleaned = re.sub(r"^[\-\*\d\. ]+", "", l)
                        steps.append(cleaned.strip())
                if not steps:
                    steps = [reply]

        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")

        return {"plan": steps}

    except Exception as e:
        print(f"[plan_agent]  LLM 调用失败: {str(e)}")
        return {"plan": ["__LLM_CALL_FAILED__"]}
