from openai import OpenAI

client = OpenAI(
    api_key="ailab_XfFgDfq4a26+HiFjKDzuOY0A2Fx7xjSyuJQDShHbACxhoEcJ0rxpaPFKYeQ9rC58DIKPN7tryyVzzkvlkeC3qrkD4ibjJFVpjzLrwoNkUANKbJ8bcedxrgA=",
    base_url="https://dev.iwhalecloud.com/faas/serverless/gpt-api-gw/v1"
)

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好，测试一下模型是否可用"}
    ]
)
print(resp.choices[0].message.content)
