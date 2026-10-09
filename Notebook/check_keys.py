"""Checks the keys in keys.py with one minimal request each; never prints a key.

    python check_keys.py
"""
from jif.config import load_keys

keys = load_keys()

print("大模型")
if not keys["LLM_API_KEY"]:
    print("  未填写 LLM_API_KEY")
else:
    from openai import OpenAI
    client = OpenAI(base_url=keys["LLM_BASE_URL"], api_key=keys["LLM_API_KEY"], timeout=30)
    print(f"  地址：{keys['LLM_BASE_URL']}")
    try:
        models = [m.id for m in client.models.list().data]
        print(f"  可用模型：{', '.join(models[:20])}{' …' if len(models) > 20 else ''}")
    except Exception as e:
        models = []
        print(f"  列出模型失败（有些服务不支持，可忽略）：{type(e).__name__}")
    if not keys["LLM_MODEL"]:
        print("  未填写 LLM_MODEL，请从上面的列表里选一个填进 keys.py")
    else:
        try:
            r = client.chat.completions.create(model=keys["LLM_MODEL"], max_tokens=5,
                                               messages=[{"role": "user", "content": "回复“好”"}])
            print(f"  {keys['LLM_MODEL']} 调用成功：{r.choices[0].message.content!r}")
        except Exception as e:
            print(f"  {keys['LLM_MODEL']} 调用失败：{type(e).__name__}: {str(e)[:200]}")

print("Jev")
if not keys["TYPESAFE_API_KEY"]:
    print("  未填写 TYPESAFE_API_KEY")
else:
    try:
        from typesafe_sdk import Noul, TypeSafeClient
        r = TypeSafeClient(model=keys["JEV_MODEL"]).system_one("今天天气很好", {"q": Noul(instructions="这句话是在说天气吗？")})
        print(f"  {r.model} 调用成功：noul={r.nouls['q'].noul:.2f}")
    except Exception as e:
        print(f"  调用失败：{type(e).__name__}: {str(e)[:200]}")
