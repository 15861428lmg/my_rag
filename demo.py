from openai import OpenAI
from config import (
    MINIMAX_API_KEY,
    MINIMAX_API_BASE,
    MINIMAX_MODEL,
    QWEN_API_KEY,
    QWEN_API_BASE,
    QWEN_EMBEDDING_MODEL
)


def test_embedding():
    print("=== 测试千问 Embedding ===")
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)

    try:
        response = client.embeddings.create(
            model=QWEN_EMBEDDING_MODEL,
            input="你好，这是一个测试"
        )
        print(f"Embedding 成功! 维度: {len(response.data[0].embedding)}")
        return True
    except Exception as e:
        print(f"Embedding 失败: {e}")
        return False


def test_chat():
    print("\n=== 测试 MiniMax Chat ===")
    client = OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_API_BASE)

    try:
        response = client.chat.completions.create(
            model=MINIMAX_MODEL,
            messages=[{"role": "user", "content": "你好，请回复'测试成功'"}],
            extra_body={"reasoning_split": True},
        )
        print(f"Chat 成功! 回复: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Chat 失败: {e}")
        return False


if __name__ == "__main__":
    emb_ok = test_embedding()
    chat_ok = test_chat()

    print("\n=== 结果汇总 ===")
    print(f"千问 Embedding: {'✓ 通过' if emb_ok else '✗ 失败'}")
    print(f"MiniMax Chat: {'✓ 通过' if chat_ok else '✗ 失败'}")