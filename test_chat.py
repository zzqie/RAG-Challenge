import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    # 1. 加载环境变量
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 未找到 OPENAI_API_KEY，请检查 .env 文件")
        return

    print("✅ API Key 已读取")
    print("正在连接 ChatAnywhere...\n")

    try:
        # 2. 初始化客户端（关键：自定义 base_url）
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.chatanywhere.tech"
        )

        # 3. 发送测试请求
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # ChatAnywhere通常支持
            messages=[
                {"role": "user", "content": "请回复：ChatAnywhere API 测试成功"}
            ],
            temperature=0
        )

        # 4. 输出结果
        answer = response.choices[0].message.content

        print("模型回复：")
        print(answer)
        print("\n🎉 ChatAnywhere API 正常，可用！")

    except Exception as e:
        print("\n❌ API 调用失败")
        print("错误信息：")
        print(e)


if __name__ == "__main__":
    main()
