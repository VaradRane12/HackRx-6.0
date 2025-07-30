from langchain_openai import ChatOpenAI  # ✅ use new import

def get_llm():
    return ChatOpenAI(
        model_name="gpt-4",  # or "gpt-3.5-turbo" for cheaper testing
        temperature=0,
    )
