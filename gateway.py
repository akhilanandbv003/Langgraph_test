from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()


def llm_gateway(
    messages: list,
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    if provider == "anthropic":
        llm = ChatAnthropic(
            model=model or "claude-haiku-4-5-20251001",
            temperature=temperature,
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider!r}. Choose 'anthropic' or 'openai'.")

    return llm.invoke(messages).content
