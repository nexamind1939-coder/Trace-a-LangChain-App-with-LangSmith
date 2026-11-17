
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import time

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompts = [
    "Explain LangChain in one simple sentence.",
    "What is LangSmith? Explain its benefits for beginners.",
    """Explain how LangChain, OpenAI and LangSmith work together.
    Include their purpose, how they connect, a business example,
    and three benefits for AI developers."""
]

for i, prompt in enumerate(prompts, start=1):

    print(f"\n========== RUN {i} ==========")
    print("Prompt:", prompt)

    start_time = time.time()

    response = llm.invoke(prompt)

    end_time = time.time()
    execution_time = end_time - start_time

    print("\nAnswer:")
    print(response.content)

    # Token usage
    usage = response.usage_metadata or {}

    print("\n--- PERFORMANCE DETAILS ---")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Input Tokens: {usage.get('input_tokens', 'N/A')}")
    print(f"Output Tokens: {usage.get('output_tokens', 'N/A')}")
    print(f"Total Tokens: {usage.get('total_tokens', 'N/A')}")
