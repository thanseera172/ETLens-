import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def query_llm(prompt: str, system_prompt: str = None) -> str:
    """
    Send a prompt to Ollama and return the text response.
    
    Args:
        prompt: The user's question or instruction
        system_prompt: Optional system-level instruction to guide LLM behavior
    
    Returns:
        The LLM's response as a plain string
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=600  # LLMs can be slow, give it 2 minutes
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure it's running with: ollama serve"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama took too long to respond. Try a shorter prompt."
    except requests.exceptions.HTTPError as e:
        return f"ERROR: Ollama returned an error — {str(e)}"
    except KeyError:
        return f"ERROR: Unexpected response format from Ollama: {data}"


if __name__ == "__main__":
    print("Testing Ollama LLM connection...\n")

    # Test 1: Basic connectivity
    response = query_llm("Say 'Hello from Ollama!' and nothing else.")
    print(f"Test 1 - Basic response:\n{response}\n")

    # Test 2: With a system prompt (how we'll use it in doc generation)
    response2 = query_llm(
        prompt="This script reads a CSV, filters rows where amount > 1000, and saves to SQLite.",
        system_prompt="You are a technical documentation expert. Explain ETL scripts in simple, clear English."
    )
    print(f"Test 2 - With system prompt:\n{response2}\n")

    print("✅ LLM client test complete!")