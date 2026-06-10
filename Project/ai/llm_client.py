import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def query_llm(prompt: str, system_prompt: str = None) -> str:
    try:
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"ERROR: {str(e)}"
