import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import OpenAI


# =====================================
# Configuration
# =====================================

BASE_URL = os.getenv("AGICTO_BASE_URL", "https://api.agicto.cn/v1")
API_KEY = os.getenv("AGICTO_API_KEY")

MODEL_NAME = "gpt-4o-mini"


# =====================================
# Receive query1 question from main script
# =====================================

QUESTION = sys.stdin.read().strip()

if not QUESTION:
    raise ValueError("No query1 question received from main script.")


# =====================================
# Call GPT
# =====================================

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a scientific assistant. "
                "Answer briefly and directly. "
                "Prefer one concise scientific variable or phrase."
            ),
        },
        {
            "role": "user",
            "content": QUESTION,
        },
    ],
)

answer = response.choices[0].message.content.strip()

print("Question:")
print(QUESTION)

print("\nAnswer:")
print(answer)