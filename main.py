import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


message = "I love my debit card"

response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=[
       {
        "role": "system",
        "content": """
    You are a customer support ticket classifier.

    For every customer message determine:
    - category
    - priority
    - whether account data is needed
    - a short summary
    """
    },
        {
            "role": "user",
            "content": message
        }
    ]
)

result = response.choices[0].message.content

print(response.choices[0].message.content)