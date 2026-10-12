import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal


# -------------------------
# 1. Setup
# -------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# -------------------------
# 2. Our data contract
# -------------------------

class TicketClassification(BaseModel):
    category: Literal["PAYMENT", "ACCOUNT", "CARD", "OTHER"]
    priority: Literal["LOW", "MEDIUM", "HIGH"]
    needs_account_data: bool
    summary: str


# -------------------------
# 3. System instruction
# -------------------------

SYSTEM_PROMPT = """
You are a customer support ticket classifier.

Rules:

- category must be PAYMENT, ACCOUNT, CARD, or OTHER
- priority must be LOW, MEDIUM, or HIGH
- needs_account_data must be true only when answering the specific
  customer issue requires current/private account or transaction data
- summary must be short and factual
"""


# -------------------------
# 4. Main classifier
# -------------------------

def classify_ticket(message: str) -> TicketClassification:

    schema = TicketClassification.model_json_schema()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": message
        }
    ]

    MAX_ATTEMPTS = 2

    last_error = None

    for attempt in range(MAX_ATTEMPTS):

        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            messages=messages,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ticket_classification",
                    "schema": schema
                }
            }
        )

        content = response.choices[0].message.content

        try:
            # Step 1: Parsing
            data = json.loads(content)

            # Step 2: Validation
            ticket = TicketClassification.model_validate(data)

            return ticket

        except (json.JSONDecodeError, ValidationError) as e:

            last_error = e

            # no attempts left
            if attempt == MAX_ATTEMPTS - 1:
                break

            # Tell model what went wrong
            messages.append({
                "role": "assistant",
                "content": content
            })

            messages.append({
                "role": "user",
                "content": f"""
Your previous response was invalid.

Validation/parsing error:
{str(e)}

Correct the previous response.

Follow the required output schema exactly.
"""
            })

    raise RuntimeError(
        "LLM failed to produce a valid ticket classification."
    ) from last_error

# ticket = classify_ticket(
#     "My card payment was declined yesterday."
# )
# print(ticket)
# print(ticket.category)
# print(ticket.priority)
# print(ticket.needs_account_data)
# print(ticket.summary)

#our fake database

payments = {
    9001: {
        "status": "DECLINED",
        "reason": "INSUFFICIENT_FUNDS"
    },

    9002: {
        "status": "COMPLETED",
        "reason": None
    }
}

def get_payment_details(payment_id: int):
    return payments.get(payment_id)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_payment_details",

            "description": "Get the current details of a payment by its payment ID.",

            "parameters": {
                "type": "object",

                "properties": {
                    "payment_id": {
                        "type": "integer",
                        "description": "The ID of the payment."
                    }
                },

                "required": ["payment_id"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a payment support assistant. "
                "Use the available tool when payment details are needed. "
                "Do not invent payment information."
            )
        },
        {
            "role": "user",
            "content": "Why was payment 9001 declined?"
        }
    ],
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

print("Content:", message.content)
print("Tool calls:", message.tool_calls)