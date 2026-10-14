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


import os
import json
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# ==================================
# 1. Gemini Client
# ==================================

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL = "gemini-3.5-flash-lite"


# ==================================
# 2. Fake Database
# ==================================

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


# ==================================
# 3. Actual Backend Function
# ==================================

def get_payment_details(payment_id: int) -> dict:
    payment = payments.get(payment_id)

    if payment is None:
        return {"error": "PAYMENT_NOT_FOUND"}

    return {
        "payment_id": payment_id,
        **payment
    }


# ==================================
# 4. Tool Argument Validation
# ==================================

class PaymentArgs(BaseModel):
    payment_id: int = Field(strict=True, gt=0)


# ==================================
# 5. Tool Definition for Gemini
# ==================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_payment_details",
            "description": (
                "Retrieve the current status and reason "
                "for a payment using its payment ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_id": {
                        "type": "integer",
                        "description": "The payment identifier"
                    }
                },
                "required": ["payment_id"]
            }
        }
    }
]


# ==================================
# 6. Main Tool Calling Workflow
# ==================================

def answer_payment_question(question: str) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a payment support assistant. "
                "Use the payment tool when current payment "
                "information is needed. "
                "Never invent payment information. "
                "If a payment is not found, say so."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # FIRST GEMINI CALL
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # Gemini might answer without requesting a tool.
    if not assistant_message.tool_calls:
        return assistant_message.content or "No answer generated."

    # Preserve the assistant's tool-call request.
    messages.append(assistant_message)

    # Execute every tool call returned by Gemini.
    for tool_call in assistant_message.tool_calls:

        tool_name = tool_call.function.name

        if tool_name != "get_payment_details":
            raise ValueError(f"Unknown tool: {tool_name}")

        # JSON string -> validated Pydantic object
        args = PaymentArgs.model_validate_json(
            tool_call.function.arguments
        )

        # Actual Python function execution
        result = get_payment_details(
            payment_id=args.payment_id
        )

        print("Tool executed:", tool_name)
        print("Arguments:", args.model_dump())
        print("Tool result:", result)

        # Send the result back to Gemini.
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # SECOND GEMINI CALL
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return (
        final_response.choices[0].message.content
        or "No final answer generated."
    )


# ==================================
# 7. Run
# ==================================

if __name__ == "__main__":

    question = "Why was payment 9001 declined?"

    answer = answer_payment_question(question)

    print("\nFinal answer:")
    print(answer)
