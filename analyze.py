import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini"

PRICES = {
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},
    "gpt-4.1-mini":   {"input": 0.40, "output": 1.60},
}

COST_LIMIT_USD = 20.0

total_input_tokens = 0
total_output_tokens = 0
total_cost_usd = 0.0


def calc_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    price = PRICES.get(model, {"input": 0, "output": 0})
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


ANALYSIS_PROMPT = """
You are a quality assurance analyst for a customer support team.
Analyze the support chat dialogue and return a JSON object with exactly these fields:

- intent: client's main request category. Choose one from:
  "payment issues", "technical errors", "account access", "tariff inquiries", "refunds", "other"
- satisfaction: client's satisfaction level. Choose one from:
  "satisfied", "neutral", "unsatisfied"
- quality_score: integer 1–5 rating the agent's performance (1 = very poor, 5 = excellent)
- agent_mistakes: list of mistakes made by the agent. Choose any from:
  "ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation"
  Use an empty list [] if no mistakes were made.

Return ONLY a valid JSON object, no extra text.
"""


def format_dialog(messages: list) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['text']}" for m in messages)


# Load dataset
with open("generate.json", "r", encoding="utf-8") as f:
    raw_dataset = json.load(f)

all_chats = {}
for batch in raw_dataset:
    all_chats.update(batch)

print(f"Uploaded dialogs: {len(all_chats)}\n")

results = {}

for chat_id, messages in all_chats.items():
    if total_cost_usd >= COST_LIMIT_USD:
        print(f"\nLimit ${COST_LIMIT_USD:.2f} is reached. Program was interrupted on {chat_id}.")
        break

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": f"Dialogue:\n{format_dialog(messages)}"},
            ],
        )

        usage = response.usage
        in_tok = usage.prompt_tokens
        out_tok = usage.completion_tokens
        cost = calc_cost(in_tok, out_tok, OPENAI_MODEL)

        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_cost_usd += cost

        analysis = json.loads(response.choices[0].message.content)
        results[chat_id] = analysis

        print(
            f"[{chat_id}] "
            f"intent={analysis.get('intent')} | "
            f"satisfaction={analysis.get('satisfaction')} | "
            f"score={analysis.get('quality_score')} | "
            f"mistakes={analysis.get('agent_mistakes')} | "
            f"total: ${total_cost_usd:.4f}"
        )

    except Exception as e:
        print(f"Error with {chat_id}: {e}")

print(
    f"Number of analysed dialogs : {len(results)}\n"
    f"Input tokens : {total_input_tokens}\n"
    f"Output tokens : {total_output_tokens}\n"
    f"Totally cost : ${total_cost_usd:.4f}"
)

with open("analysis.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
