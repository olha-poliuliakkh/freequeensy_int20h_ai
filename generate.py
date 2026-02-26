import json
import itertools
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini"

# Pricing per 1M tokens (USD)
PRICES = {
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":     {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":   {"input": 0.50,  "output": 1.50},
}

total_input_tokens = 0
total_output_tokens = 0
total_cost_usd = 0.0


def calc_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    price = PRICES.get(model, {"input": 0, "output": 0})
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def print_cost_summary(batch_num: int, input_tok: int, output_tok: int, batch_cost: float):
    print(
        f"[Batch {batch_num}] "
        f"input: {input_tok} tok | output: {output_tok} tok | "
        f"cost: ${batch_cost:.4f} | "
        f"total so far: ${total_cost_usd:.4f}"
    )


SYSTEM_PROMPT = """
You are a professional dataset generator.
Your goal is to create realistic support chat dialogues.
RULES:
1. Return ONLY clean JSON.
2. Format: {"chat_1": [{"role": "client", "text": "..."}, {"role": "agent", "text": "..."}, ...], "chat_2": ...}.
3. Language: English.
"""


themes = ["payment issues", "technical errors", "account access", "tariff inquiries", "refunds"]
casetypes = ["successful cases", "problematic cases", "conflict cases", "cases with agent mistakes"]
tone_errors = ["rude tone", "passive-aggressive behavior", "lack of empathy", "robotic response"]
logical_errors = ["ignored question", "incorrect info", "unnecessary escalation"]
all_mistakes = tone_errors + logical_errors
lengths = ["short", "average", "long"]
emotions = ["satisfaction", "dissatisfaction", "hidden dissatisfaction"]
completion_types = ["logically finished", "abruptly ended"]

def generate_all_scenarios():
    all_combinations = []

    for t, c, l, e, comp in itertools.product(themes, casetypes, lengths, emotions, completion_types):
        mistake_info = ""
        
        if c == "cases with agent mistakes":           
            for i in range(len(all_mistakes)):
                mistake_info = f"Specific mistake: {all_mistakes[i]}."
                scenario = (f"Theme: {t}, Case: {c}, Length: {l}, Client emotion: {e}. {mistake_info}. Conversation flow: {comp}")
                all_combinations.append(scenario)
        else:
            mistake_info = "Agent is professional."
            scenario = (f"Theme: {t}, Case: {c}, Length: {l}, Client emotion: {e}. {mistake_info}. Conversation flow: {comp}")
            all_combinations.append(scenario)
    
    return all_combinations

all_scenarios = generate_all_scenarios()

COST_LIMIT_USD = 20.0

dataset = []
for i in range(3):
    if total_cost_usd >= COST_LIMIT_USD:
        print(f"Ліміт ${COST_LIMIT_USD:.2f} досягнуто. Зупинка після батчу {i}.")
        break

    batch = all_scenarios[i*10:i*10+10]
    formatted_scenarios = "\n".join([f"{j+1}. {s}" for j, s in enumerate(batch)])

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate dialogues based on these exact scenarios:\n{formatted_scenarios}"},
            ],
        )

        usage = response.usage
        batch_input = usage.prompt_tokens
        batch_output = usage.completion_tokens
        batch_cost = calc_cost(batch_input, batch_output, OPENAI_MODEL)

        total_input_tokens += batch_input
        total_output_tokens += batch_output
        total_cost_usd += batch_cost

        print_cost_summary(i + 1, batch_input, batch_output, batch_cost)

        chat_data = json.loads(response.choices[0].message.content)
        dataset.append(chat_data)

    except Exception as e:
        print(f"Помилка: {e}")

print(
    f"Input tokens : {total_input_tokens}\n"
    f"Output tokens : {total_output_tokens}\n"
    f"Totally cost : ${total_cost_usd:.4f}"
)

with open("generate.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)