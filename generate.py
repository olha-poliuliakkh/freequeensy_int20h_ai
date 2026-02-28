import json
import itertools

from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
import os
import threading

load_dotenv()

PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

COST_LIMIT_USD = 20.0

total_input_tokens = 0
total_output_tokens = 0
total_cost_usd = 0.0
_lock = threading.Lock()


MAX_RETRIES = 3


def calc_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    price = PRICES.get(model, {"input": 0, "output": 0})
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def _is_valid_chat(chat) -> bool:
    if not chat.messages:
        return False
    for msg in chat.messages:
        if msg.role not in ("client", "agent"):
            return False
        if not msg.text.strip():
            return False
    return True


class ChatMessage(BaseModel):
    role: str = Field(description="Role of the speaker: 'client' or 'agent'")
    text: str = Field(description="The content of the message")

class SingleChat(BaseModel):
    chat_id: int = Field(description="The sequential number of the chat (e.g., 1, 2, 3)")
    messages: List[ChatMessage] = Field(description="List of messages in this conversation")

class ChatDataset(BaseModel):
    chats: List[SingleChat] = Field(description="A list of generated conversations")


class ChatGenerator:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = instructor.from_openai(OpenAI(api_key=api_key))
        self.model = model
        self.system_prompt = ("You are a professional generator of realistic support chats in English")

    def _call_api(self, scenario: str, num_chats: int) -> tuple:
        """Single API call; returns (ChatDataset, completion)."""
        prompt = f"Generate {num_chats} diverse chat conversations based on the following scenario: {scenario}"
        return self.client.chat.completions.create_with_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            response_model=ChatDataset,
            temperature=0,
            seed=27,
            top_p=0.9,
            presence_penalty=0.3,
            frequency_penalty=0.4,
        )

    def _track_usage(self, completion) -> None:
        global total_input_tokens, total_output_tokens, total_cost_usd
        usage = completion.usage
        in_tok = usage.prompt_tokens
        out_tok = usage.completion_tokens
        cost = calc_cost(in_tok, out_tok, self.model)
        with _lock:
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_cost_usd += cost
        print(f"  input: {in_tok} tok | output: {out_tok} tok | cost: ${cost:.4f} | total: ${total_cost_usd:.4f}")

    def generate(self, scenario: str, num_chats: int = 2) -> ChatDataset:
        collected = []

        for attempt in range(1, MAX_RETRIES + 1):
            needed = num_chats - len(collected)
            result, completion = self._call_api(scenario, needed)
            self._track_usage(completion)

            valid = [c for c in result.chats if _is_valid_chat(c)]
            discarded = len(result.chats) - len(valid)
            if discarded:
                print(f"  [attempt {attempt}] discarded {discarded} invalid chat(s), kept {len(valid)}")

            collected.extend(valid)

            if len(collected) >= num_chats:
                break

            if attempt < MAX_RETRIES:
                print(f"  [attempt {attempt}] have {len(collected)}/{num_chats}, retrying for {needed - len(valid)} more...")

        if len(collected) < num_chats:
            print(f"  Warning: only {len(collected)}/{num_chats} valid chats after {MAX_RETRIES} attempts")

        final = collected[:num_chats]
        for i, chat in enumerate(final, start=1):
            chat.chat_id = i

        return ChatDataset(chats=final)


themes = ["payment issues", "technical errors", "account access", "tariff inquiries", "refunds"]
casetypes = ["successful", "problematic", "conflict", "inappropriate support agent communication style", "support agent reasoning error"]
duration = ["short", "average", "long"]
emotions = ["satisfaction", "dissatisfaction", "hidden dissatisfaction"]
completion_types = ["logically finished", "abruptly ended"]

def generate_all_scenarios():
    all_combinations = []

    for t, c, l, e, comp in itertools.product(themes, casetypes, duration, emotions, completion_types):
        if (c=="successful" and e in ["dissatisfaction","hidden dissatisfaction"]):
            continue
        
        scenario = (f"Theme: {t}, Case: {c}, Duration: {l}, Client emotion: {e}, Conversation flow: {comp}")
        all_combinations.append(scenario)
    
    return all_combinations

all_scenarios = generate_all_scenarios()


if __name__ == "__main__":
    generator = ChatGenerator(api_key=os.environ.get("OPENAI_API_KEY"))  # noqa

    dataset = []
    for i, scen in enumerate(all_scenarios):
        if total_cost_usd >= COST_LIMIT_USD:
            print(f"Limit ${COST_LIMIT_USD:.2f} reached. Stopped at scenario {i}.")
            break

        print(f"Generating...  {i}/{len(all_scenarios)}: {scen}")

        try:
            result = generator.generate(scen, num_chats=1)
            chat_entry = {
                "scenario_index": i,
                "scenario_description": scen,
                "generated_data": result.model_dump()
            }
            dataset.append(chat_entry)

        except Exception as err:
            print(f"Error with {i}: {err}")

    output_file = "generated_chats_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(
        f"\nCompleted! Chats are in {output_file}\n"
        f"Input tokens:  {total_input_tokens}\n"
        f"Output tokens: {total_output_tokens}\n"
        f"Total cost:    ${total_cost_usd:.4f}"
    )