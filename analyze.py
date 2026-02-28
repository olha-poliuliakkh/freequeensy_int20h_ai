import json
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List, Dict, Literal

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


def calc_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    price = PRICES.get(model, {"input": 0, "output": 0})
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


class ChatAnalysis(BaseModel):
    intent: Literal[
       "payment issues", "technical errors", "account access", "tariff inquiries",
         "refunds", "other"] = Field(description="The primary category of the request"
    )
    
    satisfaction: Literal["satisfied", "neutral", "unsatisfied"] = Field(
        description="Customer emotion level based on the conversation"
    )
    
    quality_score: int = Field(
        ge=1, le=5, 
        description="Rating of the support agent's performance from 1 to 5"
    )
    
    agent_mistakes: List[str] = Field(description="List of specific mistakes made by the agent. " \
        "For example, ignored_question, incorrect_info, rude_tone, " \
        "no_resolution or unnecessary_escalation")

    summary_reasoning: str = Field(description="Brief explanation of why these scores were given")


class ChatAnalyzer:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = instructor.from_openai(OpenAI(api_key=api_key))
        self.model = model
        self.system_prompt = (
            "You are an expert Quality Assurance specialist in customer support. "
            "Analyze the provided dialogue and provide a structured evaluation."
        )

    def analyze_chat(self, chat_messages: List[Dict]) -> ChatAnalysis:
        global total_input_tokens, total_output_tokens, total_cost_usd

        formatted_chat = "\n".join([f"{m['role']}: {m['text']}" for m in chat_messages])

        result, completion = self.client.chat.completions.create_with_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Analyze this conversation: {formatted_chat}"}
            ],
            response_model=ChatAnalysis,
            temperature=0,
            seed=27
        )

        usage = completion.usage
        in_tok = usage.prompt_tokens
        out_tok = usage.completion_tokens
        cost = calc_cost(in_tok, out_tok, self.model)

        with _lock:
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_cost_usd += cost

        print(f"  input: {in_tok} tok | output: {out_tok} tok | cost: ${cost:.4f} | total: ${total_cost_usd:.4f}")

        return result


if __name__ == "__main__":
    analyzer = ChatAnalyzer(api_key=os.environ.get("OPENAI_API_KEY"))

    with open("generated_chats_dataset.json", "r", encoding="utf-8") as f:
        chats_to_analyze = json.load(f)

    analyzed_dataset = []
    for entry in chats_to_analyze:
        if total_cost_usd >= COST_LIMIT_USD:
            print(f"Limit ${COST_LIMIT_USD:.2f} reached. Stopped at scenario '{entry['scenario_description']}'.")
            break

        print(f"Chat scenario for analysis: {entry['scenario_description']}")
        for i, chat in enumerate(entry['generated_data']['chats']):
            if total_cost_usd >= COST_LIMIT_USD:
                print(f"Limit ${COST_LIMIT_USD:.2f} reached. Stopped at chat {i + 1}.")
                break

            analysis_result = analyzer.analyze_chat(chat['messages'])
            chat_analysis = {
                "scenario_index": entry['scenario_index'],
                "scenario": entry['scenario_description'],
                "chat_id": i + 1,
                "analysis": analysis_result.model_dump()
            }
            analyzed_dataset.append(chat_analysis)

    output_file = "analyzed_support_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analyzed_dataset, f, indent=4, ensure_ascii=False)

    print(
        f"\nAnalysis is completed! Results in {output_file}\n"
        f"Input tokens:  {total_input_tokens}\n"
        f"Output tokens: {total_output_tokens}\n"
        f"Total cost:    ${total_cost_usd:.4f}"
    )