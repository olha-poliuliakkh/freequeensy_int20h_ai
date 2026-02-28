import json
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

import generate as gen_module
import analyze as ana_module
from generate import ChatGenerator, generate_all_scenarios
from analyze import ChatAnalyzer

COST_LIMIT_USD = 20.0
MAX_WORKERS = 5

load_dotenv()

GENERATE_OUTPUT = "generated_chats_dataset.json"
ANALYZE_OUTPUT = "analyzed_support_data.json"
NUM_CHATS_PER_SCENARIO = 1


def run_generation(key: str):
    scenarios = generate_all_scenarios()
    generator = ChatGenerator(api_key=key)

    def generate_one(args):
        i, scenario = args
        if gen_module.total_cost_usd >= COST_LIMIT_USD:
            print(f"  Limit reached. Skipping scenario {i + 1}.")
            return None
        print(f"[{i + 1}/{len(scenarios)}] {scenario}")
        try:
            result = generator.generate(scenario, num_chats=NUM_CHATS_PER_SCENARIO)
            return {"scenario_index": i, "scenario_description": scenario, "generated_data": result.model_dump()}
        except Exception as e:
            print(f"  Error [{i + 1}]: {e}")
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(generate_one, enumerate(scenarios)))

    data = [r for r in results if r is not None]

    with open(GENERATE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nGeneration done. {len(data)} scenarios saved to {GENERATE_OUTPUT}\n")
    return data


def run_analysis(key: str, data: list):
    analyzer = ChatAnalyzer(api_key=key)

    # Flatten all chats into a single list for parallel processing
    tasks = [
        (entry['scenario_index'], entry['scenario_description'], i + 1, chat)
        for entry in data
        for i, chat in enumerate(entry['generated_data']['chats'])
    ]

    def analyze_one(args):
        scenario_index, scenario_desc, chat_id, chat = args
        if gen_module.total_cost_usd + ana_module.total_cost_usd >= COST_LIMIT_USD:
            print(f"  Limit reached. Skipping chat {chat_id}.")
            return None
        try:
            result = analyzer.analyze_chat(chat['messages'])
            return {"scenario_index": scenario_index, "scenario": scenario_desc, "chat_id": chat_id, "analysis": result.model_dump()}
        except Exception as e:
            print(f"  Error on chat {chat_id}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(analyze_one, tasks))

    analyzed = [r for r in results if r is not None]

    with open(ANALYZE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(analyzed, f, indent=4, ensure_ascii=False)

    print(f"\nAnalysis done. {len(analyzed)} chats saved to {ANALYZE_OUTPUT}")


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    dataset = run_generation(api_key)
    run_analysis(api_key, dataset)

    total_cost = gen_module.total_cost_usd + ana_module.total_cost_usd
    total_input = gen_module.total_input_tokens + ana_module.total_input_tokens
    total_output = gen_module.total_output_tokens + ana_module.total_output_tokens

    print(
        f"\n=== TOTAL ===\n"
        f"Input tokens:  {total_input}\n"
        f"Output tokens: {total_output}\n"
        f"Generation:    ${gen_module.total_cost_usd:.4f}\n"
        f"Analysis:      ${ana_module.total_cost_usd:.4f}\n"
        f"Total cost:    ${total_cost:.4f}"
    )