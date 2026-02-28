import json
import os
from dotenv import load_dotenv

from generate import ChatGenerator, generate_all_scenarios
from analyze import ChatAnalyzer

load_dotenv()

GENERATE_OUTPUT = "generated_chats_dataset.json"
ANALYZE_OUTPUT = "analyzed_support_data.json"
NUM_CHATS_PER_SCENARIO = 1


def run_generation(api_key: str):
    scenarios = generate_all_scenarios()
    generator = ChatGenerator(api_key=api_key)
    dataset = []

    for i, scenario in enumerate(scenarios):
        print(f"[{i + 1}/{len(scenarios)}] {scenario}")
        try:
            result = generator.generate(scenario, num_chats=NUM_CHATS_PER_SCENARIO)
            dataset.append({
                "scenario_index": i,
                "scenario_description": scenario,
                "generated_data": result.model_dump()
            })
        except Exception as e:
            print(f"  Error: {e}")

    with open(GENERATE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nGeneration done. {len(dataset)} scenarios saved to {GENERATE_OUTPUT}\n")
    return dataset


def run_analysis(api_key: str, dataset: list):

    analyzer = ChatAnalyzer(api_key=api_key)
    analyzed_dataset = []

    for entry in dataset:
        print(f"Analyzing: {entry['scenario_description']}")
        for i, chat in enumerate(entry['generated_data']['chats']):
            try:
                result = analyzer.analyze_chat(chat['messages'])
                analyzed_dataset.append({
                    "scenario_index": entry['scenario_index'],
                    "scenario": entry['scenario_description'],
                    "chat_id": i + 1,
                    "analysis": result.model_dump()
                })
            except Exception as e:
                print(f"  Error on chat {i + 1}: {e}")

    with open(ANALYZE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(analyzed_dataset, f, indent=4, ensure_ascii=False)

    print(f"\nAnalysis done. {len(analyzed_dataset)} chats saved to {ANALYZE_OUTPUT}")


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    dataset = run_generation(api_key)
    run_analysis(api_key, dataset)