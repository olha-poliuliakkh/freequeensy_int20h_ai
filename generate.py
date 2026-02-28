import json
import itertools

from groq import Groq
import instructor
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

class ChatMessage(BaseModel):
    role: str = Field(description="Role of the speaker: 'client' or 'agent'")
    text: str = Field(description="The content of the message")

class SingleChat(BaseModel):
    chat_id: int = Field(description="The sequential number of the chat (e.g., 1, 2, 3)")
    messages: List[ChatMessage] = Field(description="List of messages in this conversation")

class ChatDataset(BaseModel):
    chats: List[SingleChat] = Field(description="A list of generated conversations")


class ChatGenerator:
    def __init__(self, api_key, model="openai/gpt-oss-20b"):
        self.client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        self.model = model
        self.system_prompt = ("You are a professional generator of realistic support chats in English")

    def generate(self, scenario, num_chats=2):
        prompt = f"Generate {num_chats} diverse chat conversations based on the following scenario: {scenario}"
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            response_model=ChatDataset,
            temperature=0,
            seed=27,
            top_p=0.9,
            presence_penalty=0.3,
            frequency_penalty=0.4
        )
        
        return response


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
    generator = ChatGenerator(api_key=os.environ.get("OPENAI_API_KEY"))

    dataset = []
    for i, scenario in enumerate(all_scenarios):
        print(f"Generating...  {i}/{len(all_scenarios)}: {scenario}")

        try:
            result = generator.generate(scenario, num_chats=1)
            chat_entry = {
                "scenario_index": i,
                "scenario_description": scenario,
                "generated_data": result.model_dump()
            }
            dataset.append(chat_entry)

        except Exception as e:
            print(f"Error with {i}: {e}")

    output_file = "generated_chats_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Completed! Chats are in {output_file}")