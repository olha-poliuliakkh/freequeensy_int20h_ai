import json
from groq import Groq
import instructor
from pydantic import BaseModel, Field
from typing import List, Dict, Literal

from dotenv import load_dotenv
import os

load_dotenv()


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
    def __init__(self, api_key, model="openai/gpt-oss-20b"):
        self.client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        self.model = model
        self.system_prompt = (
            "You are an expert Quality Assurance specialist in customer support. "
            "Analyze the provided dialogue and provide a structured evaluation."
        )

    def analyze_chat(self, chat_messages: List[Dict]) -> ChatAnalysis:
        formatted_chat = "\n".join([f"{m['role']}: {m['text']}" for m in chat_messages])
        
        analysis_result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Analyze this conversation: {formatted_chat}"}
            ],
            response_model=ChatAnalysis,
            temperature=0,
            seed=27
        )

        return analysis_result


if __name__ == "__main__":
    analyzer = ChatAnalyzer(api_key=os.environ.get("OPENAI_API_KEY"))

    with open("generated_chats_dataset.json", "r", encoding="utf-8") as f:
        chats_to_analyze = json.load(f)

    analyzed_dataset = []
    for entry in chats_to_analyze:
        print(f"Chat scenario for analysis: {entry['scenario_description']}")
        for i, chat in enumerate(entry['generated_data']['chats']):
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

    print(f"Analysis is completed! Results in {output_file}.")