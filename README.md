# Customer Support Chat Dataset Generator & Analyzer

A tool for generating synthetic customer support dialogues and automatically evaluating agent quality using the Groq API.

## How It Works

The pipeline consists of two steps:

**Step 1 — Generation (`grokim.py`)**
Generates realistic support chat dialogues based on combinations of scenarios. Each scenario is defined by:
- **Theme**: `payment issues`, `technical errors`, `account access`, `tariff inquiries`, `refunds`
- **Case type**: `successful`, `problematic`, `conflict`, `inappropriate communication style`, `reasoning error`
- **Duration**: `short`, `average`, `long`
- **Client emotion**: `satisfaction`, `dissatisfaction`, `hidden dissatisfaction`
- **Conversation flow**: `logically finished`, `abruptly ended`

Dialogues are returned as structured JSON via Pydantic models.

**Step 2 — Analysis (`grokan.py`)**
Each generated dialogue is evaluated by the model across four dimensions:

| Field | Description |
|---|---|
| `intent` | Client's request category (one of the themes above, or `other`) |
| `satisfaction` | `satisfied` / `neutral` / `unsatisfied` |
| `quality_score` | Agent performance score from 1 (very poor) to 5 (excellent) |
| `agent_mistakes` | List of detected mistakes: `ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation` |

## Project Structure

```
.
├── main.py                       # Entry point — runs generation then analysis
├── grokim.py                     # Chat generation logic
├── grokan.py                     # Chat analysis logic
├── .env                          # API keys (not committed)
├── requirements.txt              # Python dependencies
├── generated_chats_dataset.json  # Output: generated dialogues
└── analyzed_support_data.json    # Output: analysis results
```

## Setup

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

Or with `uv`:
```bash
uv sync
```

**2. Create a `.env` file in the project root:**
```
OPENAI_API_KEY=your_api_key_here
```

## Running

```bash
python main.py
```

The script will run generation and analysis sequentially and print progress to the console:

```
[1/270] Theme: payment issues, Case: successful, Duration: short, ...
[2/270] Theme: payment issues, Case: successful, Duration: short, ...
...
Generation done. 270 scenarios saved to generated_chats_dataset.json

Analyzing: Theme: payment issues, Case: successful, ...
...
Analysis done. 270 chats saved to analyzed_support_data.json
```

## Output Format

**`generated_chats_dataset.json`**
```json
[
  {
    "scenario_index": 0,
    "scenario_description": "Theme: payment issues, Case: successful, ...",
    "generated_data": {
      "chats": [
        {
          "chat_id": 1,
          "messages": [
            {"role": "client", "text": "..."},
            {"role": "agent", "text": "..."}
          ]
        }
      ]
    }
  }
]
```

**`analyzed_support_data.json`**
```json
[
  {
    "scenario_index": 0,
    "scenario": "Theme: payment issues, Case: successful, ...",
    "chat_id": 1,
    "analysis": {
      "intent": "payment issues",
      "satisfaction": "satisfied",
      "quality_score": 5,
      "agent_mistakes": [],
      "summary_reasoning": "The agent resolved the issue promptly and professionally."
    }
  }
]
```