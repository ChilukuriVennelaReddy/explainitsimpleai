# Explain It Simple

Explain It Simple is an AI-powered mentor tool that helps users understand difficult topics in simple language. The user provides a topic and a target audience, and the tool generates a plain-language explanation, a real-world analogy, and a 3-question multiple-choice quiz. The application then grades the user's answers and provides encouraging feedback.

## Features

- Simple AI explanations
- Real-world analogy
- 3-question quiz
- AI grading
- Encouraging feedback

## Requirements

Python 3.12 and a Groq API key.

## Setup

Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `.env`:
```text
GROQ_API_KEY=your_actual_key
```

Run the application:
```bash
uvicorn main:app --reload
```

Open the local application at http://127.0.0.1:8000.

## API endpoints

- `POST /api/mentor/generate` - Generate simplified explanation, analogy, and quiz.
- `POST /api/mentor/grade` - Grade the user's answers and provide feedback.
