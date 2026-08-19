import os
import json
import logging
import traceback
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from groq import AsyncGroq

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("explain-it-simple")

# Load environment variables
load_dotenv()

app = FastAPI(title="Explain It Simple — AI Mentor Tool")

# Custom exception handler for request validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.warning(f"Request validation failed: {errors}")
    traceback.print_exc()
    
    # Extract the first error message to present cleanly to the user
    detail_msg = "Invalid input details provided. Please check your entries."
    if errors:
        err = errors[0]
        loc = err.get("loc", [])
        field = loc[-1] if loc else "field"
        err_msg = err.get("msg", "invalid value")
        detail_msg = f"Validation failed for '{field}': {err_msg}"
        
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": detail_msg}
    )

# Custom exception handler for HTTPExceptions to log them
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.status_code} - {exc.detail}")
    if exc.status_code >= 500:
        traceback.print_exc()
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Catch-all exception handler to avoid leaking tracebacks
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("An unhandled exception occurred")
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

# -----------------
# PYDANTIC MODELS
# -----------------

class GenerateRequest(BaseModel):
    topic: str = Field(..., max_length=150)
    audience: str = Field(..., max_length=150)

    @field_validator("topic", "audience", mode="before")
    @classmethod
    def trim_and_validate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Value must be a string")
        v_trimmed = v.strip()
        if not v_trimmed:
            raise ValueError("Field cannot be empty or contain only whitespace")
        return v_trimmed

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_index: int = Field(..., ge=0, le=3)

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("Options list must have exactly 4 items")
        for i, opt in enumerate(v):
            if not isinstance(opt, str):
                raise ValueError(f"Option at index {i} must be a string")
            if not opt.strip():
                raise ValueError(f"Option at index {i} cannot be empty or blank")
        return [opt.strip() for opt in v]

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Question must be a non-empty string")
        return v.strip()

class Exercise(BaseModel):
    prompt: str
    model_answer: str

    @field_validator("prompt", "model_answer", mode="before")
    @classmethod
    def validate_exercise_strings(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Exercise prompt and model answer must be non-empty strings")
        return v.strip()

class VisualStep(BaseModel):
    label: str
    description: str

    @field_validator("label", "description", mode="before")
    @classmethod
    def validate_strings(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Label and description must be non-empty strings")
        return v.strip()

class GenerateResponse(BaseModel):
    image_prompt: str
    explanation: str
    analogy: str
    quiz: List[QuizQuestion]
    exercises: List[Exercise]
    visual_steps: Optional[List[VisualStep]] = None

    @field_validator("quiz")
    @classmethod
    def validate_quiz_length(cls, v: List[QuizQuestion]) -> List[QuizQuestion]:
        if len(v) < 1:
            raise ValueError("Quiz must contain at least 1 question")
        return v

    @field_validator("exercises")
    @classmethod
    def validate_exercises_length(cls, v: List[Exercise]) -> List[Exercise]:
        if len(v) != 2:
            raise ValueError("Exercises list must contain exactly 2 items")
        return v

    @field_validator("image_prompt", "explanation", "analogy", mode="before")
    @classmethod
    def validate_strings(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("image_prompt, explanation, and analogy must be non-empty strings")
        return v.strip()

class GradeRequest(BaseModel):
    topic: str = Field(..., max_length=150)
    quiz: List[QuizQuestion]
    user_answers: List[int]
    exercises: List[Exercise]
    user_exercise_answers: List[str]

    @field_validator("topic", mode="before")
    @classmethod
    def trim_topic(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Topic must be a string")
        v_trimmed = v.strip()
        if not v_trimmed:
            raise ValueError("Topic cannot be empty")
        return v_trimmed

    @field_validator("quiz")
    @classmethod
    def validate_quiz_length(cls, v: List[QuizQuestion]) -> List[QuizQuestion]:
        if len(v) < 1:
            raise ValueError("Quiz must contain at least 1 question")
        return v

    @field_validator("user_answers")
    @classmethod
    def validate_answers(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("User answers cannot be empty")
        for i, ans in enumerate(v):
            if ans is None or ans < 0 or ans > 3:
                raise ValueError(f"Answer at index {i} must be between 0 and 3")
        return v

    @field_validator("exercises")
    @classmethod
    def validate_exercises_length(cls, v: List[Exercise]) -> List[Exercise]:
        if len(v) != 2:
            raise ValueError("Exercises must contain exactly 2 items")
        return v

    @field_validator("user_exercise_answers")
    @classmethod
    def validate_user_exercise_answers(cls, v: List[str]) -> List[str]:
        if len(v) != 2:
            raise ValueError("User exercise answers must contain exactly 2 items")
        for i, ans in enumerate(v):
            if not isinstance(ans, str) or not ans.strip():
                raise ValueError(f"Exercise answer at index {i} cannot be empty or blank")
        return [ans.strip() for ans in v]

class GradeResponse(BaseModel):
    score: int = Field(..., ge=0, le=3)
    total: int = Field(3, ge=3, le=3)
    feedback: str

    @field_validator("feedback", mode="before")
    @classmethod
    def validate_feedback(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Feedback must be a non-empty string")
        return v.strip()

# -----------------
# GROQ CLIENT HELPER
# -----------------

_groq_client = None

def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.strip() == "your_groq_api_key_here":
            logger.error("GROQ_API_KEY is not set or is set to placeholder in environment")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY is not configured on the server. Please set GROQ_API_KEY in your Vercel Environment Variables (or .env file for local development)."
            )
        _groq_client = AsyncGroq(api_key=api_key.strip())
    return _groq_client

# -----------------
# SYSTEM PROMPTS
# -----------------

GENERATION_SYSTEM_PROMPT = """You are a master educator who specializes in providing a crystal-clear, deep, and intuitive understanding of ANY topic to a beginner.

DO NOT output any <think> tags, chain-of-thought reasoning, or explanatory text before or after the JSON. Your response MUST begin immediately with '{' and end with '}'.

Do the following:

1. EXPLANATION SEPARATE HEADINGS DIRECTIVE: Write a plain-language explanation of the topics requested. If the user provides multiple concepts (e.g. "Quantum Superstitions, arrays, strings, recursion"), explain EVERY requested concept clearly under its OWN distinct Markdown sub-heading (e.g. "### 1. Quantum Superstitions\n...", "### 2. Arrays\n...", "### 3. Strings\n...", "### 4. Recursion\n..."). Ensure every single topic receives thorough, clear, and individual attention tailored to the target audience.

2. Avoid unnecessary jargon. If a technical term is necessary,
immediately explain it in simple language.

3. Give ONE simple real-world analogy that connects the core ideas or explains the main concept.

4. EVERY-TOPIC QUIZ DIRECTIVE: Generate multiple-choice quiz questions that test whether the user understood the explanation. CRITICAL RULE: Generate at least 1 question for EVERY single topic/concept provided in the request (e.g., if 4 concepts are given like 'Quantum Superstitions, arrays, strings, recursion', generate 4 questions in total so that every concept has its own dedicated question). If only 1 concept is given, generate 3 questions. Each question must clearly test its specific concept.

5. Each question must have exactly 4 options and exactly one correct answer.
correct_index must be an integer from 0 to 3.

6. HANDS-ON EXERCISE DIRECTIVE: Generate exactly 2 short, practical exercises. CRITICAL RULE: Every exercise question MUST be based STRICTLY AND EXCLUSIVELY on the facts, concepts, and definitions explained in your plain-language explanation (`explanation`). DO NOT ask questions beyond what was explicitly explained in your text, and DO NOT test outside/advanced syntax or external knowledge that was not introduced in your explanation. Include a brief model answer for each based strictly on your text.

7. If (and only if) the topic naturally involves a sequence, process,
or set of steps, also generate a "visual_steps" array of 3-5 short steps, each
with a "label" (2-4 words) and a "description" (one short sentence).
Otherwise omit visual_steps or set it to null.

8. IMAGE PROMPT RULE: Provide a direct, literal, educational illustration prompt for the topic in "image_prompt". The prompt MUST describe the actual subject matter itself clearly and visually. DO NOT describe abstract metaphors, toy blocks, glowing lightbulbs, or conceptual decorations.

Output ONLY valid JSON matching this exact structure:
{
  "explanation": "...",
  "analogy": "...",
  "quiz": [
    {
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0
    }
  ],
  "exercises": [
    {
      "prompt": "...",
      "model_answer": "..."
    }
  ],
  "visual_steps": [
    {
      "label": "...",
      "description": "..."
    }
  ],
  "image_prompt": "..."
}

Do not include any text outside the JSON object."""

GRADING_SYSTEM_PROMPT = """You are a supportive teacher reviewing a student's quiz and exercise answers.

Determine the student's score on the multiple-choice quiz (out of 3).

Then write short encouraging feedback in 3-5 sentences.

The feedback should:

- Mention what the student understood well in the quiz and exercises.
- Gently clarify anything they got wrong in the quiz or exercises.
- Provide a brief tip on how their practical exercise answers could be improved (based on the model answers), while remaining highly supportive.
- Avoid harsh or judgmental language.

Return ONLY this JSON object:

{
  "score": 0,
  "total": 3,
  "feedback": "string"
}

The score must be an integer between 0 and 3.

The total must be 3.

Do not include Markdown.

Do not include ```json.

Do not include comments.

Do not include any text outside the JSON object."""

# -----------------
# API ENDPOINTS
# -----------------

_resolved_model = None

async def get_model_id(client: AsyncGroq) -> str:
    global _resolved_model
    if _resolved_model is not None:
        return _resolved_model
    try:
        models_list = await client.models.list()
        available_ids = [m.id for m in models_list.data]
        logger.info(f"Available models on Groq: {available_ids}")
        
        valid_models = [
            m for m in available_ids 
            if not any(bad in m.lower() for bad in ["whisper", "guard", "arabic", "canopy", "orpheus"])
        ]
        
        preferred_models = [
            "openai/gpt-oss-20b",
            "allam-2-7b",
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-120b",
            "groq/compound-mini",
            "groq/compound"
        ]
        for pm in preferred_models:
            if pm in valid_models:
                _resolved_model = pm
                break
                
        if _resolved_model is None and valid_models:
            _resolved_model = valid_models[0]
        elif _resolved_model is None:
            _resolved_model = "openai/gpt-oss-20b"
            
    except Exception as e:
        logger.error(f"Failed to query model list from Groq: {e}.")
        _resolved_model = "openai/gpt-oss-20b"
    logger.info(f"Resolved Groq model to use: '{_resolved_model}'")
    return _resolved_model

def repair_truncated_json(json_str: str) -> str:
    """Repairs truncated JSON strings by balancing quotes, brackets, and braces."""
    json_str = json_str.strip()
    if not json_str:
        return "{}"
    first_brace = json_str.find('{')
    if first_brace != -1:
        json_str = json_str[first_brace:]
    else:
        return "{}"
        
    in_string = False
    escape = False
    stack = []
    
    for char in json_str:
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if stack:
                    stack.pop()
                    
    if in_string:
        json_str += '"'
        
    json_str = json_str.rstrip()
    if json_str.endswith(','):
        json_str = json_str[:-1]
        
    while stack:
        opener = stack.pop()
        if opener == '{':
            json_str += '}'
        elif opener == '[':
            json_str += ']'
            
    return json_str

def safe_parse_json(text: str) -> dict:
    """Parses JSON text robustly, handling thinking tags, codeblocks, unescaped newlines, and truncation."""
    if not text:
        return {}
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        fb = text.find('{')
        if fb != -1:
            text = text[fb:].strip()
            
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        text = text[first_brace:last_brace + 1]
    elif first_brace != -1:
        text = text[first_brace:]
        
    try:
        return json.loads(text, strict=False)
    except Exception:
        repaired = repair_truncated_json(text)
        try:
            return json.loads(repaired, strict=False)
        except Exception:
            lines = text.splitlines()
            joined = " ".join(l.strip() for l in lines)
            repaired_joined = repair_truncated_json(joined)
            return json.loads(repaired_joined, strict=False)

@app.post("/api/mentor/generate", response_model=GenerateResponse)
async def generate_explanation(request: GenerateRequest):
    logger.info(f"Generating explanation for topic='{request.topic}', audience='{request.audience}'")
    
    client = get_groq_client()
    model_id = await get_model_id(client)
    system_prompt = GENERATION_SYSTEM_PROMPT
    
    candidate_models = [model_id, "openai/gpt-oss-20b", "allam-2-7b", "qwen/qwen3.6-27b", "openai/gpt-oss-120b"]
    # De-duplicate candidate models order
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    parsed_json = None
    last_response_text = ""

    for target_model in models_to_try:
        try:
            logger.info(f"Attempting completion with Groq model: '{target_model}'")
            try:
                chat_completion = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Explain '{request.topic}' to '{request.audience}'. Respond ONLY with valid JSON."}
                    ],
                    model=target_model,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=4000,
                )
            except Exception as format_err:
                logger.warning(f"Completion with response_format json_object failed for '{target_model}': {format_err}. Retrying without response_format...")
                chat_completion = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Explain '{request.topic}' to '{request.audience}'. Respond ONLY with valid JSON."}
                    ],
                    model=target_model,
                    temperature=0.3,
                    max_tokens=4000,
                )

            last_response_text = chat_completion.choices[0].message.content or ""
            parsed = safe_parse_json(last_response_text)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            
            if isinstance(parsed, dict):
                explanation_str = str(parsed.get("explanation", "")).strip()
                analogy_str = str(parsed.get("analogy", "")).strip()
                # Ensure explanation and analogy are substantive AI outputs, not empty or generic dummy fallbacks
                if len(explanation_str) > 30 and len(analogy_str) > 15:
                    parsed_json = parsed
                    logger.info(f"Successfully generated full explanation with model '{target_model}'")
                    break
                else:
                    logger.warning(f"Model '{target_model}' returned insufficient explanation or analogy content. Lengths: exp={len(explanation_str)}, ana={len(analogy_str)}. Retrying next model...")
        except Exception as err:
            logger.warning(f"Completion failed with model '{target_model}': {err}. Trying next fallback model...")

    if not parsed_json or not isinstance(parsed_json, dict):
        logger.error(f"Failed to generate valid explanation across all candidate models. Content: {last_response_text[:500]}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to generate explanation from AI model. Please try again."
        )

    # Fill optional/secondary defaults safely if needed, preserving the real explanation and analogy
    if "image_prompt" not in parsed_json or not parsed_json.get("image_prompt"):
        parsed_json["image_prompt"] = f"A clear, crisp, high-contrast educational illustration for {request.topic}."

    if "quiz" not in parsed_json or not isinstance(parsed_json.get("quiz"), list) or len(parsed_json["quiz"]) == 0:
        parsed_json["quiz"] = [
            {"question": f"What is a core aspect of {request.topic}?", "options": ["Essential foundational element", "Unrelated option", "Incorrect detail", "None of these"], "correct_index": 0},
            {"question": f"Why is understanding {request.topic} valuable?", "options": ["It provides key structural insights", "It has no value", "It slows things down", "None"], "correct_index": 0},
            {"question": f"How is {request.topic} applied?", "options": ["Through systematic implementation", "By ignoring rules", "By deleting data", "None"], "correct_index": 0}
        ]

    if "exercises" not in parsed_json or not isinstance(parsed_json.get("exercises"), list) or len(parsed_json["exercises"]) == 0:
        parsed_json["exercises"] = [
            {"prompt": f"Explain {request.topic} in your own words.", "model_answer": f"A summary of {request.topic}."},
            {"prompt": f"Give a practical example of {request.topic}.", "model_answer": f"An example illustrating {request.topic}."}
        ]

    # Validate JSON against Pydantic model
    try:
        validated_response = GenerateResponse(**parsed_json)
        return validated_response
    except Exception as ve:
        logger.error(f"AI response failed Pydantic validation: {ve}. Content: {last_response_text}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Something went wrong while generating the explanation. The AI response was malformed."
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected Groq API failure: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Something went wrong while generating the explanation. Please try again."
        )

@app.post("/api/mentor/grade", response_model=GradeResponse)
async def grade_quiz(request: GradeRequest):
    logger.info(f"Grading quiz for topic='{request.topic}', user_answers={request.user_answers}")
    
    # Format questions and correct answers for the grading prompt
    quiz_lines = []
    for i, q in enumerate(request.quiz):
        correct_opt = q.options[q.correct_index] if 0 <= q.correct_index < len(q.options) else "Unknown"
        quiz_lines.append(
            f"Question {i+1}: {q.question}\n"
            f"Options: {q.options}\n"
            f"Correct Option Index: {q.correct_index} (Answer Text: {correct_opt})"
        )
    quiz_with_answers = "\n\n".join(quiz_lines)
    
    # Format user submitted answers
    user_answers_lines = []
    for i, ans in enumerate(request.user_answers):
        opt_text = "Unknown"
        if i < len(request.quiz) and 0 <= ans < len(request.quiz[i].options):
            opt_text = request.quiz[i].options[ans]
        user_answers_lines.append(f"Question {i+1} answer index: {ans} (Student answered: {opt_text})")
    user_answers_str = "\n".join(user_answers_lines)
    # Format exercises and model answers
    exercises_lines = []
    for i, ex in enumerate(request.exercises):
        exercises_lines.append(
            f"Exercise {i+1}: {ex.prompt}\n"
            f"Model Answer: {ex.model_answer}"
        )
    exercises_with_answers = "\n\n".join(exercises_lines)

    # Format user exercise answers
    user_exercise_lines = []
    for i, ans in enumerate(request.user_exercise_answers):
        user_exercise_lines.append(f"Exercise {i+1} answer: {ans}")
    user_exercise_answers_str = "\n".join(user_exercise_lines)
    
    client = get_groq_client()
    model_id = await get_model_id(client)
    system_prompt = GRADING_SYSTEM_PROMPT
    
    try:
        try:
            chat_completion = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Topic: '{request.topic}'\n\nMultiple-Choice Questions and correct answers:\n{quiz_with_answers}\n\nStudent's submitted multiple-choice answers:\n{user_answers_str}\n\nPractical Exercises and Model Answers:\n{exercises_with_answers}\n\nStudent's submitted exercise answers:\n{user_exercise_answers_str}\n\nGrade the student's response."}
                ],
                model=model_id,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except Exception as api_err:
            if "json_validate_failed" in str(api_err) or "400" in str(api_err):
                logger.warning(f"Retrying grading chat completion without response_format due to Groq error: {api_err}")
                chat_completion = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Topic: '{request.topic}'\n\nMultiple-Choice Questions and correct answers:\n{quiz_with_answers}\n\nStudent's submitted multiple-choice answers:\n{user_answers_str}\n\nPractical Exercises and Model Answers:\n{exercises_with_answers}\n\nStudent's submitted exercise answers:\n{user_exercise_answers_str}\n\nGrade the student's response. Respond ONLY with a valid JSON object."}
                    ],
                    model=model_id,
                    temperature=0.1,
                )
            else:
                raise api_err
        
        response_text = chat_completion.choices[0].message.content or ""
        logger.debug(f"Raw Groq response for grade: {response_text}")
        
        # Clean up CoT thinking blocks (e.g. from DeepSeek/Qwen models)
        if "</think>" in response_text:
            cleaned_text = response_text.split("</think>")[-1].strip()
        else:
            cleaned_text = response_text.strip()

        # Clean up markdown formatting if the LLM included it
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Parse the JSON response
        try:
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Fallback: extract the JSON object using string searching (first '{' and last '}')
            first_brace = cleaned_text.find('{')
            last_brace = cleaned_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = cleaned_text[first_brace:last_brace + 1]
                try:
                    parsed_json = json.loads(json_candidate)
                except json.JSONDecodeError as jde:
                    logger.error(f"Failed to parse Groq grading response even after JSON extraction. Content: {cleaned_text}. Error: {jde}")
                    traceback.print_exc()
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Something went wrong while grading the quiz. The AI returned an invalid response format."
                    )
            else:
                logger.error(f"Failed to parse Groq grading response as JSON (no braces found). Content: {cleaned_text}")
                traceback.print_exc()
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Something went wrong while grading the quiz. The AI returned an invalid response format."
                )
            
        # If LLM returned a list containing the JSON object, extract the first item
        if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
            parsed_json = parsed_json[0]

        # Validate JSON against Pydantic model
        try:
            validated_response = GradeResponse(**parsed_json)
            return validated_response
        except Exception as ve:
            logger.error(f"AI grading response failed Pydantic validation: {ve}. Content: {response_text}")
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Something went wrong while grading the quiz. The AI response was malformed."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected Groq API failure during grading: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Something went wrong while grading the quiz. Please try again."
        )

# Serve the static files. Note: This route must come AFTER API routes.
# Ensure the static/ directory exists. If not, FastAPI will throw an error on startup.
# We'll create the static/ directory first.
@app.get("/")
async def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend static index.html not found.")
    return FileResponse(index_path)

# Mount static directory for resources like css, assets, or index.html if needed
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")
