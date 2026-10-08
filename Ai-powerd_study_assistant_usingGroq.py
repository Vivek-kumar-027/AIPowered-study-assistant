import gradio as gr
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

# Create Groq client
client = Groq(api_key=api_key)


# -----------------------------
# AI Personalities
# -----------------------------

personalities = {
    "Friendly": """
You are a friendly, enthusiastic, and highly encouraging Study Assistant.

Your goal is to break down complex concepts into simple,
beginner-friendly explanations.

Use:
- Simple language
- Analogies
- Real-world examples
- Step-by-step explanations

Always ask one follow-up question to check the student's understanding.
""",

    "Academic": """
You are a strictly academic, highly detailed, and professional
university Professor.

Use:
- Precise terminology
- Formal explanations
- Proper structure
- Important concepts
- Real-world examples

Your goal is to break down complex concepts into simple,
beginner-friendly explanations while maintaining academic accuracy.

Always ask one follow-up question to check the student's understanding.
"""
}


# -----------------------------
# Study Assistant Function
# -----------------------------

def study_assistant(question, persona):

    system_prompt = personalities[persona]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],

        temperature=0.4,
        max_completion_tokens=2000
    )

    return response.choices[0].message.content


# -----------------------------
# Gradio Interface
# -----------------------------

demo = gr.Interface(
    fn=study_assistant,

    inputs=[
        gr.Textbox(
            lines=4,
            placeholder="Ask a question...",
            label="Question"
        ),

        gr.Radio(
            choices=list(personalities.keys()),
            value="Friendly",
            label="Personality"
        )
    ],

    outputs=gr.Textbox(
        lines=10,
        label="Response"
    ),

    title="AI Study Assistant",

    description=(
        "Ask a question and get an answer from your "
        "AI study assistant with a chosen personality."
    )
)


# -----------------------------
# Launch App
# -----------------------------

demo.launch()