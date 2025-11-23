import os
from groq import Groq
from dotenv import load_dotenv

# Loads environment variables from the .env file.
load_dotenv()

# Retrieves the API key from environment variables.
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    # Raise an error if the GROQ_API_KEY is missing or empty.
    raise ValueError("GROQ_API_KEY not found or empty in environment variables.")

# Initialize the Groq client.
client = Groq(api_key=api_key) 

# Use the model specified in the project requirements.
MODEL_NAME = "llama-3.3-70b-versatile"
MAX_TOKENS = 500 # Set based on the maximum token budget for the Debater (Opening).

def generate_baseline_response(prompt: str) -> str:
    """
    Generates a response for the B1: Single-Agent (Zero-shot) baseline.
    """
    print(f"--- Calling Model: {MODEL_NAME} ---")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME, 
            max_tokens=MAX_TOKENS, 
            # Set temperature close to the default (0.0) for initial testing consistency.
            temperature=0.01 
        )
        
        # Returns the generated response text.
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"API Call Error: {e}"

# Execute legal reasoning test
# The prompt is translated to English for international consistency in the code.
test_prompt = "Briefly explain the legal basis for punishing an attempt (misubum) under criminal law and the requirements for its establishment."
response = generate_baseline_response(test_prompt)

print("\n" + "="*50)
print(f"[TEST PROMPT]\n{test_prompt}")
print("\n" + "-"*50)
print(f"[Llama 3.3 70B Response]\n{response}")
print("="*50 + "\n")