"""
Prompt templates for MAD legal reasoning.
"""


def get_debater_opening_prompt(question: str, prompt_context: str, choices: list, position: str) -> str:
    """
    Generate opening argument prompt for debater.

    Args:
        question: The legal question
        prompt_context: Context/prompt from dataset
        choices: List of 4 answer choices
        position: Which choice to defend (A, B, C, or D)

    Returns:
        Formatted prompt string
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    prompt = f"""You are a legal expert participating in a debate.

{full_question}

Answer choices:
{choices_text}

Your task: Argue convincingly that answer choice {position} is the legally correct answer.
Cite relevant legal authorities (statutes, cases, legal doctrines).

Respond in JSON format:
{{
  "position": "{position}",
  "argument": "Your detailed legal argument here...",
  "citations": ["Citation 1", "Citation 2", "..."]
}}"""

    return prompt


def get_debater_rebuttal_prompt(
    question: str,
    prompt_context: str,
    my_position: str,
    my_opening: dict,
    opponent_opening: dict
) -> str:
    """
    Generate rebuttal prompt for debater.

    Args:
        question: The legal question
        prompt_context: Context from dataset
        my_position: This debater's position
        my_opening: This debater's opening argument (dict)
        opponent_opening: Opponent's opening argument (dict)

    Returns:
        Formatted prompt string
    """
    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    prompt = f"""You are continuing your legal debate.

Question:
{full_question}

Your previous argument (defending {my_position}):
{my_opening.get('argument', '')}

Opponent's argument (defending {opponent_opening.get('position', '')}):
{opponent_opening.get('argument', '')}

Your task:
1. Identify weaknesses in opponent's argument
2. Explain why your position ({my_position}) is legally superior
3. Reinforce your argument with additional legal reasoning

Respond in JSON format:
{{
  "rebuttal": "Your rebuttal argument here...",
  "counterarguments": ["Point against opponent 1", "Point against opponent 2"],
  "citations": ["Additional citation 1", "..."]
}}"""

    return prompt


def get_judge_decision_prompt(
    question: str,
    prompt_context: str,
    choices: list,
    debate_history: dict
) -> str:
    """
    Generate judge decision prompt.

    Args:
        question: The legal question
        prompt_context: Context from dataset
        choices: List of 4 answer choices
        debate_history: Full debate history with openings and rebuttals

    Returns:
        Formatted prompt string
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    # Extract debate content
    debater_x = debate_history.get('debater_x', {})
    debater_y = debate_history.get('debater_y', {})

    prompt = f"""You are an impartial legal judge reviewing a debate between two legal experts.

Question:
{full_question}

Answer choices:
{choices_text}

Debater X (defending {debater_x.get('opening', {}).get('position', '')}):
Opening argument: {debater_x.get('opening', {}).get('argument', '')}
Rebuttal: {debater_x.get('rebuttal', {}).get('rebuttal', '')}

Debater Y (defending {debater_y.get('opening', {}).get('position', '')}):
Opening argument: {debater_y.get('opening', {}).get('argument', '')}
Rebuttal: {debater_y.get('rebuttal', {}).get('rebuttal', '')}

Your task: Based on the legal arguments presented, select the most legally sound answer choice.
Consider:
- Accuracy of legal reasoning
- Quality and relevance of citations
- Strength of application to the facts
- How well each side addressed counterarguments

Respond in JSON format:
{{
  "decision": "A, B, C, or D",
  "rationale": "Brief explanation of why this answer is most legally sound..."
}}"""

    return prompt
