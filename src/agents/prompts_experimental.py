"""
Prompt templates for MAD legal reasoning.
"""


def get_debater_opening_prompt(question: str, prompt_context: str, choices: list, position: str = None) -> str:
    """
    Generate opening argument prompt for debater.

    Args:
        question: The legal question
        prompt_context: Context/prompt from dataset
        choices: List of 4 answer choices
        position: Which choice to defend (A, B, C, or D). If None, debater chooses freely.

    Returns:
        Formatted prompt string
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    if position is None:
        # Debater chooses position freely
        prompt = f"""You are a legal expert participating in a debate.

{full_question}

Answer choices:
{choices_text}

Your task:
1. Analyze the question and select which answer choice you believe is most legally correct
2. Argue convincingly for that choice
3. Cite relevant legal authorities (statutes, cases, legal doctrines)

Respond in JSON format:
{{
  "position": "A, B, C, or D (your selected answer)",
  "argument": "Your detailed legal argument here...",
  "citations": ["Citation 1", "Citation 2", "..."]
}}"""
    else:
        # Debater must defend assigned position
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

Your task: Based on the legal arguments presented, select the most legally correct answer choice.
Consider:
- Accuracy of legal reasoning
- Quality and relevance of citations
- Strength of application to the facts
- How well each side addressed counterarguments

Respond in JSON format with 4 fields:
{{
  "rationale": "Concise analysis of the key points from each debater's arguments, identifying strengths and weaknesses of each position...",
  "winner": "debater_x, debater_y, or tie (which debater presented the stronger legal argument overall)",
  "decision": "A, B, C, or D (the legally correct answer)",
  "synthesis": "Your final explanation of why this is the legally correct answer, applying relevant legal principles and addressing the key issues in the question..."
}}"""

    return prompt


# ==================== IRAC PROMPTS ====================

def get_debater_opening_prompt_irac(question: str, prompt_context: str, choices: list, position: str = None) -> str:
    """
    Generate IRAC-structured opening argument prompt for debater.

    Args:
        question: The legal question
        prompt_context: Context/prompt from dataset
        choices: List of 4 answer choices
        position: Which choice to defend (A, B, C, or D). If None, debater chooses freely.

    Returns:
        Formatted prompt string with IRAC structure
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    if position is None:
        # Debater chooses position freely
        prompt = f"""You are a legal expert participating in a debate.

{full_question}

Answer choices:
{choices_text}

Your task:
1. Analyze the question and select which answer choice you believe is most legally correct
2. Structure your argument using IRAC methodology (Issue, Rule, Application, Conclusion)
3. Cite relevant legal authorities

Respond in JSON format using IRAC structure:
{{
  "position": "A, B, C, or D (your selected answer)",
  "irac": {{
    "issue": "Identify the key legal issue at the heart of the scenario",
    "rule": "Detail the specific laws or legal principles that govern the identified issue",
    "application": "Examine how the laws or principles apply to the facts of the case, discussing the legal merits based on this application",
    "conclusion": "Conclude by synthesizing the analysis to state why your chosen answer is correct"
  }},
  "citations": ["Citation 1", "Citation 2", "..."]
}}"""
    else:
        # Debater must defend assigned position
        prompt = f"""You are a legal expert participating in a debate.

{full_question}

Answer choices:
{choices_text}

Your task: Argue convincingly that answer choice {position} is the legally correct answer.
Structure your argument using IRAC methodology (Issue, Rule, Application, Conclusion).
Cite relevant legal authorities.

Respond in JSON format using IRAC structure:
{{
  "position": "{position}",
  "irac": {{
    "issue": "Identify the key legal issue at the heart of the scenario",
    "rule": "Detail the specific laws or legal principles that govern the identified issue",
    "application": "Examine how the laws or principles apply to the facts of the case, discussing the legal merits based on this application",
    "conclusion": "Conclude by synthesizing the analysis to state why {position} is correct"
  }},
  "citations": ["Citation 1", "Citation 2", "..."]
}}"""

    return prompt


def get_debater_rebuttal_prompt_irac(
    question: str,
    prompt_context: str,
    my_position: str,
    my_opening: dict,
    opponent_opening: dict
) -> str:
    """
    Generate IRAC-structured rebuttal prompt for debater.

    Args:
        question: The legal question
        prompt_context: Context from dataset
        my_position: This debater's position
        my_opening: This debater's opening argument (dict with IRAC structure)
        opponent_opening: Opponent's opening argument (dict with IRAC structure)

    Returns:
        Formatted prompt string with structured critique
    """
    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    # Extract opponent's IRAC components
    opponent_irac = opponent_opening.get('irac', {})
    opponent_pos = opponent_opening.get('position', '')

    prompt = f"""You are continuing your legal debate.

Question:
{full_question}

Your position: {my_position}

Opponent's position: {opponent_pos}
Opponent's IRAC argument:
- Issue: {opponent_irac.get('issue', '')}
- Rule: {opponent_irac.get('rule', '')}
- Application: {opponent_irac.get('application', '')}
- Conclusion: {opponent_irac.get('conclusion', '')}

Your task: Critique opponent's IRAC argument and reinforce why your position ({my_position}) is legally superior.

Respond in JSON format with structured critique:
{{
  "rebuttal": {{
    "issue_critique": "Explain if opponent misidentified the legal issue or missed key aspects",
    "rule_critique": "Explain if opponent's legal rule is incorrect, incomplete, or misapplied",
    "application_critique": "Explain flaws in how opponent applied the rule to the facts",
    "my_reinforcement": "Reinforce why your IRAC analysis is superior and leads to the correct answer"
  }},
  "citations": ["Additional citation 1", "..."]
}}"""

    return prompt


def get_judge_decision_prompt_irac(
    question: str,
    prompt_context: str,
    choices: list,
    debate_history: dict
) -> str:
    """
    Generate IRAC-structured judge decision prompt.

    Args:
        question: The legal question
        prompt_context: Context from dataset
        choices: List of 4 answer choices
        debate_history: Full debate history with IRAC-structured openings and rebuttals

    Returns:
        Formatted prompt string with IRAC synthesis requirement
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    # Extract debate content
    debater_x = debate_history.get('debater_x', {})
    debater_y = debate_history.get('debater_y', {})

    x_irac = debater_x.get('opening', {}).get('irac', {})
    y_irac = debater_y.get('opening', {}).get('irac', {})
    x_rebuttal = debater_x.get('rebuttal', {}).get('rebuttal', {})
    y_rebuttal = debater_y.get('rebuttal', {}).get('rebuttal', {})

    prompt = f"""You are an impartial legal judge reviewing a debate between two legal experts.

Question:
{full_question}

Answer choices:
{choices_text}

Debater X (defending {debater_x.get('opening', {}).get('position', '')}):
IRAC Analysis:
- Issue: {x_irac.get('issue', '')}
- Rule: {x_irac.get('rule', '')}
- Application: {x_irac.get('application', '')}
- Conclusion: {x_irac.get('conclusion', '')}
Rebuttal critique: {x_rebuttal}

Debater Y (defending {debater_y.get('opening', {}).get('position', '')}):
IRAC Analysis:
- Issue: {y_irac.get('issue', '')}
- Rule: {y_irac.get('rule', '')}
- Application: {y_irac.get('application', '')}
- Conclusion: {y_irac.get('conclusion', '')}
Rebuttal critique: {y_rebuttal}

Your task: Based on the IRAC arguments presented, select the most legally correct answer choice.
Evaluate each debater's IRAC components for accuracy and completeness.

Respond in JSON format:
{{
  "rationale": "Compare both debaters' IRAC analyses, identifying which correctly identified the issue, applied the right rule, and reached the correct conclusion",
  "winner": "debater_x, debater_y, or tie (which debater's IRAC analysis was more legally sound)",
  "decision": "A, B, C, or D (the legally correct answer)",
  "synthesis": {{
    "issue": "The key legal issue in this scenario",
    "rule": "The correct legal principle(s) that govern this issue",
    "application": "How the rule applies to these specific facts",
    "conclusion": "Why [decision] is the legally correct answer"
  }}
}}"""

    return prompt


def get_judge_decision_prompt_hybrid(
    question: str,
    prompt_context: str,
    choices: list,
    debate_history: dict
) -> str:
    """
    Generate hybrid judge decision prompt for IRAC openings + vanilla rebuttals.

    Args:
        question: The legal question
        prompt_context: Context from dataset
        choices: List of 4 answer choices
        debate_history: Debate history with IRAC openings and vanilla rebuttals

    Returns:
        Formatted prompt string with vanilla synthesis
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    # Extract debate content
    debater_x = debate_history.get('debater_x', {})
    debater_y = debate_history.get('debater_y', {})

    x_pos = debater_x.get('opening', {}).get('position', '')
    y_pos = debater_y.get('opening', {}).get('position', '')

    # Extract IRAC openings
    x_irac = debater_x.get('opening', {}).get('irac', {})
    y_irac = debater_y.get('opening', {}).get('irac', {})

    # Extract vanilla rebuttals
    x_rebuttal = debater_x.get('rebuttal', {}).get('rebuttal', '')
    y_rebuttal = debater_y.get('rebuttal', {}).get('rebuttal', '')

    prompt = f"""You are an impartial legal judge. Determine which debater presented the stronger argument.

Question: {full_question}

Choices: {choices_text}

ROUND 1 - Opening Arguments:

Debater X (answer {x_pos}):
<issue>{x_irac.get('issue', '')}</issue>
<rule>{x_irac.get('rule', '')}</rule>
<application>{x_irac.get('application', '')}</application>
<conclusion>{x_irac.get('conclusion', '')}</conclusion>

Debater Y (answer {y_pos}):
<issue>{y_irac.get('issue', '')}</issue>
<rule>{y_irac.get('rule', '')}</rule>
<application>{y_irac.get('application', '')}</application>
<conclusion>{y_irac.get('conclusion', '')}</conclusion>

ROUND 2 - Rebuttals:

Debater X: {x_rebuttal}

Debater Y: {y_rebuttal}

IMPORTANT: You must judge which debater won. Your decision MUST be either {x_pos} (if debater_x won) OR {y_pos} (if debater_y won). You CANNOT choose any other answer.

Respond in JSON:
{{
  "winner": "debater_x or debater_y",
  "decision": "{x_pos} or {y_pos} only - must match your winner",
  "rationale": "Why this debater's argument was legally stronger",
  "synthesis": "Your final explanation of why this is the legally correct answer, applying relevant legal principles and addressing the key issues in the question"
}}"""

    return prompt
