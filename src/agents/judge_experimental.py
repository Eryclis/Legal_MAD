"""
Judge agent for MAD system.
"""

from typing import Dict
from src.utils.api_client_experimental import GroqClient
from src.agents.prompts_experimental import get_judge_decision_prompt


class Judge:
    """Judge agent that synthesizes debate and makes final decision."""

    def __init__(self, client: GroqClient):
        """
        Initialize judge.

        Args:
            client: Groq API client
        """
        self.client = client

    def make_decision(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        debate_history: Dict
    ) -> Dict:
        """
        Make final decision based on debate.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            debate_history: Full debate history with structure:
                {
                    'debater_x': {'opening': {...}, 'rebuttal': {...}},
                    'debater_y': {'opening': {...}, 'rebuttal': {...}}
                }

        Returns:
            Dictionary with decision and rationale
        """
        prompt = get_judge_decision_prompt(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            debate_history=debate_history
        )

        response = self.client.generate_json(prompt, max_tokens=800)

        # Validate response
        if 'decision' not in response:
            raise ValueError(f"Invalid judge response: {response}")

        # Ensure decision is valid choice
        if response['decision'] not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid decision choice: {response['decision']}")

        return response

    # ==================== IRAC METHOD ====================

    def make_decision_irac(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        debate_history: Dict
    ) -> Dict:
        """
        Make final decision based on IRAC-structured debate.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            debate_history: Full debate history with IRAC structure:
                {
                    'debater_x': {'opening': {'position': ..., 'irac': {...}}, 'rebuttal': {...}},
                    'debater_y': {'opening': {'position': ..., 'irac': {...}}, 'rebuttal': {...}}
                }

        Returns:
            Dictionary with decision and IRAC-structured synthesis
        """
        from src.agents.prompts_experimental import get_judge_decision_prompt_irac

        prompt = get_judge_decision_prompt_irac(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            debate_history=debate_history
        )

        response = self.client.generate_json(prompt, max_tokens=1000)

        # Validate response
        if 'decision' not in response:
            raise ValueError(f"Invalid judge IRAC response: {response}")

        # Ensure decision is valid choice
        if response['decision'] not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid decision choice: {response['decision']}")

        # Validate synthesis IRAC structure
        if 'synthesis' not in response:
            raise ValueError(f"Missing synthesis in judge IRAC response: {response}")

        synthesis = response.get('synthesis', {})
        required_keys = ['issue', 'rule', 'application', 'conclusion']
        for key in required_keys:
            if key not in synthesis:
                raise ValueError(f"Missing synthesis IRAC component '{key}': {response}")

        return response

    def make_decision_hybrid(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        debate_history: Dict
    ) -> Dict:
        """
        Make final decision based on hybrid debate (IRAC openings + vanilla rebuttals).

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            debate_history: Debate history with structure:
                {
                    'debater_x': {'opening': {'position': ..., 'irac': {...}}, 'rebuttal': {...}},
                    'debater_y': {'opening': {'position': ..., 'irac': {...}}, 'rebuttal': {...}}
                }

        Returns:
            Dictionary with decision and vanilla synthesis (string)
        """
        from src.agents.prompts_experimental import get_judge_decision_prompt_hybrid

        prompt = get_judge_decision_prompt_hybrid(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            debate_history=debate_history
        )

        response = self.client.generate_json(prompt, max_tokens=1000)

        # Validate response
        if 'decision' not in response:
            raise ValueError(f"Invalid judge hybrid response: {response}")

        # Ensure decision is valid choice
        if response['decision'] not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid decision choice: {response['decision']}")

        # Validate decision matches winner's position
        winner = response.get('winner', '')
        debater_x_pos = debate_history.get('debater_x', {}).get('opening', {}).get('position', '')
        debater_y_pos = debate_history.get('debater_y', {}).get('opening', {}).get('position', '')

        if winner == 'debater_x' and response['decision'] != debater_x_pos:
            raise ValueError(
                f"Decision {response['decision']} doesn't match winner debater_x's position {debater_x_pos}"
            )
        elif winner == 'debater_y' and response['decision'] != debater_y_pos:
            raise ValueError(
                f"Decision {response['decision']} doesn't match winner debater_y's position {debater_y_pos}"
            )

        return response
