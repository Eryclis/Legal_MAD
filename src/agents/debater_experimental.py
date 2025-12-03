"""
Debater agent for MAD system.
"""

from typing import Dict, Optional
from src.utils.api_client_experimental import GroqClient
from src.agents.prompts_experimental import get_debater_opening_prompt, get_debater_rebuttal_prompt


class Debater:
    """Debater agent that argues for a specific position."""

    def __init__(self, client: GroqClient, name: str = "Debater"):
        """
        Initialize debater.

        Args:
            client: Groq API client
            name: Debater name for logging
        """
        self.client = client
        self.name = name
        self.position = None
        self.opening_argument = None

    def generate_opening(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        position: str = None
    ) -> Dict:
        """
        Generate opening argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            position: Position to defend (A, B, C, or D). If None, debater chooses freely.

        Returns:
            Dictionary with position, argument, and citations
        """
        prompt = get_debater_opening_prompt(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            position=position
        )

        response = self.client.generate_json(prompt, max_tokens=750)

        # Validate response
        if 'position' not in response or 'argument' not in response:
            raise ValueError(f"Invalid debater response: {response}")

        # Store the chosen/assigned position
        self.position = response['position']
        self.opening_argument = response
        return response

    def generate_rebuttal(
        self,
        question: str,
        prompt_context: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate rebuttal argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            opponent_opening: Opponent's opening argument

        Returns:
            Dictionary with rebuttal, counterarguments, and citations
        """
        if not self.opening_argument:
            raise ValueError("Must generate opening argument before rebuttal")

        prompt = get_debater_rebuttal_prompt(
            question=question,
            prompt_context=prompt_context,
            my_position=self.position,
            my_opening=self.opening_argument,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=650)

        # Validate response
        if 'rebuttal' not in response:
            raise ValueError(f"Invalid rebuttal response: {response}")

        return response

    # ==================== IRAC METHODS ====================

    def generate_opening_irac(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        position: str = None
    ) -> Dict:
        """
        Generate IRAC-structured opening argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            position: Position to defend (A, B, C, or D). If None, debater chooses freely.

        Returns:
            Dictionary with position, irac structure, and citations
        """
        from src.agents.prompts import get_debater_opening_prompt_irac

        prompt = get_debater_opening_prompt_irac(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            position=position
        )

        response = self.client.generate_json(prompt, max_tokens=900)

        # Validate response
        if 'position' not in response or 'irac' not in response:
            raise ValueError(f"Invalid IRAC debater response: {response}")

        # Validate IRAC structure
        irac = response.get('irac', {})
        required_keys = ['issue', 'rule', 'application', 'conclusion']
        for key in required_keys:
            if key not in irac:
                raise ValueError(f"Missing IRAC component '{key}': {response}")

        # Store the chosen/assigned position
        self.position = response['position']
        self.opening_argument = response
        return response

    def generate_rebuttal_irac(
        self,
        question: str,
        prompt_context: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate IRAC-structured rebuttal argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            opponent_opening: Opponent's IRAC opening argument

        Returns:
            Dictionary with structured rebuttal and citations
        """
        if not self.opening_argument:
            raise ValueError("Must generate opening argument before rebuttal")

        from src.agents.prompts import get_debater_rebuttal_prompt_irac

        prompt = get_debater_rebuttal_prompt_irac(
            question=question,
            prompt_context=prompt_context,
            my_position=self.position,
            my_opening=self.opening_argument,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=700)

        # Validate response
        if 'rebuttal' not in response:
            raise ValueError(f"Invalid IRAC rebuttal response: {response}")

        # Validate rebuttal structure
        rebuttal = response.get('rebuttal', {})
        expected_keys = ['issue_critique', 'rule_critique', 'application_critique', 'my_reinforcement']
        for key in expected_keys:
            if key not in rebuttal:
                raise ValueError(f"Missing rebuttal component '{key}': {response}")

        return response
