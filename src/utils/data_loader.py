"""
Data loader for legal QA datasets.
"""

from typing import List, Dict, Optional
import pandas as pd
from huggingface_hub import hf_hub_download


def load_bar_exam_qa(sample_size: Optional[int] = None, split: str = "train") -> List[Dict]:
    """
    Load Bar Exam QA dataset from HuggingFace (reglab/barexam_qa).

    Downloads CSV files directly from the repository to avoid the deprecated
    loading script system.

    Args:
        sample_size: Number of questions to sample (None = all)
        split: Dataset split to load ('train', 'validation', or 'test')

    Returns:
        List of question dictionaries with structure:
        {
            'id': str,
            'prompt': str,
            'question': str,
            'choices': List[str],
            'answer': str,
            'gold_passage': str,
            'gold_idx': str
        }
    """
    print(f"Loading Bar Exam QA dataset ({split} split)...")

    # Download the CSV file from HuggingFace Hub
    file_path = hf_hub_download(
        repo_id="reglab/barexam_qa",
        filename=f"data/qa/{split}.csv",
        repo_type="dataset"
    )

    # Load CSV with pandas
    df = pd.read_csv(file_path)

    questions = []
    for idx, row in df.iterrows():
        if sample_size and len(questions) >= sample_size:
            break

        question_dict = {
            'id': str(row.get('idx', idx)),
            'prompt': str(row.get('prompt', '')),
            'question': str(row['question']),
            'choices': [
                str(row['choice_a']),
                str(row['choice_b']),
                str(row['choice_c']),
                str(row['choice_d'])
            ],
            'answer': str(row['answer']),
            'gold_passage': str(row.get('gold_passage', '')),
            'gold_idx': str(row.get('gold_idx', ''))
        }

        questions.append(question_dict)

    print(f"Loaded {len(questions)} questions from Bar Exam QA ({split} split)")
    return questions


def load_oab_open_ended(file_path: str) -> List[Dict]:
    """
    Load OAB open-ended questions from local file.

    Args:
        file_path: Path to OAB dataset file

    Returns:
        List of question dictionaries
    """
    raise NotImplementedError("OAB data loader will be implemented after data collection")


if __name__ == "__main__":
    # Test data loading
    questions = load_bar_exam_qa(sample_size=5)

    print("\nSample question:")
    q = questions[0]
    print(f"ID: {q['id']}")
    print(f"Prompt: {q['prompt'][:100] if q['prompt'] else 'None'}...")
    print(f"Question: {q['question'][:100]}...")
    print(f"Choices: A) {q['choices'][0][:50]}...")
    print(f"Answer: {q['answer']}")
    print(f"Gold passage length: {len(q['gold_passage'])} chars")
