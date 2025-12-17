"""
Automatic evaluation metrics for OAB open-ended questions.
Includes Citation F1 and BERTScore (Portuguese).
"""

from typing import List, Dict, Tuple
import warnings

# Suppress warnings from transformers/bert-score
warnings.filterwarnings('ignore')


def citation_f1(predicted: List[str], expected: List[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1-score for legal citations.

    Args:
        predicted: List of citations from model response
        expected: List of citations from ground truth

    Returns:
        Dictionary with precision, recall, f1, and counts
    """
    if not predicted and not expected:
        # Both empty - perfect match
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'predicted_count': 0,
            'expected_count': 0,
            'matched_count': 0
        }

    if not predicted or not expected:
        # One is empty - zero scores
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'predicted_count': len(predicted),
            'expected_count': len(expected),
            'matched_count': 0
        }

    # Convert to sets for matching
    pred_set = set(predicted)
    exp_set = set(expected)

    # Calculate intersection
    matched = pred_set & exp_set

    # Calculate metrics
    precision = len(matched) / len(pred_set) if pred_set else 0.0
    recall = len(matched) / len(exp_set) if exp_set else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'predicted_count': len(predicted),
        'expected_count': len(expected),
        'matched_count': len(matched)
    }


def bertscore_portuguese(prediction: str, reference: str, model_name: str = "neuralmind/bert-base-portuguese-cased") -> Dict[str, float]:
    """
    Calculate BERTScore using BERTimbau (Portuguese BERT model).

    Uses bert-score library with custom HuggingFace model for efficiency.
    The model is cached after first download, making subsequent calls fast.

    Args:
        prediction: Model-generated answer
        reference: Ground truth reference answer
        model_name: HuggingFace model name (default: BERTimbau)

    Returns:
        Dictionary with precision, recall, f1 scores
    """
    try:
        from bert_score import score
    except ImportError:
        raise ImportError(
            "bert-score not installed. Run: pip install bert-score"
        )

    if not prediction or not reference:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    # Calculate BERTScore using BERTimbau
    # bert-score accepts custom HuggingFace models and caches them
    P, R, F1 = score(
        [prediction],
        [reference],
        model_type=model_name,
        num_layers=12,  # BERTimbau has 12 layers
        verbose=False,
        device='cpu',
        lang='pt',
        rescale_with_baseline=False  # No baseline rescaling for stability
    )

    return {
        'precision': round(P.item(), 4),
        'recall': round(R.item(), 4),
        'f1': round(F1.item(), 4)
    }


def evaluate_single_result(result: Dict) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single question result with all metrics.

    Args:
        result: Result dictionary from experiment with structure:
            {
                'question_id': str,
                'judge': {'final_answer': str, 'key_citations': [...]},
                'answer': str (for baselines),
                'key_citations': [...] (for baselines),
                'ground_truth': {
                    'reference_answer': str,
                    'key_citations_expected': [...]
                }
            }

    Returns:
        Dictionary with all metrics:
            {
                'citation_f1': {...},
                'bertscore': {...}
            }
    """
    # Extract data based on result type (MAD or Baseline)
    if 'judge' in result:
        # MAD result
        predicted_answer = result['judge'].get('final_answer', '')
        predicted_citations = result['judge'].get('key_citations', [])
    else:
        # Baseline result
        predicted_answer = result.get('answer', '')
        predicted_citations = result.get('key_citations', [])

    # Extract ground truth
    ground_truth = result.get('ground_truth', {})
    reference_answer = ground_truth.get('reference_answer', '')
    expected_citations = ground_truth.get('key_citations_expected', [])

    # Calculate metrics
    metrics = {}

    # 1. Citation F1
    metrics['citation_f1'] = citation_f1(predicted_citations, expected_citations)

    # 2. BERTScore
    try:
        metrics['bertscore'] = bertscore_portuguese(predicted_answer, reference_answer)
    except Exception as e:
        print(f"Warning: BERTScore failed for question {result.get('question_id', 'unknown')}: {e}")
        metrics['bertscore'] = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across all questions.

    Args:
        all_metrics: List of metric dictionaries from evaluate_single_result

    Returns:
        Aggregated metrics (mean across all questions)
    """
    if not all_metrics:
        return {}

    # Check if LLM-as-Judge is present
    has_llm_judge = 'llm_judge' in all_metrics[0]

    # Initialize aggregators
    aggregated = {
        'citation_f1': {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        },
        'bertscore': {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    }

    if has_llm_judge:
        aggregated['llm_judge'] = {
            'correctness': 0.0,
            'reasoning': 0.0,
            'citations': 0.0,
            'total': 0.0,
            'normalized': 0.0
        }

    # Sum all metrics
    n = len(all_metrics)
    for metrics in all_metrics:
        for metric_type in ['citation_f1', 'bertscore']:
            for score_type in ['precision', 'recall', 'f1']:
                aggregated[metric_type][score_type] += metrics[metric_type][score_type]

        if has_llm_judge and 'llm_judge' in metrics:
            for score_type in ['correctness', 'reasoning', 'citations', 'total', 'normalized']:
                aggregated['llm_judge'][score_type] += metrics['llm_judge'][score_type]

    # Calculate mean
    for metric_type in ['citation_f1', 'bertscore']:
        for score_type in ['precision', 'recall', 'f1']:
            aggregated[metric_type][score_type] = round(
                aggregated[metric_type][score_type] / n, 4
            )

    if has_llm_judge:
        for score_type in ['correctness', 'reasoning', 'citations', 'total', 'normalized']:
            aggregated['llm_judge'][score_type] = round(
                aggregated['llm_judge'][score_type] / n, 4
            )

    return aggregated


# Test cases
if __name__ == "__main__":
    print("="*70)
    print("AUTOMATIC METRICS - TEST SUITE")
    print("="*70)

    # Test 1: Citation F1
    print("\n" + "─"*70)
    print("Test 1: Citation F1")
    print("─"*70)

    predicted = ["Art. 74, § 1º, CF/88", "Lei 8.112/1990"]
    expected = ["Art. 74, § 1º, CF/88", "Lei 9.784/1999"]

    result = citation_f1(predicted, expected)
    print(f"Predicted: {predicted}")
    print(f"Expected:  {expected}")
    print(f"\nResults:")
    print(f"  Precision: {result['precision']:.2%}")
    print(f"  Recall:    {result['recall']:.2%}")
    print(f"  F1:        {result['f1']:.2%}")
    print(f"  Matched:   {result['matched_count']}/{result['predicted_count']}")

    # Test 2: BERTScore
    print("\n" + "─"*70)
    print("Test 2: BERTScore (Portuguese)")
    print("─"*70)

    prediction = "Sim, Jaqueline deveria ter dado ciência ao TCU, conforme Art. 74, § 1º, CF/88."
    reference = "Sim, Jaqueline, como agente público responsável pelo controle interno, deveria ter dado ciência ao Tribunal de Contas da União, conforme Art. 74 § 1º, da CRFB/88."

    print(f"Prediction: {prediction[:80]}...")
    print(f"Reference:  {reference[:80]}...")

    try:
        result = bertscore_portuguese(prediction, reference)
        print(f"\nResults:")
        print(f"  Precision: {result['precision']:.2%}")
        print(f"  Recall:    {result['recall']:.2%}")
        print(f"  F1:        {result['f1']:.2%}")
    except ImportError as e:
        print(f"\nWarning: {e}")
        print("   Install with: pip install bert-score")

    print("\n" + "="*70)
    print("Test suite completed!")
    print("="*70)
