"""
LLM-as-Judge evaluation for OAB legal reasoning questions.
Uses GPT-4o-mini via OpenRouter to avoid bias (different from experiment models).
"""

import json
from typing import Dict, Optional
from src.utils.api_client_experimental import OpenRouterClient


def evaluate_with_llm_judge(
    prediction: str,
    reference: str,
    question: str,
    client: Optional[OpenRouterClient] = None
) -> Dict[str, float]:
    """
    Evaluate a legal answer using LLM-as-Judge approach.

    Uses GPT-4o-mini via OpenRouter to avoid bias (not used in experiments).

    Args:
        prediction: Model-generated answer
        reference: Ground truth reference answer
        question: Original question text
        client: Optional OpenRouterClient instance (creates new if None)

    Returns:
        Dictionary with scores:
        {
            'correctness': 0-4 (legal correctness),
            'reasoning': 0-3 (logical reasoning quality),
            'citations': 0-4 (citation accuracy and completeness),
            'total': 0-11 (sum of all scores),
            'normalized': 0-1 (total/11),
            'justification': str (brief explanation)
        }
    """
    if not prediction or not reference:
        return {
            'correctness': 0.0,
            'reasoning': 0.0,
            'citations': 0.0,
            'total': 0.0,
            'normalized': 0.0,
            'justification': 'Empty prediction or reference'
        }

    # Create OpenRouter client if not provided (GPT-4o-mini)
    if client is None:
        client = OpenRouterClient(
            model="openai/gpt-4o-mini",
            temperature=0.1,
            max_tokens=500
        )

    # Build evaluation prompt with XML structure
    prompt = f"""Você é um avaliador especializado em questões jurídicas do Exame da OAB (Ordem dos Advogados do Brasil).

<task>
Avalie a qualidade de uma resposta jurídica comparada com a resposta de referência (espelho de correção oficial da OAB).
</task>

<question>
{question}
</question>

<reference_answer>
{reference}
</reference_answer>

<candidate_answer>
{prediction}
</candidate_answer>

<evaluation_criteria>
Avalie a RESPOSTA DO CANDIDATO segundo os seguintes critérios:

1. CORREÇÃO JURÍDICA (0-4 pontos):
   0 = Completamente incorreta ou irrelevante
   1 = Parcialmente correta, mas com erros graves de fundamentação
   2 = Correta mas incompleta ou superficial
   3 = Correta e completa
   4 = Correta, completa e excepcionalmente bem fundamentada

2. RACIOCÍNIO JURÍDICO (0-3 pontos):
   0 = Sem lógica jurídica ou raciocínio incoerente
   1 = Raciocínio básico presente
   2 = Raciocínio claro e estruturado
   3 = Raciocínio excelente, estruturado (ex: IRAC ou similar)

3. CITAÇÕES LEGAIS (0-4 pontos):
   0 = Nenhuma citação ou citação completamente errada
   1 = Citou legislação/código correto mas artigo/dispositivo errado
   2 = Citou artigo próximo ou relacionado ao correto
   3 = Citou artigo correto mas faltou complementos (parágrafos, incisos)
   4 = Citação perfeita e completa
</evaluation_criteria>

<instructions>
IMPORTANTE:
- Compare cuidadosamente a resposta do candidato com a resposta de referência
- Seja rigoroso mas justo na avaliação
- Considere que respostas podem ser corretas mesmo usando palavras diferentes
- A justificativa deve ser objetiva e técnica (máximo 2 frases)

Retorne APENAS um JSON válido no seguinte formato:
{{
    "correctness": <número de 0 a 4>,
    "reasoning": <número de 0 a 3>,
    "citations": <número de 0 a 4>,
    "justification": "<explicação breve e objetiva>"
}}
</instructions>"""

    try:
        # Get LLM evaluation with JSON mode
        response = client.generate_json(
            prompt=prompt,
            temperature=0.1,
            max_tokens=500
        )

        # Validate and extract scores
        correctness = float(response.get('correctness', 0))
        reasoning = float(response.get('reasoning', 0))
        citations = float(response.get('citations', 0))
        justification = response.get('justification', '')

        # Clamp scores to valid ranges
        correctness = max(0.0, min(4.0, correctness))
        reasoning = max(0.0, min(3.0, reasoning))
        citations = max(0.0, min(4.0, citations))

        total = correctness + reasoning + citations
        normalized = total / 11.0  # Max possible: 11 points

        return {
            'correctness': round(correctness, 2),
            'reasoning': round(reasoning, 2),
            'citations': round(citations, 2),
            'total': round(total, 2),
            'normalized': round(normalized, 4),
            'justification': justification
        }

    except Exception as e:
        print(f"Warning: LLM-as-Judge evaluation failed: {e}")
        return {
            'correctness': 0.0,
            'reasoning': 0.0,
            'citations': 0.0,
            'total': 0.0,
            'normalized': 0.0,
            'justification': f'Error: {str(e)}'
        }


# Test case
if __name__ == "__main__":
    print("="*70)
    print("LLM-AS-JUDGE - TEST SUITE")
    print("="*70)

    question = """QUESTÃO

A União fez publicar um edital de licitação, na modalidade concorrência, para uma grande obra de infraestrutura, inicialmente orçada em R$300.000.000,00 (trezentos milhões de reais), a caracterizar, portanto, um contrato de grande vulto nos termos da nova Lei de Licitações, aplicável à hipótese.

O edital em questão deveria contemplar a matriz de alocação de riscos entre contratante e contratado? Justifique."""

    reference = """Sim. Nos contratos de grande vulto, assim caracterizados nos termos do Art. 6º, inciso XXII, da Lei nº 14.133/21, o edital deverá prever a obrigatoriedade de implementação de programa de integridade, consoante o Art. 25, § 4º, da Lei nº 14.133/21."""

    prediction = """Sim, o edital em questão deve contemplar a matriz de alocação de riscos entre a União e a contratada. Isso porque a obra de infraestrutura tem um orçamento de R$300.000.000,00, caracterizando um contrato de grande vulto, conforme definido pela Lei nº 14.133/21, art. 6º, inciso XXII. A matriz de alocação de riscos é fundamental para garantir a transparência e a clareza das responsabilidades entre as partes, conforme Art. 25, § 4º, da Lei nº 14.133/21."""

    print(f"\nQuestion: {question[:100]}...")
    print(f"\nReference: {reference[:100]}...")
    print(f"\nPrediction: {prediction[:100]}...")

    try:
        result = evaluate_with_llm_judge(prediction, reference, question)
        print(f"\nResults:")
        print(f"  Correctness: {result['correctness']}/4")
        print(f"  Reasoning:   {result['reasoning']}/3")
        print(f"  Citations:   {result['citations']}/4")
        print(f"  Total:       {result['total']}/11")
        print(f"  Normalized:  {result['normalized']:.2%}")
        print(f"\n  Justification: {result.get('justification', 'N/A')}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Test suite completed!")
    print("="*70)
