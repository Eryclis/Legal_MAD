"""
Citation extraction from Brazilian legal texts.
Extracts articles, laws, súmulas from reference answers.
"""

import re
from typing import List, Set


def extract_citations(text: str) -> List[str]:
    """
    Extract legal citations from Brazilian legal text.

    Detects:
    - Articles: "Art. 74, § 1º, CF/88"
    - Laws: "Lei 8.112/90", "Lei nº 8.112/1990"
    - Súmulas: "Súmula 473 STF", "Súmula Vinculante 13"
    - Codes: "Art. 121, CP", "Art. 186, CC"

    Args:
        text: Legal text containing citations

    Returns:
        List of normalized citations (deduplicated and sorted)
    """
    if not text:
        return []

    citations_set: Set[str] = set()

    # Pattern 1: Article + Source (Art. X, §Y, Lei/CF/CP/CC)
    # Examples: "Art. 74, § 1º, CF/88", "Artigo 121 do CP"
    pattern_art_source = r'''
        (?:Art(?:igo)?\.?\s+|art(?:igo)?\.?\s+)  # "Art." or "Artigo"
        (\d+)                                      # Article number
        (?:,?\s*§\s*(\d+º?))?                     # Optional paragraph
        (?:,?\s*(?:da|do|de)?\s*)?                # Optional connector
        (CF(?:/88)?|CRFB(?:/88)?|CP|CC|CLT|CDC|CPC|CPP|ECA|CTN)  # Source code
    '''

    for match in re.finditer(pattern_art_source, text, re.IGNORECASE | re.VERBOSE):
        article_num = match.group(1)
        paragraph = match.group(2)
        source = match.group(3).upper()

        # Normalize source
        if source in ['CF', 'CF/88', 'CRFB', 'CRFB/88']:
            source = 'CF/88'

        # Build citation
        if paragraph:
            citation = f"Art. {article_num}, § {paragraph}, {source}"
        else:
            citation = f"Art. {article_num}, {source}"

        citations_set.add(citation)

    # Pattern 2: Article + Law (Art. X da Lei Y/ZZZZ)
    # This captures "Art. 6º, inciso XXII, da Lei nº 14.133/21"
    pattern_art_lei = r'''
        (?:Art(?:igo)?\.?\s+|art(?:igo)?\.?\s+)  # "Art." or "Artigo"
        (\d+º?)                                   # Article number (may have º)
        (?:,?\s*(?:inciso|alínea)\s+([IVXivx]+|[a-z]))?  # Optional inciso/alínea
        (?:,?\s*§\s*(\d+º?))?                    # Optional paragraph
        (?:,?\s*(?:da|do|de)\s+)?                # Connector
        Lei\s+(?:nº\s*|n\.?\s*)?                 # "Lei nº"
        (\d+(?:\.\d+)?)                          # Law number
        (?:/|,?\s*de\s+)                         # Separator
        (\d{2,4})                                # Year
    '''

    for match in re.finditer(pattern_art_lei, text, re.IGNORECASE | re.VERBOSE):
        article_num = match.group(1)
        inciso = match.group(2)
        paragraph = match.group(3)
        law_num = match.group(4)
        year = match.group(5)

        # Normalize year
        if len(year) == 2:
            year = f"20{year}" if int(year) <= 50 else f"19{year}"
        elif len(year) == 4:
            year = year

        # Build detailed citation
        citation_parts = [f"Art. {article_num}"]
        if inciso:
            citation_parts.append(f"inciso {inciso.upper()}")
        if paragraph:
            citation_parts.append(f"§ {paragraph}")
        citation_parts.append(f"Lei {law_num}/{year}")

        citation = ", ".join(citation_parts)
        citations_set.add(citation)

    # Pattern 3: Standalone Laws (Lei X/YYYY without article reference)
    # We'll filter these manually to avoid conflicts
    pattern_lei_alone = r'''
        Lei\s+(?:nº\s*|n\.?\s*)?   # "Lei", "Lei nº", "Lei n."
        (\d+(?:\.\d+)?)             # Law number (e.g., 8.112)
        (?:/|,?\s*de\s+)            # Separator: "/" or "de"
        (\d{2,4})                   # Year (90 or 1990)
    '''

    # Track positions of already-matched citations to avoid duplicates
    matched_positions = set()

    # First, find all Article + Law positions
    for match in re.finditer(pattern_art_lei, text, re.IGNORECASE | re.VERBOSE):
        matched_positions.add((match.start(), match.end()))

    # Now find standalone laws, skipping overlapping regions
    for match in re.finditer(pattern_lei_alone, text, re.IGNORECASE | re.VERBOSE):
        # Check if this match overlaps with any Article + Law match
        match_start = match.start()
        match_end = match.end()

        overlaps = False
        for pos_start, pos_end in matched_positions:
            # Check if ranges overlap
            if not (match_end <= pos_start or match_start >= pos_end):
                overlaps = True
                break

        if not overlaps:
            law_num = match.group(1)
            year = match.group(2)

            # Normalize year (90 → 1990)
            if len(year) == 2:
                year = f"20{year}" if int(year) <= 50 else f"19{year}"

            citation = f"Lei {law_num}/{year}"
            citations_set.add(citation)

    # Pattern 4: Súmulas (Súmula X STF/STJ, Súmula Vinculante X)
    pattern_sumula = r'''
        Súmula\s+
        (?:(Vinculante)\s+)?      # Optional "Vinculante"
        (?:nº\s*|n\.?\s*)?        # Optional "nº"
        (\d+)                     # Súmula number
        (?:\s+(?:do\s+)?(STF|STJ))?  # Optional court
    '''

    for match in re.finditer(pattern_sumula, text, re.IGNORECASE | re.VERBOSE):
        vinculante = match.group(1)
        sumula_num = match.group(2)
        court = match.group(3)

        if vinculante:
            citation = f"Súmula Vinculante {sumula_num}"
        elif court:
            citation = f"Súmula {sumula_num} {court.upper()}"
        else:
            citation = f"Súmula {sumula_num}"

        citations_set.add(citation)

    # Pattern 5: Standalone articles with context (Art. X do Código Y)
    pattern_art_codigo = r'''
        (?:Art(?:igo)?\.?\s+|art(?:igo)?\.?\s+)
        (\d+º?)
        (?:,?\s*§\s*(\d+º?))?
        (?:\s+do\s+|\s+da\s+)
        (Código\s+(?:Civil|Penal|de\s+Processo\s+(?:Civil|Penal))|Constituição(?:\s+Federal)?)
    '''

    for match in re.finditer(pattern_art_codigo, text, re.IGNORECASE | re.VERBOSE):
        article_num = match.group(1)
        paragraph = match.group(2)
        codigo = match.group(3)

        # Map to abbreviation
        if 'Civil' in codigo and 'Processo' not in codigo:
            source = 'CC'
        elif 'Penal' in codigo and 'Processo' not in codigo:
            source = 'CP'
        elif 'Processo Civil' in codigo:
            source = 'CPC'
        elif 'Processo Penal' in codigo:
            source = 'CPP'
        elif 'Constituição' in codigo:
            source = 'CF/88'
        else:
            continue

        if paragraph:
            citation = f"Art. {article_num}, § {paragraph}, {source}"
        else:
            citation = f"Art. {article_num}, {source}"

        citations_set.add(citation)

    return sorted(list(citations_set))


# Test cases (for validation)
if __name__ == "__main__":
    test_cases = [
        # Case 1: From user's example
        "Sim, Jaqueline, como agente público responsável pelo controle interno, ao tomar conhecimento da ilegalidade por fraude contratual, deveria ter dado ciência ao Tribunal de Contas da União e, diante de sua omissão, está sujeita à responsabilidade solidária, conforme dispõe o Art. 74 § 1º, da CRFB/88.",

        # Case 2: Multiple citations
        "A conduta configura crime previsto no Art. 121 do CP, com agravante do Art. 61, II, 'a', CP. Aplica-se também a Súmula 231 STJ.",

        # Case 3: Laws
        "Conforme Lei 8.112/90, art. 127, a penalidade aplicável é demissão. Ver também Lei nº 9.784 de 1999.",

        # Case 4: Súmula Vinculante
        "Segundo Súmula Vinculante 13 do STF, é vedado nepotismo.",

        # Case 5: Article + Law with inciso (user's reported issue)
        "Sim. Nos contratos de grande vulto, assim caracterizados nos termos do Art. 6º, inciso XXII, da Lei nº 14.133/21, o edital deverá prever a obrigatoriedade de implementação de programa de integridade, consoante o Art. 25, § 4º, da Lei nº 14.133/21."
    ]

    print("="*70)
    print("CITATION PARSER - TEST SUITE")
    print("="*70)

    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test Case {i}:")
        print(f"Text: {text[:100]}...")
        print(f"\n📋 Extracted Citations:")
        citations = extract_citations(text)
        if citations:
            for cite in citations:
                print(f"  ✓ {cite}")
        else:
            print("  (No citations found)")

    print(f"\n{'='*70}")
    print("✅ Test suite completed!")
    print("="*70)
