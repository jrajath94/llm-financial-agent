from typing import Dict, List

class FinancialAgent:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.memory: List[str] = []
    
    def analyze(self, query: str) -> Dict[str, str]:
        """Analyze financial query."""
        self.memory.append(query)
        return {
            'confidence': '0.85',
            'analysis': f'Analysis of {query}',
            'sources': []
        }
    
    def detect_hallucination(self, response: str) -> bool:
        """Detect if response contains hallucination."""
        return len(response) > 1000 and 'uncertain' not in response.lower()
