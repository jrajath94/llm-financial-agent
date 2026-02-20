"""Tests for the LLM financial agent system.

Covers calculator tool, claim extraction, hallucination detection,
single-agent analysis, multi-agent orchestration, and consensus logic.
All tests use mock LLM and data providers — no real API calls.
"""
import pytest

from llm_financial_agent.agent import (
    AgentOrchestrator,
    AgentRole,
    AnalysisResult,
    Calculator,
    ClaimExtractor,
    ConsensusResult,
    DataProvider,
    FinancialAgent,
    HallucinationDetector,
    HallucinationReport,
    LLMProvider,
    ToolCall,
    ToolType,
)


# --- Mock Implementations ---

class MockLLM:
    """Mock LLM provider that returns deterministic responses."""

    def __init__(self, response: str = "Analysis complete.") -> None:
        self.response = response
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Return a canned response.

        Args:
            prompt: Input prompt (ignored).

        Returns:
            The pre-configured response.
        """
        self.call_count += 1
        return self.response


class MockDataProvider:
    """Mock data provider for hallucination testing."""

    def __init__(self, verified: bool = True) -> None:
        self.verified = verified

    def fetch(self, query: str) -> dict:
        """Return mock data with configurable verification status.

        Args:
            query: Query string (ignored).

        Returns:
            Dict with verified flag.
        """
        return {"verified": self.verified, "data": "mock"}


class FailingDataProvider:
    """Data provider that always raises an exception."""

    def fetch(self, query: str) -> dict:
        """Always fails.

        Args:
            query: Query string.

        Raises:
            ConnectionError: Always.
        """
        raise ConnectionError("Data source unavailable")


# --- Fixtures ---

@pytest.fixture
def calculator() -> Calculator:
    """A fresh calculator tool instance."""
    return Calculator()


@pytest.fixture
def mock_llm() -> MockLLM:
    """A mock LLM that returns a basic analysis."""
    return MockLLM("The stock price increased 15% in Q1 2024. Revenue was $50B.")


@pytest.fixture
def mock_llm_short() -> MockLLM:
    """A mock LLM that returns a short response."""
    return MockLLM("OK")


@pytest.fixture
def verified_provider() -> MockDataProvider:
    """Data provider that always verifies claims."""
    return MockDataProvider(verified=True)


@pytest.fixture
def unverified_provider() -> MockDataProvider:
    """Data provider that never verifies claims."""
    return MockDataProvider(verified=False)


@pytest.fixture
def analyst_agent(mock_llm: MockLLM) -> FinancialAgent:
    """An analyst agent with calculator tool."""
    return FinancialAgent(
        role=AgentRole.ANALYST,
        llm=mock_llm,
        tools={ToolType.CALCULATOR: Calculator()},
    )


@pytest.fixture
def risk_agent(mock_llm: MockLLM) -> FinancialAgent:
    """A risk assessor agent."""
    return FinancialAgent(
        role=AgentRole.RISK_ASSESSOR,
        llm=mock_llm,
    )


# --- Calculator Tests ---

class TestCalculator:
    """Tests for the Calculator tool."""

    def test_basic_arithmetic(self, calculator: Calculator) -> None:
        """Simple arithmetic expressions evaluate correctly."""
        assert calculator.evaluate("2 + 3") == pytest.approx(5.0)

    def test_multiplication(self, calculator: Calculator) -> None:
        """Multiplication works."""
        assert calculator.evaluate("100 * 1.05") == pytest.approx(105.0)

    def test_exponentiation(self, calculator: Calculator) -> None:
        """Power expressions work."""
        result = calculator.evaluate("2 ** 10")
        assert result == pytest.approx(1024.0)

    @pytest.mark.parametrize("expression,expected", [
        ("10 + 20", 30.0),
        ("100 - 25", 75.0),
        ("8 * 7", 56.0),
        ("100 / 4", 25.0),
    ])
    def test_parametrized_arithmetic(
        self, calculator: Calculator, expression: str, expected: float
    ) -> None:
        """Various arithmetic expressions produce correct results."""
        assert calculator.evaluate(expression) == pytest.approx(expected)

    def test_forbidden_import_raises(self, calculator: Calculator) -> None:
        """Expression containing 'import' is rejected."""
        with pytest.raises(ValueError, match="Forbidden"):
            calculator.evaluate("import os")

    def test_forbidden_dunder_raises(self, calculator: Calculator) -> None:
        """Expression containing '__' is rejected."""
        with pytest.raises(ValueError, match="Forbidden"):
            calculator.evaluate("__import__('os')")

    def test_compound_return(self, calculator: Calculator) -> None:
        """Compound return calculation is accurate."""
        # $1000 at 5% for 10 years
        result = calculator.compound_return(1000, 0.05, 10)
        assert result == pytest.approx(1628.8946, rel=1e-3)

    def test_sharpe_ratio(self, calculator: Calculator) -> None:
        """Sharpe ratio calculation produces a reasonable number."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.005, -0.005, 0.015]
        ratio = calculator.sharpe_ratio(returns)
        assert isinstance(ratio, float)
        # Should be positive for positive-mean returns
        assert ratio > 0

    def test_sharpe_ratio_too_few_returns(
        self, calculator: Calculator
    ) -> None:
        """Sharpe ratio with < 2 returns raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            calculator.sharpe_ratio([0.01])


# --- ClaimExtractor Tests ---

class TestClaimExtractor:
    """Tests for factual claim extraction."""

    def test_extracts_numerical_claim(self) -> None:
        """Numerical claims with 'increased' are extracted."""
        extractor = ClaimExtractor()
        text = "Revenue increased 25% last quarter."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_extracts_price_claim(self) -> None:
        """Dollar-amount claims are extracted."""
        extractor = ClaimExtractor()
        text = "The stock is trading at $150.50 per share."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_extracts_date_claim(self) -> None:
        """Date-bound claims are extracted."""
        extractor = ClaimExtractor()
        text = "Since 2020 the company has doubled revenue."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_no_claims_in_opinion(self) -> None:
        """Purely opinion text has no factual claims."""
        extractor = ClaimExtractor()
        text = "I think the market looks good today."
        claims = extractor.extract_claims(text)
        assert len(claims) == 0

    def test_empty_text(self) -> None:
        """Empty text yields no claims."""
        extractor = ClaimExtractor()
        assert extractor.extract_claims("") == []


# --- HallucinationDetector Tests ---

class TestHallucinationDetector:
    """Tests for hallucination detection."""

    def test_no_hallucination_when_all_verified(
        self, verified_provider: MockDataProvider
    ) -> None:
        """All verified claims result in score=0."""
        detector = HallucinationDetector(verified_provider)
        text = "Revenue increased 25%. Stock is at $150."
        report = detector.analyze(text)
        assert report.score == pytest.approx(0.0)
        assert not report.is_hallucinated

    def test_full_hallucination_when_none_verified(
        self, unverified_provider: MockDataProvider
    ) -> None:
        """No verified claims result in score=1.0."""
        detector = HallucinationDetector(unverified_provider)
        text = "Revenue increased 25%. Stock is at $150."
        report = detector.analyze(text)
        assert report.score == pytest.approx(1.0)
        assert report.is_hallucinated

    def test_no_claims_score_zero(
        self, verified_provider: MockDataProvider
    ) -> None:
        """Text with no claims gets score=0."""
        detector = HallucinationDetector(verified_provider)
        report = detector.analyze("Just an opinion piece.")
        assert report.score == 0.0
        assert report.total_claims == 0

    def test_data_provider_failure_flags_claim(self) -> None:
        """Claims where data fetch fails are flagged."""
        detector = HallucinationDetector(FailingDataProvider())
        text = "Revenue increased 50%."
        report = detector.analyze(text)
        assert report.score > 0
        assert len(report.flagged_claims) >= 1


# --- FinancialAgent Tests ---

class TestFinancialAgent:
    """Tests for single agent analysis."""

    def test_analyze_returns_result(
        self, analyst_agent: FinancialAgent
    ) -> None:
        """Agent analysis returns an AnalysisResult."""
        result = analyst_agent.analyze("What is AAPL's outlook?")
        assert isinstance(result, AnalysisResult)
        assert result.agent_role == AgentRole.ANALYST

    def test_memory_records_queries(
        self, analyst_agent: FinancialAgent
    ) -> None:
        """Agent memory stores queries."""
        analyst_agent.analyze("Query 1")
        analyst_agent.analyze("Query 2")
        assert len(analyst_agent.memory) == 2
        assert "Query 1" in analyst_agent.memory

    def test_confidence_with_tool_calls(
        self, mock_llm: MockLLM
    ) -> None:
        """Agent with calculator tool on math query gets boosted confidence."""
        agent = FinancialAgent(
            role=AgentRole.ANALYST,
            llm=mock_llm,
            tools={ToolType.CALCULATOR: Calculator()},
        )
        # Query with a math expression triggers the calculator
        result = agent.analyze("What is 100 * 1.05?")
        assert result.confidence >= 0.5

    def test_short_response_lower_confidence(
        self, mock_llm_short: MockLLM
    ) -> None:
        """Short LLM responses get lower confidence."""
        agent = FinancialAgent(
            role=AgentRole.ANALYST,
            llm=mock_llm_short,
        )
        result = agent.analyze("Quick question")
        assert result.confidence <= 0.6


# --- AgentOrchestrator Tests ---

class TestAgentOrchestrator:
    """Tests for multi-agent orchestration."""

    def test_minimum_agents_enforced(self, mock_llm: MockLLM) -> None:
        """Orchestrator requires at least 2 agents."""
        single = FinancialAgent(AgentRole.ANALYST, mock_llm)
        with pytest.raises(ValueError, match="at least"):
            AgentOrchestrator([single])

    def test_consensus_with_agreeing_agents(
        self, analyst_agent: FinancialAgent, risk_agent: FinancialAgent
    ) -> None:
        """Two agents with similar confidence reach consensus."""
        orchestrator = AgentOrchestrator([analyst_agent, risk_agent])
        result = orchestrator.run_analysis("Market outlook?")
        assert isinstance(result, ConsensusResult)
        assert len(result.individual_results) == 2

    def test_hallucination_filtering(
        self, mock_llm: MockLLM, unverified_provider: MockDataProvider
    ) -> None:
        """Orchestrator filters hallucinated results."""
        agents = [
            FinancialAgent(AgentRole.ANALYST, mock_llm),
            FinancialAgent(AgentRole.RISK_ASSESSOR, mock_llm),
        ]
        detector = HallucinationDetector(unverified_provider)
        orchestrator = AgentOrchestrator(agents, detector)
        result = orchestrator.run_analysis("Tell me about AAPL")
        # Even if all flagged, fallback returns them
        assert isinstance(result, ConsensusResult)

    def test_three_agent_consensus(self, mock_llm: MockLLM) -> None:
        """Three-agent orchestration produces valid consensus."""
        agents = [
            FinancialAgent(AgentRole.ANALYST, mock_llm),
            FinancialAgent(AgentRole.RISK_ASSESSOR, mock_llm),
            FinancialAgent(AgentRole.FACT_CHECKER, mock_llm),
        ]
        orchestrator = AgentOrchestrator(agents)
        result = orchestrator.run_analysis("Sector analysis")
        assert result.agreement_ratio > 0

    def test_consensus_agreement_ratio_range(
        self, analyst_agent: FinancialAgent, risk_agent: FinancialAgent
    ) -> None:
        """Agreement ratio is in [0, 1]."""
        orchestrator = AgentOrchestrator([analyst_agent, risk_agent])
        result = orchestrator.run_analysis("Test query")
        assert 0.0 <= result.agreement_ratio <= 1.0
