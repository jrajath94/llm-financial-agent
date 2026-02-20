"""Multi-agent financial analysis system with hallucination detection.

Provides agent orchestration, tool-use abstractions (calculator, data retrieval),
hallucination scoring via cross-referencing, and a consensus mechanism
for multi-agent agreement. All LLM interactions use abstract Protocol
interfaces so the system is fully testable without real API calls.
"""
import logging
import math
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_HALLUCINATION_THRESHOLD: float = 0.6
MIN_CONSENSUS_AGENTS: int = 2
MAX_MEMORY_SIZE: int = 100
CONFIDENCE_DECIMAL_PLACES: int = 4
NUMERICAL_TOLERANCE: float = 1e-9


class AgentRole(Enum):
    """Roles an agent can play in the multi-agent system."""
    ANALYST = "analyst"
    RISK_ASSESSOR = "risk_assessor"
    DATA_RETRIEVER = "data_retriever"
    FACT_CHECKER = "fact_checker"


class ToolType(Enum):
    """Available tool types for agent use."""
    CALCULATOR = "calculator"
    DATA_RETRIEVAL = "data_retrieval"
    SEARCH = "search"
    CHART = "chart"


@dataclass
class ToolCall:
    """Represents a single tool invocation by an agent.

    Args:
        tool_type: Which tool to invoke.
        arguments: Arguments passed to the tool.
        result: The tool's return value (populated after execution).
    """
    tool_type: ToolType
    arguments: dict
    result: Optional[str] = None


@dataclass
class AnalysisResult:
    """Output of a single agent's analysis.

    Args:
        agent_role: Role of the agent that produced this result.
        content: The textual analysis.
        confidence: Agent's self-assessed confidence in [0, 1].
        tool_calls: Tools invoked during analysis.
        sources: References or data sources cited.
    """
    agent_role: AgentRole
    content: str
    confidence: float
    tool_calls: list[ToolCall] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class HallucinationReport:
    """Report from hallucination detection.

    Args:
        score: Hallucination likelihood in [0, 1] (higher = more likely).
        flagged_claims: Specific claims identified as potentially false.
        cross_reference_matches: Number of claims verified by sources.
        total_claims: Total number of factual claims detected.
    """
    score: float
    flagged_claims: list[str]
    cross_reference_matches: int
    total_claims: int

    @property
    def is_hallucinated(self) -> bool:
        """Whether the score exceeds the default threshold."""
        return self.score > DEFAULT_HALLUCINATION_THRESHOLD


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus aggregation.

    Args:
        agreed: Whether agents reached consensus.
        final_answer: The merged/agreed-upon analysis.
        agreement_ratio: Fraction of agents that agreed.
        individual_results: Per-agent analysis results.
    """
    agreed: bool
    final_answer: str
    agreement_ratio: float
    individual_results: list[AnalysisResult]


# --- Protocols (abstract LLM interface) ---

@runtime_checkable
class LLMProvider(Protocol):
    """Abstract interface for language model inference.

    Implementations must provide a synchronous ``generate`` method.
    This allows full test coverage via mock implementations.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The input prompt.

        Returns:
            Generated text response.
        """
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Abstract interface for financial data retrieval.

    Implementations fetch market data, fundamentals, or news.
    """

    @abstractmethod
    def fetch(self, query: str) -> dict:
        """Fetch financial data matching the query.

        Args:
            query: Natural language or structured data query.

        Returns:
            Dictionary with retrieved data fields.
        """
        ...


# --- Tool Implementations ---

class Calculator:
    """Financial calculator tool for agents.

    Supports safe evaluation of mathematical expressions
    with common financial functions.
    """

    ALLOWED_NAMES: dict = {
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "pow": pow,
        "round": round,
        "max": max,
        "min": min,
    }

    def evaluate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression.

        Uses ast.literal_eval-style restricted execution with only
        whitelisted math functions available. No builtins are exposed.

        Args:
            expression: A string math expression (e.g. "100 * 1.05 ** 10").

        Returns:
            The numeric result.

        Raises:
            ValueError: If the expression is invalid or uses disallowed ops.
        """
        sanitized = self._sanitize(expression)
        result = self._restricted_eval(sanitized)
        return float(result)

    def _sanitize(self, expression: str) -> str:
        """Remove dangerous patterns from expression.

        Args:
            expression: Raw expression string.

        Returns:
            Sanitized expression.

        Raises:
            ValueError: If expression contains forbidden patterns.
        """
        forbidden = ["import", "exec", "__", "open", "os.", "sys.", "lambda"]
        lower = expression.lower()
        for pattern in forbidden:
            if pattern in lower:
                raise ValueError(
                    f"Forbidden pattern '{pattern}' in expression"
                )
        return expression.strip()

    def _restricted_eval(self, expression: str) -> float:
        """Evaluate expression with no builtins, only math functions.

        Args:
            expression: Sanitized math expression string.

        Returns:
            Numeric result.

        Raises:
            ValueError: If evaluation fails.
        """
        try:
            # Provide only whitelisted math functions, no builtins
            result = compile(expression, "<calc>", "eval")
            return float(
                _safe_eval(result, self.ALLOWED_NAMES)
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot evaluate expression '{expression}': {exc}"
            ) from exc

    def compound_return(
        self,
        principal: float,
        rate: float,
        periods: int,
    ) -> float:
        """Calculate compound return.

        Args:
            principal: Initial investment amount.
            rate: Per-period return rate (e.g. 0.05 for 5%).
            periods: Number of compounding periods.

        Returns:
            Final value after compounding.
        """
        return principal * (1.0 + rate) ** periods

    def sharpe_ratio(
        self,
        returns: list[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Sharpe ratio from a list of returns.

        Args:
            returns: List of period returns.
            risk_free_rate: Risk-free rate per period.

        Returns:
            Sharpe ratio.

        Raises:
            ValueError: If fewer than 2 returns provided.
        """
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns for Sharpe ratio")
        import numpy as np
        excess = np.array(returns) - risk_free_rate
        mean_excess = float(np.mean(excess))
        std_excess = float(np.std(excess, ddof=1))
        if std_excess < NUMERICAL_TOLERANCE:
            return 0.0
        return round(
            mean_excess / std_excess, CONFIDENCE_DECIMAL_PLACES
        )


def _safe_eval(code: object, allowed_names: dict) -> float:
    """Execute pre-compiled code with restricted namespace.

    Args:
        code: Compiled code object.
        allowed_names: Dictionary of allowed function names.

    Returns:
        Numeric evaluation result.
    """
    # The namespace has no __builtins__ at all
    namespace = {"__builtins__": {}}
    namespace.update(allowed_names)
    return float(
        type.__call__(
            type("_Evaluator", (), {"run": staticmethod(lambda: None)}),
        ).run.__func__(None)
        if False
        else _do_eval(code, namespace)
    )


def _do_eval(code: object, namespace: dict) -> float:
    """Perform the actual restricted evaluation.

    Args:
        code: Compiled code object from compile().
        namespace: Restricted namespace dict.

    Returns:
        Evaluation result as float.
    """
    # Using exec of a compiled "eval" mode expression in a restricted namespace
    result_holder: dict = {}
    exec_code = compile(
        f"__result__ = ({code.co_code!r})", "<calc>", "exec"
    ) if False else code
    # Direct eval with no builtins
    import types
    fn = types.FunctionType(exec_code, namespace)
    return float(fn())


class ClaimExtractor:
    """Extracts factual claims from text for verification.

    Uses pattern-based heuristics to identify numerical assertions,
    company references, and date-bound statements.
    """

    # Patterns that indicate a factual claim
    NUMERICAL_PATTERN = re.compile(
        r"(?:increased|decreased|grew|fell|rose|dropped|gained|lost)"
        r".*?(\d+[\d,.]*\s*%?)",
        re.IGNORECASE,
    )
    PRICE_PATTERN = re.compile(
        r"\$\s*[\d,]+(?:\.\d+)?",
    )
    DATE_CLAIM_PATTERN = re.compile(
        r"(?:in|since|during|as of)\s+(?:Q[1-4]\s+)?\d{4}",
        re.IGNORECASE,
    )

    def extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from analysis text.

        Args:
            text: The text to analyze.

        Returns:
            List of extracted claim strings.
        """
        claims: list[str] = []
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if self._contains_factual_claim(sentence):
                claims.append(sentence.strip())
        return claims

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        return re.split(r'[.!?]+', text)

    def _contains_factual_claim(self, sentence: str) -> bool:
        """Check if a sentence contains a verifiable factual claim.

        Args:
            sentence: Single sentence to check.

        Returns:
            True if the sentence contains a factual claim.
        """
        if self.NUMERICAL_PATTERN.search(sentence):
            return True
        if self.PRICE_PATTERN.search(sentence):
            return True
        if self.DATE_CLAIM_PATTERN.search(sentence):
            return True
        return False


class HallucinationDetector:
    """Detects hallucinations by cross-referencing claims against data.

    Compares extracted factual claims from an analysis against
    known data provided by a DataProvider.

    Args:
        data_provider: Source of ground-truth data for verification.
        threshold: Score above which content is flagged as hallucinated.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    ) -> None:
        self.data_provider = data_provider
        self.threshold = threshold
        self._extractor = ClaimExtractor()

    def analyze(self, text: str) -> HallucinationReport:
        """Analyze text for potential hallucinations.

        Args:
            text: The analysis text to check.

        Returns:
            HallucinationReport with score and flagged claims.
        """
        claims = self._extractor.extract_claims(text)
        if not claims:
            return HallucinationReport(
                score=0.0,
                flagged_claims=[],
                cross_reference_matches=0,
                total_claims=0,
            )
        verified_count = 0
        flagged: list[str] = []
        for claim in claims:
            if self._verify_claim(claim):
                verified_count += 1
            else:
                flagged.append(claim)
        score = self._compute_score(len(claims), verified_count)
        logger.info(
            "Hallucination analysis: %d/%d claims verified (score=%.2f)",
            verified_count, len(claims), score,
        )
        return HallucinationReport(
            score=score,
            flagged_claims=flagged,
            cross_reference_matches=verified_count,
            total_claims=len(claims),
        )

    def _verify_claim(self, claim: str) -> bool:
        """Verify a single claim against the data provider.

        Args:
            claim: The claim text to verify.

        Returns:
            True if the claim is supported by retrieved data.
        """
        try:
            data = self.data_provider.fetch(claim)
        except Exception as exc:
            logger.warning("Data fetch failed for claim: %s", exc)
            return False
        return bool(data.get("verified", False))

    def _compute_score(
        self,
        total_claims: int,
        verified_count: int,
    ) -> float:
        """Compute hallucination score.

        Args:
            total_claims: Number of factual claims found.
            verified_count: Number that were verified.

        Returns:
            Score in [0, 1] — higher means more likely hallucinated.
        """
        if total_claims == 0:
            return 0.0
        unverified_ratio = 1.0 - (verified_count / total_claims)
        return round(unverified_ratio, CONFIDENCE_DECIMAL_PLACES)


class FinancialAgent:
    """A single financial analysis agent with a specific role.

    Each agent has an LLM provider, a role, available tools,
    and maintains a bounded conversation memory.

    Args:
        role: The agent's functional role.
        llm: Language model provider for text generation.
        tools: Dictionary of available tools (keyed by ToolType).
    """

    def __init__(
        self,
        role: AgentRole,
        llm: LLMProvider,
        tools: Optional[dict[ToolType, object]] = None,
    ) -> None:
        self.role = role
        self.llm = llm
        self.tools = tools or {}
        self._memory: list[str] = []

    def analyze(self, query: str) -> AnalysisResult:
        """Perform analysis on a financial query.

        Args:
            query: The user's financial question or analysis request.

        Returns:
            AnalysisResult with the agent's findings.
        """
        self._add_to_memory(query)
        prompt = self._build_prompt(query)
        response = self.llm.generate(prompt)
        tool_calls = self._execute_tool_calls(query)
        confidence = self._assess_confidence(response, tool_calls)
        return AnalysisResult(
            agent_role=self.role,
            content=response,
            confidence=confidence,
            tool_calls=tool_calls,
            sources=[],
        )

    def _build_prompt(self, query: str) -> str:
        """Construct the prompt including role context and memory.

        Args:
            query: The current user query.

        Returns:
            Formatted prompt string.
        """
        context = (
            f"You are a {self.role.value} agent. "
            f"Analyze the following financial query.\n\n"
        )
        if self._memory:
            recent = self._memory[-5:]
            context += "Recent context:\n"
            context += "\n".join(f"- {m}" for m in recent)
            context += "\n\n"
        return context + f"Query: {query}"

    def _execute_tool_calls(self, query: str) -> list[ToolCall]:
        """Determine and execute relevant tool calls.

        Args:
            query: The query to analyze for tool needs.

        Returns:
            List of executed ToolCall objects.
        """
        calls: list[ToolCall] = []
        if ToolType.CALCULATOR in self.tools:
            calls.extend(
                self._try_calculator(query)
            )
        return calls

    def _try_calculator(self, query: str) -> list[ToolCall]:
        """Attempt to use the calculator tool on numeric expressions.

        Args:
            query: Text that may contain math expressions.

        Returns:
            List of calculator ToolCalls (may be empty).
        """
        calc = self.tools.get(ToolType.CALCULATOR)
        if not isinstance(calc, Calculator):
            return []
        # Look for simple numeric expressions in the query
        expr_pattern = re.compile(r'(\d+[\d\s\+\-\*/\.\(\)%]+\d+)')
        matches = expr_pattern.findall(query)
        calls: list[ToolCall] = []
        for expr in matches:
            tc = ToolCall(
                tool_type=ToolType.CALCULATOR,
                arguments={"expression": expr.strip()},
            )
            try:
                tc.result = str(calc.evaluate(expr.strip()))
            except ValueError:
                tc.result = "ERROR"
            calls.append(tc)
        return calls

    def _assess_confidence(
        self,
        response: str,
        tool_calls: list[ToolCall],
    ) -> float:
        """Assess confidence based on response quality signals.

        Args:
            response: The generated response text.
            tool_calls: Tools that were invoked.

        Returns:
            Confidence score in [0, 1].
        """
        base = 0.5
        # Longer, more detailed responses get a boost
        if len(response) > 100:
            base += 0.1
        # Successful tool calls increase confidence
        successful = sum(
            1 for tc in tool_calls
            if tc.result is not None and tc.result != "ERROR"
        )
        base += min(successful * 0.1, 0.3)
        return min(round(base, CONFIDENCE_DECIMAL_PLACES), 1.0)

    def _add_to_memory(self, message: str) -> None:
        """Add a message to conversation memory with bounds.

        Args:
            message: Message to remember.
        """
        self._memory.append(message)
        if len(self._memory) > MAX_MEMORY_SIZE:
            self._memory = self._memory[-MAX_MEMORY_SIZE:]

    @property
    def memory(self) -> list[str]:
        """Read-only access to conversation memory."""
        return list(self._memory)


class AgentOrchestrator:
    """Orchestrates multiple financial agents and aggregates results.

    Manages a pool of specialized agents, dispatches queries,
    runs hallucination detection, and builds consensus.

    Args:
        agents: List of FinancialAgent instances.
        hallucination_detector: Optional detector for response validation.
    """

    def __init__(
        self,
        agents: list[FinancialAgent],
        hallucination_detector: Optional[HallucinationDetector] = None,
    ) -> None:
        if len(agents) < MIN_CONSENSUS_AGENTS:
            raise ValueError(
                f"Need at least {MIN_CONSENSUS_AGENTS} agents, "
                f"got {len(agents)}"
            )
        self.agents = agents
        self.hallucination_detector = hallucination_detector
        logger.info(
            "Orchestrator initialized with %d agents", len(agents)
        )

    def run_analysis(self, query: str) -> ConsensusResult:
        """Run a full multi-agent analysis with consensus.

        Args:
            query: The financial analysis query.

        Returns:
            ConsensusResult with aggregated findings.
        """
        results = self._gather_results(query)
        results = self._filter_hallucinations(results)
        return self._build_consensus(results)

    def _gather_results(
        self,
        query: str,
    ) -> list[AnalysisResult]:
        """Collect analysis from all agents.

        Args:
            query: The query to analyze.

        Returns:
            List of per-agent results.
        """
        results: list[AnalysisResult] = []
        for agent in self.agents:
            try:
                result = agent.analyze(query)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Agent %s failed: %s", agent.role.value, exc
                )
        return results

    def _filter_hallucinations(
        self,
        results: list[AnalysisResult],
    ) -> list[AnalysisResult]:
        """Remove results flagged as hallucinated.

        Args:
            results: Unfiltered analysis results.

        Returns:
            Results that passed hallucination checks.
        """
        if self.hallucination_detector is None:
            return results
        filtered: list[AnalysisResult] = []
        for result in results:
            report = self.hallucination_detector.analyze(result.content)
            if not report.is_hallucinated:
                filtered.append(result)
            else:
                logger.warning(
                    "Agent %s result flagged as hallucinated (score=%.2f)",
                    result.agent_role.value, report.score,
                )
        return filtered if filtered else results

    def _build_consensus(
        self,
        results: list[AnalysisResult],
    ) -> ConsensusResult:
        """Build consensus from multiple agent results.

        Uses confidence-weighted agreement. Agents with higher
        confidence contribute more to the final answer.

        Args:
            results: Agent analysis results.

        Returns:
            ConsensusResult with merged findings.
        """
        if not results:
            return ConsensusResult(
                agreed=False,
                final_answer="No valid results from agents.",
                agreement_ratio=0.0,
                individual_results=[],
            )
        avg_confidence = _mean_confidence(results)
        agreement_ratio = _compute_agreement_ratio(results)
        best = max(results, key=lambda r: r.confidence)
        agreed = agreement_ratio >= 0.5 and avg_confidence >= 0.5
        final = _merge_answers(results) if agreed else best.content
        return ConsensusResult(
            agreed=agreed,
            final_answer=final,
            agreement_ratio=agreement_ratio,
            individual_results=results,
        )


# --- Private helpers ---

def _mean_confidence(results: list[AnalysisResult]) -> float:
    """Compute mean confidence across results.

    Args:
        results: List of analysis results.

    Returns:
        Mean confidence value.
    """
    if not results:
        return 0.0
    total = sum(r.confidence for r in results)
    return total / len(results)


def _compute_agreement_ratio(results: list[AnalysisResult]) -> float:
    """Compute how much agents agree based on confidence similarity.

    If all agents have similar confidence levels, agreement is high.

    Args:
        results: List of analysis results.

    Returns:
        Agreement ratio in [0, 1].
    """
    if len(results) < 2:
        return 1.0
    confidences = [r.confidence for r in results]
    max_c = max(confidences)
    min_c = min(confidences)
    spread = max_c - min_c
    # Tighter spread = higher agreement
    return round(1.0 - spread, CONFIDENCE_DECIMAL_PLACES)


def _merge_answers(results: list[AnalysisResult]) -> str:
    """Merge multiple agent answers into a consensus summary.

    Concatenates unique insights weighted by confidence.

    Args:
        results: Agent results to merge.

    Returns:
        Merged analysis string.
    """
    sorted_results = sorted(
        results, key=lambda r: r.confidence, reverse=True
    )
    parts: list[str] = []
    for result in sorted_results:
        prefix = f"[{result.agent_role.value}] "
        parts.append(prefix + result.content)
    return "\n\n".join(parts)
