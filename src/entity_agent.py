import json
import os
from typing import List, Optional

from src.document_registry import DocumentRegistry

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ModuleNotFoundError:
    PYDANTIC_AVAILABLE = False

    def Field(default=None, default_factory=None, description=None):
        return default_factory() if default_factory is not None else default

    class BaseModel:
        def __init__(self, **kwargs):
            for name in getattr(self.__class__, "__annotations__", {}):
                default = getattr(self.__class__, name, None)
                if isinstance(default, list):
                    default = list(default)
                setattr(self, name, kwargs.get(name, default))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


class DocumentQueryConstraint(BaseModel):
    """Structured retrieval constraints extracted from a user question."""

    company_name: Optional[str] = Field(
        default=None,
        description="Registered company name if the question mentions or implies one.",
    )
    fiscal_year: Optional[str] = Field(
        default=None,
        description="Fiscal/reporting year mentioned by the user, such as 2015 or 2022.",
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Document type such as annual_report, quarterly_report, prospectus, or audit_report.",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency constraint if explicitly mentioned, such as USD, EUR, GBP, or CHF.",
    )
    industry: Optional[str] = Field(
        default=None,
        description="Industry constraint if explicitly mentioned.",
    )


class EntityRecognitionResult(BaseModel):
    constraints: List[DocumentQueryConstraint] = Field(
        default_factory=list,
        description="One item per target entity/document scope found in the question.",
    )
    rewritten_query: str = Field(
        default="",
        description="Standalone query after removing ambiguity, preserving the user's intent.",
    )
    reasoning: str = Field(
        default="",
        description="Short explanation of the extracted routing constraints.",
    )


class EntityRecognitionAgent:
    """LLM agent that extracts document routing constraints from a question."""

    def __init__(
        self,
        registry: DocumentRegistry,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        load_dotenv()
        self.registry = registry
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if api_key and OpenAI is not None:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
                timeout=30,
                max_retries=1,
            )

    def extract(self, question_text: str) -> EntityRecognitionResult:
        system_prompt = self._build_system_prompt()
        if self.client is None:
            return EntityRecognitionResult(rewritten_query=question_text)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ]

            if PYDANTIC_AVAILABLE:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=0,
                    messages=messages,
                    response_format=EntityRecognitionResult,
                )
                parsed = completion.choices[0].message.parsed
                return parsed or EntityRecognitionResult(rewritten_query=question_text)

            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return self._parse_json_result(completion.choices[0].message.content, question_text)
        except Exception:
            return EntityRecognitionResult(rewritten_query=question_text)

    def _parse_json_result(self, content: str, question_text: str) -> EntityRecognitionResult:
        data = json.loads(content)
        constraints = [
            DocumentQueryConstraint(**constraint)
            for constraint in data.get("constraints", [])
            if isinstance(constraint, dict)
        ]
        return EntityRecognitionResult(
            constraints=constraints,
            rewritten_query=data.get("rewritten_query", question_text),
            reasoning=data.get("reasoning", ""),
        )

    def _build_system_prompt(self) -> str:
        registry_context = {
            "companies": self.registry.company_names,
            "fiscal_years": sorted(
                {shard.fiscal_year for shard in self.registry.shards if shard.fiscal_year}
            ),
            "document_types": sorted(
                {shard.document_type for shard in self.registry.shards if shard.document_type}
            ),
            "currencies": sorted(
                {shard.currency for shard in self.registry.shards if shard.currency}
            ),
            "industries": sorted(
                {shard.industry for shard in self.registry.shards if shard.industry}
            ),
        }

        return f"""
You are an entity recognition and document routing agent for a financial RAG system.

Your task is to extract structured constraints that can route a user question to report-level knowledge shards.

Known registry values:
{json.dumps(registry_context, ensure_ascii=False, indent=2)}

Rules:
1. Prefer exact values from the registry when the user mentions a known company, industry, currency, year, or document type.
2. If the user writes a close alias or minor typo for a known company, map it to the closest registered company only when it is clear.
3. Normalize document type:
   - annual report, annual filing, 10-K, 年报 -> annual_report
   - quarterly report, 10-Q, 季报 -> quarterly_report
   - prospectus, 招股书 -> prospectus
   - audit report, 审计报告 -> audit_report
4. If the user only mentions a company and no year/document type, return a constraint with only company_name. The registry will search all documents for that company.
5. If multiple companies or document scopes are mentioned, return multiple constraints.
6. Do not invent values. Leave fields null when they are not mentioned or cannot be confidently mapped.
"""
