import re
from pathlib import Path
from typing import List, Optional, Union

from src.document_registry import DocumentRegistry, DocumentShard
from src.entity_agent import EntityRecognitionAgent


class ReportShardRouter:
    """Routes a user question to candidate report-level document shards."""

    def __init__(
        self,
        documents_dir: Union[str, Path],
        vector_db_dir: Union[str, Path],
        subset_path: Optional[Union[str, Path]] = None,
        use_agent: bool = True,
        routing_model: str = "gpt-4o-mini",
    ):
        self.registry = DocumentRegistry.from_directories(
            documents_dir=documents_dir,
            vector_db_dir=vector_db_dir,
            subset_path=subset_path,
        )
        self.agent = (
            EntityRecognitionAgent(self.registry, model=routing_model)
            if use_agent
            else None
        )

    def extract_companies(self, question_text: str) -> List[str]:
        found_companies: List[str] = []
        remaining_text = question_text

        for company in self.registry.company_names:
            pattern = rf"{re.escape(company)}(?:\W|$)"
            if re.search(pattern, remaining_text, re.IGNORECASE):
                found_companies.append(company)
                remaining_text = re.sub(pattern, "", remaining_text, flags=re.IGNORECASE)

        return found_companies

    def route_question(self, question_text: str) -> List[DocumentShard]:
        if self.agent is not None:
            routed = self._route_with_agent(question_text)
            if routed:
                return routed

        return self._route_with_rules(question_text)

    def _route_with_agent(self, question_text: str) -> List[DocumentShard]:
        result = self.agent.extract(question_text)
        shards: List[DocumentShard] = []
        seen_report_ids = set()

        for constraint in result.constraints:
            matches = self.registry.find_by_constraints(
                company_name=constraint.company_name,
                fiscal_year=constraint.fiscal_year,
                document_type=constraint.document_type,
                currency=constraint.currency,
                industry=constraint.industry,
            )
            for shard in matches:
                if shard.report_id not in seen_report_ids:
                    shards.append(shard)
                    seen_report_ids.add(shard.report_id)

        return shards

    def _route_with_rules(self, question_text: str) -> List[DocumentShard]:
        shards: List[DocumentShard] = []
        for company in self.extract_companies(question_text):
            shards.extend(self.registry.find_by_constraints(company_name=company))
        return shards
