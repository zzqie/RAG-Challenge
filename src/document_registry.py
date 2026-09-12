import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class DocumentShard:
    """Metadata for one report-level retrieval shard."""

    report_id: str
    sha1: str
    company_name: str
    document_path: Path
    index_path: Optional[Path] = None
    currency: Optional[str] = None
    industry: Optional[str] = None
    document_type: str = "annual_report"
    fiscal_year: Optional[str] = None


class DocumentRegistry:
    """Registry that maps business entities to report-level knowledge shards."""

    def __init__(self, shards: List[DocumentShard]):
        self.shards = shards
        self._by_report_id: Dict[str, DocumentShard] = {
            shard.report_id: shard for shard in shards
        }

    @classmethod
    def from_directories(
        cls,
        documents_dir: Union[str, Path],
        vector_db_dir: Union[str, Path],
        subset_path: Optional[Union[str, Path]] = None,
    ) -> "DocumentRegistry":
        documents_dir = Path(documents_dir)
        vector_db_dir = Path(vector_db_dir)
        metadata = cls._load_subset_metadata(subset_path)

        shards: List[DocumentShard] = []
        for document_path in sorted(documents_dir.glob("*.json")):
            with open(document_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            metainfo = document.get("metainfo", {})
            sha1 = metainfo.get("sha1_name") or document_path.stem
            row = metadata.get(sha1, {})

            shards.append(
                DocumentShard(
                    report_id=document_path.stem,
                    sha1=sha1,
                    company_name=metainfo.get("company_name") or row.get("company_name", ""),
                    document_path=document_path,
                    index_path=vector_db_dir / f"{sha1}.faiss",
                    currency=row.get("cur"),
                    industry=row.get("major_industry"),
                    document_type=row.get("document_type", "annual_report"),
                    fiscal_year=row.get("fiscal_year") or row.get("year"),
                )
            )

        return cls(shards)

    @staticmethod
    def _load_subset_metadata(
        subset_path: Optional[Union[str, Path]]
    ) -> Dict[str, Dict[str, str]]:
        if subset_path is None:
            return {}

        subset_path = Path(subset_path)
        if not subset_path.exists():
            return {}

        with open(subset_path, "r", encoding="utf-8") as f:
            return {row["sha1"]: row for row in csv.DictReader(f) if row.get("sha1")}

    @property
    def company_names(self) -> List[str]:
        names = {shard.company_name for shard in self.shards if shard.company_name}
        return sorted(names, key=len, reverse=True)

    def get_report(self, report_id: str) -> DocumentShard:
        if report_id not in self._by_report_id:
            raise ValueError(f"No report shard found with report_id '{report_id}'.")
        return self._by_report_id[report_id]

    def find_by_company(self, company_name: str) -> List[DocumentShard]:
        normalized = company_name.strip().casefold()
        return [
            shard
            for shard in self.shards
            if shard.company_name.strip().casefold() == normalized
        ]

    @staticmethod
    def _matches(value: Optional[str], expected: Optional[str]) -> bool:
        if expected is None or expected == "":
            return True
        if value is None:
            return False
        return str(value).strip().casefold() == str(expected).strip().casefold()

    def find_by_constraints(
        self,
        company_name: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        document_type: Optional[str] = None,
        currency: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> List[DocumentShard]:
        """Return all report shards matching the provided metadata constraints."""
        candidates = self.shards

        if company_name:
            candidates = self.find_by_company(company_name)

        return [
            shard
            for shard in candidates
            if self._matches(shard.fiscal_year, fiscal_year)
            and self._matches(shard.document_type, document_type)
            and self._matches(shard.currency, currency)
            and self._matches(shard.industry, industry)
        ]

    def resolve_report(
        self,
        company_name: str,
        fiscal_year: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> DocumentShard:
        candidates = self.find_by_constraints(
            company_name=company_name,
            fiscal_year=fiscal_year,
            document_type=document_type,
        )

        if not candidates:
            raise ValueError(f"No report shard found for company '{company_name}'.")

        return sorted(candidates, key=lambda shard: shard.report_id)[0]
