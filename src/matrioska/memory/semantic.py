"""
Semantic memory — embedding-based retrieval with GraphRAG scaffold.

Builds a knowledge graph from past runs, extracting concepts, patterns,
and common bugs.  Supports k-hop graph traversal for non-obvious connections.

Tier 3 of the hierarchical memory system (§4.5).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("matrioska.memory.semantic")


@dataclass
class Concept:
    """A concept node in the knowledge graph."""
    id: str
    label: str
    description: str
    type: str = "concept"  # concept, pattern, bug, solution
    relations: List[str] = field(default_factory=list)
    occurrences: int = 0
    last_seen: str = ""


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source: str
    target: str
    relation: str  # "uses", "solves", "causes", "depends_on", "similar_to"
    weight: float = 1.0


class SemanticMemory:
    """ChromaDB-backed semantic memory with GraphRAG scaffold.

    Concepts are extracted from run notes and stored as embeddings.
    Relationships form a knowledge graph for k-hop retrieval.
    """

    def __init__(self, work_dir: Path):
        self.root = Path(work_dir) / "knowledge"
        self.concepts_dir = self.root / "concepts"
        self.concepts_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.root / "knowledge_graph.json"

        self._concepts: Dict[str, Concept] = {}
        self._relationships: List[Relationship] = []
        self._collection: Any = None
        self._chroma_ok = False

        self._init_chroma()
        self._load_graph()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            ef = embedding_functions.DefaultEmbeddingFunction()
            client = chromadb.PersistentClient(
                path=str(self.root / ".chromadb")
            )
            self._collection = client.get_or_create_collection(
                name="semantic_memory",
                embedding_function=ef,
            )
            self._chroma_ok = True
        except ImportError:
            logger.debug("ChromaDB not available; semantic memory uses disk-only mode")
        except Exception as e:
            logger.warning("ChromaDB init failed: %s", e)

    def _load_graph(self) -> None:
        if not self.graph_path.exists():
            return
        try:
            data = json.loads(self.graph_path.read_text(encoding="utf-8"))
            self._concepts = {
                k: Concept(**v) for k, v in data.get("concepts", {}).items()
            }
            self._relationships = [
                Relationship(**r) for r in data.get("relationships", [])
            ]
        except Exception as e:
            logger.warning("Failed to load knowledge graph: %s", e)

    def _save_graph(self) -> None:
        data = {
            "concepts": {k: vars(v) for k, v in self._concepts.items()},
            "relationships": [vars(r) for r in self._relationships],
        }
        self.graph_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # ── Ingestion ────────────────────────────────────────────────────────

    def ingest_run(self, task: str, tags: List[str], status: str) -> None:
        """Extract concepts from a run and merge into the graph."""
        for tag in tags:
            concept_id = f"tag:{tag}"
            if concept_id not in self._concepts:
                self._concepts[concept_id] = Concept(
                    id=concept_id, label=tag, description=f"Tag: {tag}",
                )
            self._concepts[concept_id].occurrences += 1

        if self._chroma_ok and self._collection is not None:
            try:
                self._collection.add(
                    documents=[task],
                    metadatas=[{"tags": json.dumps(tags), "status": status}],
                    ids=[f"run:{_slug(task)}"],
                )
            except Exception as e:
                logger.debug("ChromaDB add failed: %s", e)

        self._save_graph()

    # ── Retrieval ────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> List[Concept]:
        """Search concepts by semantic similarity."""
        if self._chroma_ok and self._collection is not None:
            try:
                results = self._collection.query(
                    query_texts=[query], n_results=k,
                )
                concept_ids = set()
                concepts = []
                for meta_list in results.get("metadatas", [[]]):
                    for meta in meta_list:
                        tags_str = meta.get("tags", "[]") if meta else "[]"
                        try:
                            for tag in json.loads(tags_str):
                                cid = f"tag:{tag}"
                                if cid not in concept_ids and cid in self._concepts:
                                    concepts.append(self._concepts[cid])
                                    concept_ids.add(cid)
                        except Exception:
                            pass
                return concepts[:k]
            except Exception as e:
                logger.debug("ChromaDB query failed: %s", e)

        # Fallback: keyword match on concept labels
        query_lower = query.lower()
        return [
            c for c in self._concepts.values()
            if c.label.lower() in query_lower
        ][:k]

    def traverse(
        self, concept_id: str, hops: int = 2
    ) -> List[Concept]:
        """k-hop graph traversal from a concept node."""
        visited: set[str] = {concept_id}
        frontier: set[str] = {concept_id}
        result_ids: set[str] = {concept_id}

        for _ in range(hops):
            next_frontier: set[str] = set()
            for rel in self._relationships:
                if rel.source in frontier and rel.target not in visited:
                    next_frontier.add(rel.target)
                    result_ids.add(rel.target)
                elif rel.target in frontier and rel.source not in visited:
                    next_frontier.add(rel.source)
                    result_ids.add(rel.source)
            visited.update(next_frontier)
            frontier = next_frontier

        return [self._concepts[cid] for cid in result_ids if cid in self._concepts]

    def add_relationship(self, source: str, target: str, relation: str) -> None:
        self._relationships.append(Relationship(source=source, target=target, relation=relation))
        self._save_graph()

    def stats(self) -> Dict[str, int]:
        return {
            "concepts": len(self._concepts),
            "relationships": len(self._relationships),
            "chroma_available": self._chroma_ok,
        }


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")[:60]
