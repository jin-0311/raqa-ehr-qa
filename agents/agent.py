"""Clean hybrid agent base pipeline.

This module keeps only the core flow:
1) MetaKG generation
2) Schema injection
3) Query plan generation
4) CAQA score calculation and best-plan selection
5) Query execution
"""

import copy
import datetime as dt
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at import time
    OpenAI = None  # type: ignore


def _to_json_serializable(value: Any) -> Any:
    """Convert nested values to JSON-serializable structures."""
    try:
        from neo4j.time import Date, DateTime, Duration, Time

        neo4j_temporal_types = (Date, DateTime, Duration, Time)
    except Exception:
        neo4j_temporal_types = tuple()

    if neo4j_temporal_types and isinstance(value, neo4j_temporal_types):
        if hasattr(value, "iso_format"):
            return value.iso_format()
        return str(value)
    if isinstance(value, dict):
        return {k: _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_serializable(v) for v in value]
    return value


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from plain or fenced text."""
    if not isinstance(text, str):
        raise ValueError("Expected JSON text as a string.")

    candidates: List[str] = []
    raw = text.strip()
    candidates.append(raw)

    fenced_json = re.findall(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    fenced_any = re.findall(r"```\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidates.extend(fenced_json)
    candidates.extend(fenced_any)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        for match in re.finditer(r"\{", candidate):
            snippet = candidate[match.start() :]
            try:
                parsed, _ = decoder.raw_decode(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    raise ValueError("Failed to parse a JSON object from model output.")


def _find_first_int(value: Any) -> Optional[int]:
    """Find the first integer value from arbitrary nested output."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value.replace(",", ""))
        return int(match.group(0)) if match else None
    if isinstance(value, dict):
        for key in ("count", "COUNT(*)", "row_count", "value"):
            if key in value:
                found = _find_first_int(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _find_first_int(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_first_int(item)
            if found is not None:
                return found
    return None


def _as_int(value: Any, default: int) -> int:
    found = _find_first_int(value)
    return found if found is not None else default


def _infer_simple_type(value: Any) -> str:
    """Infer a compact primitive type label from a sample value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple, set)):
        return "list"
    if isinstance(value, dict):
        return "map"
    return "other"


class OpenAILLM:
    """Minimal OpenAI chat wrapper."""

    SYSTEM_PROMPT = (
        "You are a strict JSON generator for a database planning agent. "
        "Always respond in English. "
        "When asked for JSON, return valid JSON only. "
        "Only use tables, labels, columns, properties, and relationship types present in the given schema. "
        "Never invent schema names, and never guess unavailable fields."
    )


    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required. Install with `pip install openai`.")
        if not api_key:
            raise ValueError("api_key is required.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str, json_mode: bool = False) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()


PROMPTS = {
    "meta_kg": "Build a MetaKG aligning RDB tables/columns with GDB labels/properties; emit cross-source keys only for high-confidence ID-like matches; never invent mappings.",
    "metakg_usage_contract": "Use ONLY exact schema names from the MetaKG/schema context; never infer generic or cross-source names.",
    "planner_rules": "Return exactly 2 executable candidate plans (db_type RDB/GDB/BOTH), no placeholders, schema names only; prefer the simpler valid plan.",
    "schema_validation_checklist": "Validate every table/column/label/relationship/property against the schema before emitting a plan.",
    "execution_format_contract": "One runnable statement per query/step; for BOTH use 'Step 1) RDB: <SQL>' then 'Step 2) GDB: <Cypher>'.",
    "multi_step_orchestration_contract": "Decompose multi-step questions into ordered sub-queries whose results compose into the final answer.",
    "mysql_dialect_guard": "MySQL, case-sensitive CamelCase tables (Patient/Admission/Diagnosis/Lab/Medi/Operation/Vital); quote string IDs; there is no Day table.",
    "cypher_dialect_guard": "Cypher: bind every returned variable after WITH; toFloat() numeric string props; do not rewrap Date props in datetime().",
    "mysql_cte_contract": "Use a CTE for top-k / multi-aggregate results (never a scalar subquery returning multiple rows).",
    "sql_aggregation_contract": "Put every non-aggregated selected column in GROUP BY; use COUNT(DISTINCT ...) to avoid fan-out double counting.",
    "gdb_timeline_pattern": "Traverse all days via MATCH (d:Day {admissionKey:a.admissionKey}) ... ORDER BY d.index; (a)-[:NEXT]->(:Day) is the first day only.",
    "gdb_topology_hard_constraints": "Edges: HAS_ADMISSION/NEXT/HAS_DIAG/HAS_LAB/HAS_MEDI/HAS_VITAL/HAS_OPERATION; diagnoses attach to Day{index:0}.",
    "gdb_generation_recipe": "Patient -> Admission -> Day; attach events on their Day; aggregate with ORDER BY d.index.",
    "gdb_template_examples": "e.g. MATCH (p:Patient {patientId:'X'})-[:HAS_ADMISSION]->(a) MATCH (d:Day {admissionKey:a.admissionKey})-[:HAS_LAB]->(l) RETURN ...",
    "gdb_diagnostic_pattern": "Diagnosis-linked: match (Day{index:0})-[:HAS_DIAG]->(Diagnosis) then relate to downstream events.",
    "gdb_quality_contract": "Return a small, non-empty, meaningful result; match clinical names with toLower(x.name) CONTAINS toLower('kw').",
    "gdb_population_aggregation_template": "For cohorts, match patients by a demographic filter then aggregate the target entity across them.",
}


@dataclass
class ComplexityConstants:
    """Constants included for planner prompt transparency."""

    J_c: int = 2
    H_c: int = 3
    F_c_low: int = 1
    F_c_high: int = 10
    G_c: int = 3


class MetaKnowledgeGraphGenerator:
    """Generate and store schema-level meta knowledge for RDB + GDB."""

    def __init__(self, rdb: Any, neo4j_driver: Any, llm: OpenAILLM, sample_limit: int = 3) -> None:
        self.rdb = rdb
        self.neo4j_driver = neo4j_driver
        self.llm = llm
        self.sample_limit = sample_limit
        self.meta_kg: Dict[str, Any] = {}

    @staticmethod
    def _normalize_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    @staticmethod
    def _extract_rdb_columns_from_schema_text(schema_text: Any) -> Set[str]:
        text = str(schema_text or "")
        if not text.strip():
            return set()

        columns: Set[str] = set()
        skip_prefixes = (
            "CREATE ",
            "PRIMARY ",
            "KEY ",
            "UNIQUE ",
            "CONSTRAINT ",
            "FOREIGN ",
            "INDEX ",
            ")",
        )
        for raw_line in text.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            upper = line.upper()
            if upper.startswith(skip_prefixes):
                continue
            match = re.match(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s+[A-Za-z]", line)
            if match:
                columns.add(match.group(1))
        return columns

    def _build_rdb_table_columns(self, rdb_info: Dict[str, Any]) -> Dict[str, Set[str]]:
        table_columns: Dict[str, Set[str]] = {}
        for table_name, table_info in rdb_info.items():
            columns: Set[str] = set()
            if isinstance(table_info, dict):
                columns |= self._extract_rdb_columns_from_schema_text(table_info.get("schema"))
                sample_data = table_info.get("sample_data")
                if isinstance(sample_data, list):
                    for row in sample_data:
                        if isinstance(row, dict):
                            columns |= set(str(k) for k in row.keys())
            table_columns[str(table_name)] = columns
        return table_columns

    def _candidate_tables_for_label(self, label: str, table_names: List[str]) -> List[str]:
        label_norm = self._normalize_name(label)
        candidates: List[str] = []
        for table in table_names:
            table_norm = self._normalize_name(table)
            stripped = table_norm[6:] if table_norm.startswith("sample") else table_norm
            singular = stripped[:-1] if stripped.endswith("s") else stripped
            label_singular = label_norm[:-1] if label_norm.endswith("s") else label_norm
            if (
                label_norm in (table_norm, stripped, singular)
                or label_singular in (table_norm, stripped, singular)
                or stripped in (label_norm, label_singular)
            ):
                candidates.append(table)
        return candidates

    def _derive_cross_source_keys(self, rdb_info: Dict[str, Any], gdb_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        table_columns = self._build_rdb_table_columns(rdb_info)
        table_names = list(table_columns.keys())
        gdb_props = gdb_info.get("node_properties", {}) if isinstance(gdb_info, dict) else {}
        node_labels = list(gdb_info.get("node_labels", [])) if isinstance(gdb_info, dict) else []

        rows: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for label in node_labels:
            props = gdb_props.get(label, [])
            if not isinstance(props, list) or not props:
                continue
            props_norm = {self._normalize_name(p): str(p) for p in props if str(p).strip()}
            candidates = self._candidate_tables_for_label(label, table_names)
            for table in candidates:
                cols = table_columns.get(table, set())
                if not cols:
                    continue
                cols_norm = {self._normalize_name(c): str(c) for c in cols if str(c).strip()}
                shared_norm = sorted(set(cols_norm.keys()) & set(props_norm.keys()))
                if not shared_norm:
                    continue

                id_like = [k for k in shared_norm if k.endswith("id")]
                others = [k for k in shared_norm if k not in id_like]
                ordered = id_like + others[:8]

                for key_norm in ordered:
                    rdb_col = cols_norm[key_norm]
                    gdb_prop = props_norm[key_norm]
                    signature = f"{label}|{table}|{rdb_col}|{gdb_prop}".lower()
                    if signature in seen:
                        continue
                    seen.add(signature)

                    confidence = 1.0 if key_norm.endswith("id") else 0.9
                    rows.append(
                        {
                            "entity": str(label),
                            "rdb": {"table": str(table), "column": str(rdb_col)},
                            "gdb": {"label": str(label), "property": str(gdb_prop)},
                            "relation": "equivalent_key",
                            "confidence": confidence,
                            "source": "deterministic_name_match",
                        }
                    )

        return rows

    @staticmethod
    def _coerce_cross_source_key_item(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        rdb_obj = item.get("rdb") if isinstance(item.get("rdb"), dict) else {}
        gdb_obj = item.get("gdb") if isinstance(item.get("gdb"), dict) else {}

        entity = str(item.get("entity", "") or item.get("label", "") or gdb_obj.get("label", "")).strip()
        rdb_table = str(
            rdb_obj.get("table", "")
            or item.get("rdb_table", "")
            or item.get("table", "")
        ).strip()
        rdb_column = str(
            rdb_obj.get("column", "")
            or item.get("rdb_column", "")
            or item.get("column", "")
        ).strip()
        gdb_label = str(
            gdb_obj.get("label", "")
            or item.get("gdb_label", "")
            or item.get("label", "")
        ).strip()
        gdb_property = str(
            gdb_obj.get("property", "")
            or item.get("gdb_property", "")
            or item.get("property", "")
        ).strip()

        if not (rdb_table and rdb_column and gdb_label and gdb_property):
            return None
        if not entity:
            entity = gdb_label

        confidence = item.get("confidence", 0.9)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.9

        return {
            "entity": entity,
            "rdb": {"table": rdb_table, "column": rdb_column},
            "gdb": {"label": gdb_label, "property": gdb_property},
            "relation": str(item.get("relation", "equivalent_key") or "equivalent_key"),
            "confidence": max(0.0, min(1.0, confidence_value)),
            "source": str(item.get("source", "llm_or_derived")),
        }

    def _merge_cross_source_keys(self, generated: Any, derived: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        generated_list = generated if isinstance(generated, list) else []
        for raw in list(generated_list) + list(derived):
            item = self._coerce_cross_source_key_item(raw)
            if not item:
                continue
            sig = (
                f"{item['entity']}|{item['rdb']['table']}|{item['rdb']['column']}|"
                f"{item['gdb']['label']}|{item['gdb']['property']}"
            ).lower()
            if sig in seen:
                continue
            seen.add(sig)
            merged.append(item)
        return merged

    @staticmethod
    def _build_entity_mapping_from_cross_source_keys(cross_keys: List[Dict[str, Any]]) -> Dict[str, Any]:
        mapping: Dict[str, Dict[str, Set[str]]] = {}
        for item in cross_keys:
            entity = str(item.get("entity", "")).strip()
            if not entity:
                continue
            mapping.setdefault(entity, {"rdb_tables": set(), "gdb_labels": set()})
            rdb_table = str(item.get("rdb", {}).get("table", "")).strip()
            gdb_label = str(item.get("gdb", {}).get("label", "")).strip()
            if rdb_table:
                mapping[entity]["rdb_tables"].add(rdb_table)
            if gdb_label:
                mapping[entity]["gdb_labels"].add(gdb_label)

        out: Dict[str, Any] = {}
        for entity, item in mapping.items():
            out[entity] = {
                "rdb_tables": sorted(item["rdb_tables"]),
                "gdb_labels": sorted(item["gdb_labels"]),
            }
        return out

    @staticmethod
    def _build_attribute_mapping_from_cross_source_keys(cross_keys: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in cross_keys:
            rows.append(
                {
                    "entity": item.get("entity"),
                    "rdb_table": item.get("rdb", {}).get("table"),
                    "rdb_column": item.get("rdb", {}).get("column"),
                    "gdb_label": item.get("gdb", {}).get("label"),
                    "gdb_property": item.get("gdb", {}).get("property"),
                    "relation": item.get("relation", "equivalent_key"),
                    "confidence": item.get("confidence", 0.9),
                }
            )
        return rows

    @staticmethod
    def _build_join_hints_from_cross_source_keys(cross_keys: List[Dict[str, Any]]) -> List[str]:
        hints: List[str] = []
        seen: Set[str] = set()
        for item in cross_keys:
            rdb_table = str(item.get("rdb", {}).get("table", "")).strip()
            rdb_column = str(item.get("rdb", {}).get("column", "")).strip()
            gdb_label = str(item.get("gdb", {}).get("label", "")).strip()
            gdb_prop = str(item.get("gdb", {}).get("property", "")).strip()
            if not (rdb_table and rdb_column and gdb_label and gdb_prop):
                continue
            hint = f"{rdb_table}.{rdb_column} <-> {gdb_label}.{gdb_prop}"
            key = hint.lower()
            if key in seen:
                continue
            seen.add(key)
            hints.append(hint)
        return hints

    @staticmethod
    def _filter_cross_source_keys_for_entities(
        cross_keys: List[Dict[str, Any]],
        entities: List[str],
        max_items: int = 120,
    ) -> List[Dict[str, Any]]:
        if not entities:
            return cross_keys[:max_items]

        lowered = set(str(e).lower() for e in entities)
        selected: List[Dict[str, Any]] = []
        for item in cross_keys:
            entity = str(item.get("entity", "")).lower()
            table = str(item.get("rdb", {}).get("table", "")).lower()
            label = str(item.get("gdb", {}).get("label", "")).lower()
            if entity in lowered or table in lowered or label in lowered:
                selected.append(item)
            if len(selected) >= max_items:
                break
        return selected

    def generate_rdb_schema_info(self) -> Dict[str, Any]:
        tables: Dict[str, Any] = {}
        for table_name in self.rdb.get_table_names():
            try:
                schema_text = self.rdb.get_table_info([table_name])
            except Exception as exc:
                schema_text = f"Failed to read schema: {exc}"
            try:
                sample_rows = self.rdb.run(f"SELECT * FROM {table_name} LIMIT {self.sample_limit}")
            except Exception as exc:
                sample_rows = f"Failed to read sample rows: {exc}"
            tables[table_name] = {
                "schema": schema_text,
                "sample_data": sample_rows,
            }
        return tables

    def generate_gdb_schema_info(self) -> Dict[str, Any]:
        with self.neo4j_driver.session() as session:
            labels_data = session.run("CALL db.labels() YIELD label RETURN label").data()
            rels_data = (
                session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").data()
            )

            node_labels = [record["label"] for record in labels_data]
            relationship_types = [record["relationshipType"] for record in rels_data]

            node_properties: Dict[str, List[str]] = {}
            node_property_types: Dict[str, Dict[str, str]] = {}
            node_samples: Dict[str, List[Dict[str, Any]]] = {}

            for label in node_labels:
                safe_label = f"`{label}`"
                props_rows = session.run(
                    f"MATCH (n:{safe_label}) RETURN keys(n) AS props LIMIT 1"
                ).data()
                node_properties[label] = props_rows[0]["props"] if props_rows else []

                sample_rows = session.run(
                    f"MATCH (n:{safe_label}) RETURN n LIMIT {self.sample_limit}"
                ).data()
                samples: List[Dict[str, Any]] = []
                for row in sample_rows:
                    node_dict = dict(row["n"])
                    samples.append(_to_json_serializable(node_dict))
                node_samples[label] = samples

                # Infer lightweight property type hints from sampled values.
                inferred_types: Dict[str, str] = {}
                for prop in node_properties[label]:
                    observed: List[str] = []
                    for sample in samples:
                        if isinstance(sample, dict) and prop in sample:
                            observed.append(_infer_simple_type(sample.get(prop)))
                    if observed:
                        non_null = [t for t in observed if t != "null"]
                        votes = non_null if non_null else observed
                        inferred_types[prop] = max(set(votes), key=votes.count)
                    else:
                        inferred_types[prop] = "unknown"
                node_property_types[label] = inferred_types

        return {
            "node_labels": node_labels,
            "relationship_types": relationship_types,
            "node_properties": node_properties,
            "node_property_types": node_property_types,
            "node_samples": node_samples,
        }

    def generate_meta_kg(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self.meta_kg and not force_refresh:
            return self.meta_kg

        rdb_info = self.generate_rdb_schema_info()
        gdb_info = self.generate_gdb_schema_info()

        prompt = (
            f"{PROMPTS['meta_kg']}\n\n"
            "RDB_SCHEMA:\n"
            f"{json.dumps(rdb_info, indent=2, ensure_ascii=False)}\n\n"
            "GDB_SCHEMA:\n"
            f"{json.dumps(gdb_info, indent=2, ensure_ascii=False)}\n"
        )

        try:
            raw = self.llm.invoke(prompt, json_mode=True)
            generated = _extract_json_object(raw)
        except Exception:
            generated = {}

        meta_kg = copy.deepcopy(generated)
        meta_kg.setdefault("entity_mapping", {})
        meta_kg.setdefault("attribute_mapping", {})
        meta_kg.setdefault("join_hints", [])
        meta_kg.setdefault("query_hints", {})
        meta_kg.setdefault("temporal_hints", [])
        derived_cross_keys = self._derive_cross_source_keys(rdb_info, gdb_info)
        merged_cross_keys = self._merge_cross_source_keys(meta_kg.get("cross_source_keys"), derived_cross_keys)
        meta_kg["cross_source_keys"] = merged_cross_keys

        if not meta_kg.get("entity_mapping"):
            meta_kg["entity_mapping"] = self._build_entity_mapping_from_cross_source_keys(merged_cross_keys)
        if not meta_kg.get("attribute_mapping"):
            meta_kg["attribute_mapping"] = self._build_attribute_mapping_from_cross_source_keys(merged_cross_keys)

        existing_hints = meta_kg.get("join_hints", [])
        if not isinstance(existing_hints, list):
            existing_hints = []
        derived_hints = self._build_join_hints_from_cross_source_keys(merged_cross_keys)
        hint_seen: Set[str] = set()
        joined_hints: List[str] = []
        for hint in list(existing_hints) + list(derived_hints):
            s = str(hint).strip()
            if not s:
                continue
            key = s.lower()
            if key in hint_seen:
                continue
            hint_seen.add(key)
            joined_hints.append(s)
        meta_kg["join_hints"] = joined_hints

        meta_kg["rdb"] = rdb_info
        meta_kg["gdb"] = gdb_info
        meta_kg["generated_at"] = dt.datetime.utcnow().isoformat() + "Z"

        self.meta_kg = meta_kg
        return meta_kg

    def inject_schema_context(self, user_query: str, table_names: Optional[List[str]] = None) -> str:
        if not self.meta_kg:
            self.generate_meta_kg()

        rdb_info = self.meta_kg.get("rdb", {})
        gdb_info = self.meta_kg.get("gdb", {})

        def _compact_row(row: Any, max_fields: int = 10) -> Dict[str, Any]:
            if not isinstance(row, dict):
                return {}
            preferred = [
                "patientId",
                "admissionId",
                "admissionId",
                "admissionDate",
                "visitDate",
                "reportDate",
                "prescriptionDate",
            ]
            ordered_keys: List[str] = []
            for key in preferred:
                if key in row:
                    ordered_keys.append(key)
            for key in row.keys():
                k = str(key)
                if k not in ordered_keys:
                    ordered_keys.append(k)
            return {k: row.get(k) for k in ordered_keys[:max_fields]}

        def _extract_rdb_columns(table_info: Any, max_cols: int = 30) -> List[str]:
            if not isinstance(table_info, dict):
                return []
            columns: List[str] = []
            schema_text = str(table_info.get("schema", "") or "")
            skip_prefixes = (
                "CREATE ",
                "PRIMARY ",
                "KEY ",
                "UNIQUE ",
                "CONSTRAINT ",
                "FOREIGN ",
                "INDEX ",
                ")",
            )
            for raw_line in schema_text.splitlines():
                line = raw_line.strip().rstrip(",")
                if not line:
                    continue
                upper = line.upper()
                if upper.startswith(skip_prefixes):
                    continue
                match = re.match(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s+[A-Za-z]", line)
                if match:
                    columns.append(match.group(1))
            if not columns:
                sample_data = table_info.get("sample_data")
                if isinstance(sample_data, list):
                    for row in sample_data[:2]:
                        if isinstance(row, dict):
                            columns.extend([str(k) for k in row.keys()])
            deduped: List[str] = []
            seen: Set[str] = set()
            for col in columns:
                key = str(col).strip()
                if not key:
                    continue
                low = key.lower()
                if low in seen:
                    continue
                seen.add(low)
                deduped.append(key)
            return deduped[:max_cols]

        def _truncate_property_map(prop_map: Any, max_labels: int = 20, max_props: int = 24) -> Dict[str, Any]:
            if not isinstance(prop_map, dict):
                return {}
            out: Dict[str, Any] = {}
            for label in list(prop_map.keys())[:max_labels]:
                props = prop_map.get(label, [])
                if isinstance(props, list):
                    out[str(label)] = [str(p) for p in props[:max_props]]
            return out

        if not table_names:
            cross_keys = self.meta_kg.get("cross_source_keys", [])
            cross_keys = cross_keys if isinstance(cross_keys, list) else []
            join_hints = self.meta_kg.get("join_hints", [])
            join_hints = join_hints if isinstance(join_hints, list) else []
            table_columns = {
                str(table): _extract_rdb_columns(info, max_cols=24)
                for table, info in list(rdb_info.items())[:40]
            }
            summary = {
                "available_tables": sorted(list(rdb_info.keys())),
                "available_nodes": gdb_info.get("node_labels", []),
                "available_relationships": gdb_info.get("relationship_types", []),
                "table_columns": table_columns,
                "node_property_examples": _truncate_property_map(gdb_info.get("node_properties", {})),
                "node_property_types": gdb_info.get("node_property_types", {}),
                "join_hints": join_hints[:40],
                "cross_source_key_count": len(cross_keys),
                "cross_source_keys": cross_keys[:40],
                "topology_hint": (
                    "For event queries, prefer Patient-[:HAS_ADMISSION]->Admission-[:NEXT*0..N]->Day-[:HAS_*]->Event."
                ),
            }
            return (
                "Schema Context (Summary)\n"
                f"User Query: {user_query}\n\n"
                f"{json.dumps(summary, indent=2, ensure_ascii=False)}\n\n"
                "Use only names that appear in this schema context."
            )

        normalized_rdb = {name.lower(): name for name in rdb_info.keys()}
        normalized_gdb = {name.lower(): name for name in gdb_info.get("node_labels", [])}
        detailed: Dict[str, Any] = {}

        for requested_name in table_names:
            key = requested_name.strip()
            if not key:
                continue
            lowered = key.lower()

            if lowered in normalized_rdb:
                real_name = normalized_rdb[lowered]
                sample_data = rdb_info[real_name].get("sample_data")
                sample_row = {}
                if isinstance(sample_data, list) and sample_data:
                    sample_row = _compact_row(sample_data[0], max_fields=10)
                detailed[key] = {
                    "type": "RDB_TABLE",
                    "columns": _extract_rdb_columns(rdb_info[real_name], max_cols=30),
                    "sample_row": sample_row,
                }
                continue

            if lowered in normalized_gdb:
                label = normalized_gdb[lowered]
                node_samples = gdb_info.get("node_samples", {}).get(label, [])
                sample_node = _compact_row(node_samples[0], max_fields=10) if isinstance(node_samples, list) and node_samples else {}
                detailed[key] = {
                    "type": "GDB_NODE",
                    "properties": [str(p) for p in gdb_info.get("node_properties", {}).get(label, [])[:30]],
                    "property_types": gdb_info.get("node_property_types", {}).get(label, {}),
                    "sample_node": sample_node,
                }
                continue

            detailed[key] = {"type": "NOT_FOUND", "message": "No matching table or node found in schema."}

        detailed["_GDB_RELATIONSHIPS"] = gdb_info.get("relationship_types", [])[:80]
        cross_keys = self.meta_kg.get("cross_source_keys", [])
        cross_keys = cross_keys if isinstance(cross_keys, list) else []
        detailed["_CROSS_SOURCE_KEYS"] = self._filter_cross_source_keys_for_entities(cross_keys, table_names, max_items=30)
        return (
            "Schema Context (Detailed)\n"
            f"User Query: {user_query}\n\n"
            f"{json.dumps(detailed, indent=2, ensure_ascii=False)}\n\n"
            "Use only names that appear in this schema context."
        )


class QueryPlanner:
    """Generate executable candidate query plans."""

    def __init__(self, llm: OpenAILLM, constants: Optional[ComplexityConstants] = None) -> None:
        self.llm = llm
        self.constants = constants or ComplexityConstants()

    def generate_plans(
        self,
        user_query: str,
        schema_context: str,
        decomposed_info: str = "",
    ) -> Dict[str, Any]:
        prompt = f"""
You are a query planner for a hybrid RDB + GraphDB clinical QA agent.
Generate EXACTLY 2 candidate plans (A and B): one strong primary and one robust fallback.
Return JSON only.

USER_QUERY:
{user_query}

DECOMPOSED_INFO:
{decomposed_info if decomposed_info else "single question"}

SCHEMA_CONTEXT:
{schema_context}

Critical routing rules:
1) Use only names that exist in SCHEMA_CONTEXT.
2) patientId is the MIMIC patient key and should be treated as a quoted string in Cypher and SQL literals.
3) Temporal/sequence wording such as "first N days", "within N days", "timeline", "chronological", "followed by", "then", "before/after":
   prefer GDB Day timeline traversal:
   (Patient)-[:HAS_ADMISSION]->(Admission)-[:NEXT*0..N]->(Day)-[:HAS_*]->(Event).
4) For sequence queries with two events, anchor second hop from first event day:
   use (d1)-[:NEXT*0..N]->(d2), not admission-wide re-expansion.
5) If the question asks multiple independent metrics (e.g., admission count + diagnosis distinct + medication count),
   prefer BOTH with independently executable Step 1/Step 2 queries.
6) For BOTH, each step must be executable alone and must not reference pseudo variables from prior steps.
7) For BOTH format, use exact headers:
   Step 1) RDB:
   <single SQL statement>
   Step 2) GDB:
   <single Cypher statement>
   Never output unlabeled forms like "Step 1) MATCH ...; Step 2) SELECT ...".
8) For single-source plans (RDB/GDB), query must be one executable statement only and must NOT contain step labels.
9) Never emit unresolved placeholders ($x, ?x, <x>) or narrative text inside query.
10) Prevent join fan-out overcounting:
    if two or more one-to-many tables are joined and the question asks independent count metrics,
    compute each metric in its own subquery/CTE (or separate step) and combine only final scalars.
11) In such fan-out cases, NEVER use raw COUNT(*) or COUNT(child_col) on the exploded joined rows.
    Prefer per-metric aggregation first; use COUNT(DISTINCT key) only when the key truly matches the target metric grain.
12) SQL anti-pattern to avoid in fan-out count tasks:
    FROM A LEFT JOIN B ... LEFT JOIN C ... SELECT COUNT(B.*), COUNT(C.*)
13) For day-level timeline questions, choose an Admission with admissionId, then traverse (a:Admission)-[:NEXT*0..N]->(d:Day). Day nodes carry patientId and admissionId.
14) For latest-admission selection, use only date fields confirmed in SCHEMA_CONTEXT/sample_data.
    If both appear valid, prefer deterministic fallback ordering (e.g., admitDate then admissionDate).
15) In GDB event counting from Day, avoid over-restrictive property equality filters on event nodes
    (e.g., diag.admissionId = a.admissionId) unless those properties are explicitly confirmed in SCHEMA_CONTEXT.
16) For population temporal counts, prefer counting DISTINCT admission key/patient key from Day-window matches.
17) If topology evidence for event edges is missing, choose safer source/type rather than inventing edges.
18) For medication questions, use Day-mediated existence: (Admission)-[:NEXT*0..N]->(Day)-[:HAS_MEDI]->(Medi). Do not use (Admission)-[:HAS_MEDI]->(Medi).
19) MySQL CTE syntax must be valid: WITH cte AS (SELECT ...) SELECT ... ; never "WITH Table AS alias SELECT ...".
20) Cypher variable scope must be valid: every variable used in WHERE/RETURN/ORDER BY must remain bound via MATCH/WITH.
21) Do not concatenate multiple statements with semicolons in a single query/step.
22) For variable-length paths in Cypher, use concrete numeric bounds only (e.g., *0..100). Never use symbolic bounds like N.
23) For all-admission patient-level lab/medi/vital/op/phys counts, do not join to admin by admissionId unless the question is explicitly admission-specific. In this MIMIC RDB, lab/medi admissionId values may be stored as decimal-like strings, so a direct admin.admissionId join can incorrectly drop all rows. Use the clinical table's patientId filter for all-hospital-admission counts.
24) MIMIC table names are lower-case: patient, admin, diag, lab, medi, op, phys, vital.
25) In this MIMIC GDB, day-level event retrieval must use Admission/Day NEXT traversal. Do not use Admission-[:HAS_DAY]->Day for lab/medication/vital/physical/operation queries because it can produce empty results.
26) For GDB day-level events, do not add event.admissionId filters by default. The NEXT path already preserves admission scope; add event.admissionId only when schema and value format alignment are confirmed.
27) GDB Diagnosis contains seq_num 1-5 primary diagnoses; align RDB diag with seq_num IN (1,2,3,4,5) for cross-source diagnosis comparisons.
28) Property casing: Lab/Lab table uses testName; Vital/Physical use named columns; Medi uses medication; Operation uses operationName; Diagnosis uses diagnosisName/icd_code.
29) Prefer RDB admin for admission_type/race/insurance/language/marital_status/discharge_location/hospital_expire_flag/los; avoid AdminRecord unless explicitly needed.
30) Cypher scope safety: every variable in RETURN/ORDER BY/WHERE must be bound in the current scope after WITH. Never RETURN label unless label was explicitly bound with WITH ... AS label.
31) Do not invent temporary map variables such as lab_entry/result_entry/entry. If using x.y, x must be a bound node variable or a map variable explicitly introduced by WITH/UNWIND. Prefer flat aliases: WITH lab.testName AS metric, count(*) AS value RETURN metric, value. If using maps, UNWIND rows AS row before RETURN row.metric/row.value.
32) For Cypher length-of-stay/date arithmetic, do not use datetime(a.admitDate) on date-only MIMIC values. Prefer RDB admin.los; if Cypher is needed, use duration.inDays(date(a.admitDate), date(a.dischargeDate)).days.
33) In RDB medi, rowId is invalid. Use COUNT(*), COUNT(DISTINCT medication), or only schema-confirmed columns for composite DISTINCT.
34) In MySQL, top-k subqueries with LIMIT 3 cannot be scalar SELECT columns. Use CTE + GROUP_CONCAT/JSON_ARRAYAGG or return top-k rows.
35) Use MetaKG as the authority for concept fields. For MIMIC concept-name filters/returns: diag/Diagnosis uses diagnosisName for diagnosis text; lab/Lab uses testName for lab names; medi/Medi uses genericName for drug names; op/Operation uses operationName for operation text; phys/Physical and vital/Vital use named numeric columns (heartRate, systolicBp, height, weight, etc.).
36) Do not substitute generic or case-changed fields across entities. Do not use lab.label, medi.label, diag.label, op.label, vital.testName, or phys.testName unless SCHEMA_CONTEXT explicitly lists that exact field for that exact entity.
37) For diagnosis GDB queries, an Admission variable declared earlier may be reused as (a)-[:HAS_DIAG]->(diag:Diagnosis).

Reference query patterns (adapt names strictly to SCHEMA_CONTEXT):
- Temporal admission count:
  MATCH (p:Patient)-[:HAS_ADMISSION]->(a:Admission)-[:HAS_DIAG]->(diag:Diagnosis)
  RETURN COUNT(DISTINCT a.admissionId) AS admission_count
- Sequence constraint:
  MATCH ... (d1:Day)-[:HAS_LAB]->(lab)
  MATCH (d1)-[:NEXT*0..10]->(d2:Day)-[:HAS_MEDI]->(med)
  RETURN ...
- Independent metric bundle:
  Step 1) RDB: compute each metric in separate subquery/CTE then combine.
  Step 2) GDB: optional graph-only complementary evidence if needed.

Complexity constants:
J_c={self.constants.J_c}, H_c={self.constants.H_c}, F_c_low={self.constants.F_c_low},
F_c_high={self.constants.F_c_high}, G_c={self.constants.G_c}

Output schema:
{{
  "plans": [
    {{
      "plan_id": "A",
      "description": "short",
      "db_type": "RDB|GDB|BOTH",
      "query": "executable SQL/Cypher or Step-form BOTH query",
      "is_multi_step": false,
      "predicted_error_risk": 0.0,
      "complexity_factors": {{
        "C_relation": 0,
        "C_filter": 0,
        "C_agg": 0,
        "omega": 1
      }}
    }},
    {{
      "plan_id": "B",
      "description": "short",
      "db_type": "RDB|GDB|BOTH",
      "query": "executable SQL/Cypher or Step-form BOTH query",
      "is_multi_step": false,
      "predicted_error_risk": 0.0,
      "complexity_factors": {{
        "C_relation": 0,
        "C_filter": 0,
        "C_agg": 0,
        "omega": 1
      }}
    }}
  ]
}}
"""
        raw = self.llm.invoke(prompt, json_mode=True)
        plans = _extract_json_object(raw)
        if "plans" not in plans or not isinstance(plans["plans"], list):
            raise ValueError("Planner output does not contain a valid plans list.")
        if len(plans["plans"]) < 2:
            raise ValueError("Planner must return at least 2 plans.")
        return plans


class CAQAEvaluator:
    """Calculate CAQA scores and choose the best query plan."""

    def __init__(self, rdb: Any, neo4j_driver: Any, default_cardinality: int = 1000) -> None:
        self.rdb = rdb
        self.neo4j_driver = neo4j_driver
        self.default_cardinality = default_cardinality

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _infer_query_intent(user_query: str) -> Dict[str, Any]:
        text = str(user_query or "").lower()
        event_terms = [
            "diagnosis",
            "diag",
            "lab",
            "medication",
            "medi",
            "prescription",
            "drug",
            "vital",
            "operation",
            "physical",
        ]
        event_hits = sorted({term for term in event_terms if term in text})

        has_temporal = bool(
            re.search(
                r"\b(first|within|before|after|timeline|chronological|recent|earliest|latest)\b",
                text,
            )
        )
        has_window = bool(re.search(r"\b\d+\s*(day|days|week|weeks|month|months)\b", text))
        has_sequence = any(token in text for token in ("followed by", " then ", "after that", "subsequent"))

        metric_terms = ["count", "total", "distinct", "top", "average", "mean"]
        metric_hits = sum(1 for token in metric_terms if token in text)

        asks_admission = "admission" in text
        asks_cross_metric = asks_admission and len(event_hits) >= 2 and metric_hits >= 2
        asks_hybrid_phrase = "along with" in text or "together with" in text

        prefer_gdb_timeline = (has_temporal or has_window or has_sequence) and len(event_hits) >= 1
        prefer_both = (asks_cross_metric or asks_hybrid_phrase) and not has_sequence

        return {
            "prefer_gdb_timeline": prefer_gdb_timeline,
            "prefer_both": prefer_both,
            "has_sequence": has_sequence,
            "event_hit_count": len(event_hits),
        }

    @staticmethod
    def _has_day_next_pattern(query: str) -> bool:
        q = str(query or "")
        return bool(
            re.search(r":\s*`?Day`?", q, flags=re.IGNORECASE)
            and re.search(r":\s*`?NEXT`?", q, flags=re.IGNORECASE)
        )

    @classmethod
    def _semantic_selection_penalty(cls, plan: Dict[str, Any], user_query: str) -> float:
        intent = cls._infer_query_intent(user_query)
        db_type = str(plan.get("db_type", "RDB")).upper()
        query = str(plan.get("query", "") or "")
        is_multi_step = bool(plan.get("is_multi_step", False))
        penalty = 0.0

        if intent.get("prefer_gdb_timeline", False):
            if db_type == "RDB":
                penalty += 1.05
            elif db_type in ("GDB", "BOTH") and not cls._has_day_next_pattern(query):
                penalty += 0.55

        if intent.get("prefer_both", False):
            if db_type in ("RDB", "GDB"):
                penalty += 0.90
            elif db_type == "BOTH" and not is_multi_step:
                penalty += 0.40

        # Sequence queries often fail when the second event window restarts from Admission
        # instead of chaining from the first event day (d1 -> NEXT -> d2).
        if intent.get("has_sequence", False) and db_type in ("GDB", "BOTH"):
            has_d1_d2 = bool(re.search(r"\bd1\b", query, flags=re.IGNORECASE) and re.search(r"\bd2\b", query, flags=re.IGNORECASE))
            admission_wide_reexpand = bool(
                re.search(r"\(\s*[aA][^)]*\)\s*-\s*\[[^]]*:\s*`?NEXT`?", query, flags=re.IGNORECASE)
            )
            day_chain = bool(
                re.search(r"\(\s*d1[^)]*\)\s*-\s*\[[^]]*:\s*`?NEXT`?", query, flags=re.IGNORECASE)
            )
            if has_d1_d2 and admission_wide_reexpand and not day_chain:
                penalty += 0.35

        return min(1.50, penalty)

    @classmethod
    def _semantic_risk_bonus(cls, plan: Dict[str, Any], user_query: str) -> float:
        # Convert selection-level semantic mismatch into risk-space as well.
        penalty = cls._semantic_selection_penalty(plan, user_query)
        return min(0.60, penalty * 0.45)

    @classmethod
    def _query_pattern_risk_bonus(cls, plan: Dict[str, Any], user_query: str = "") -> float:
        """Add conservative risk for known run-but-wrong query patterns."""
        db_type = str(plan.get("db_type", "RDB")).upper()
        query = str(plan.get("query", "") or "")
        if not query:
            return 0.0

        q = query
        bonus = 0.0

        if db_type in ("GDB", "BOTH"):
            optional_cnt = len(re.findall(r"\bOPTIONAL\s+MATCH\b", q, flags=re.IGNORECASE))
            if optional_cnt >= 2:
                bonus += 0.06

            has_event_rel = bool(
                re.search(r"HAS_(DIAG|LAB|MEDI|VITAL|OPERATION|PHYSICAL)", q, flags=re.IGNORECASE)
            )
            has_count = bool(re.search(r"\bcount\s*\(", q, flags=re.IGNORECASE))
            has_count_distinct = bool(re.search(r"\bcount\s*\(\s*distinct\b", q, flags=re.IGNORECASE))
            if has_event_rel and has_count and not has_count_distinct:
                bonus += 0.12

            if re.search(r"\bORDER\s+BY\s+count\s*\(", q, flags=re.IGNORECASE):
                bonus += 0.08

            has_patient_literal = bool(
                re.search(r"patientId\s*[:=]\s*'[^']+'", q, flags=re.IGNORECASE)
            )
            if has_event_rel and has_count and not has_patient_literal:
                bonus += 0.08

            has_admin_field = bool(
                re.search(r"\b(los|admission_type|hospital_expire_flag|anchor_age)\b", q, flags=re.IGNORECASE)
            )
            has_admin_path = bool(re.search(r"HAS_ADMIN_RECORD|:AdminRecord", q, flags=re.IGNORECASE))
            if has_admin_field and not has_admin_path:
                bonus += 0.15

        if db_type == "BOTH":
            if not re.search(r"Step\s*\d+\)\s*(RDB|GDB)\s*:", q, flags=re.IGNORECASE):
                bonus += 0.08

        bonus += cls._semantic_risk_bonus(plan, user_query)
        return min(0.70, bonus)

    def _estimate_error_risk(
        self,
        plan: Dict[str, Any],
        provided_risk: Optional[float] = None,
        user_query: str = "",
    ) -> float:
        """
        Estimate per-plan error risk in [0, 1].
        Priority:
        1) externally provided risk (predicted_risks map)
        2) plan['predicted_error_risk'] from planner output
        3) fallback heuristic from plan structure
        """
        db_type = str(plan.get("db_type", "RDB")).upper()
        is_multi_step = bool(plan.get("is_multi_step", False))
        factors = plan.get("complexity_factors", {})
        omega = float(factors.get("omega", 1.0) or 1.0)

        omega_risk = self._clip01((omega - 1.0) / 10.0)
        db_risk = 0.15 if db_type == "BOTH" else 0.12 if db_type == "GDB" else 0.08
        step_risk = 0.15 if is_multi_step else 0.0
        pattern_risk = self._query_pattern_risk_bonus(plan, user_query=user_query)

        # Heuristic fallback. Keep conservative baseline instead of zero.
        heuristic_estimated = self._clip01(0.25 + 0.45 * omega_risk + db_risk + step_risk + pattern_risk)

        if provided_risk is not None:
            # Keep externally supplied uncertainty as the highest priority signal.
            return self._clip01(provided_risk)

        raw_risk = plan.get("predicted_error_risk")
        if isinstance(raw_risk, (int, float)):
            planner_risk = self._clip01(float(raw_risk))
            # Blend planner-reported risk with structural heuristic to reduce underestimation.
            blended = 0.65 * planner_risk + 0.35 * heuristic_estimated
            return self._clip01(max(blended, heuristic_estimated * 0.85))

        return heuristic_estimated

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _count_pattern(pattern: str, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL))

    @staticmethod
    def _split_top_level_csv(expr: str) -> List[str]:
        items: List[str] = []
        buffer: List[str] = []
        depth = 0
        for ch in expr:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                token = "".join(buffer).strip()
                if token:
                    items.append(token)
                buffer = []
            else:
                buffer.append(ch)
        tail = "".join(buffer).strip()
        if tail:
            items.append(tail)
        return items

    @staticmethod
    def _count_predicates(expr: str) -> int:
        if not expr or not expr.strip():
            return 0
        comparator = r"(=|<>|!=|<=|>=|<|>|\bIN\b|\bLIKE\b|\bBETWEEN\b|\bIS\b|\bCONTAINS\b|\bSTARTS\s+WITH\b|\bENDS\s+WITH\b)"
        parts = re.split(r"\bAND\b|\bOR\b", expr, flags=re.IGNORECASE)
        count = 0
        for part in parts:
            if re.search(comparator, part, flags=re.IGNORECASE):
                count += 1
        if count == 0 and re.search(comparator, expr, flags=re.IGNORECASE):
            count = 1
        return count

    def _estimate_sql_projection_width(self, query: str) -> int:
        match = re.search(r"\bSELECT\b(.*?)(?:\bFROM\b)", query, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return 0
        segment = match.group(1).strip()
        if not segment:
            return 0
        segment = re.sub(r"\bDISTINCT\b", "", segment, flags=re.IGNORECASE).strip()
        if segment == "*":
            return 5
        cols = self._split_top_level_csv(segment)
        if not cols:
            return 1
        return min(len(cols), 12)

    def _estimate_cypher_projection_width(self, query: str) -> int:
        return_match = re.search(
            r"\bRETURN\b(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|\bSKIP\b|;|$)",
            query,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not return_match:
            return 0
        segment = return_match.group(1).strip()
        if not segment:
            return 0
        segment = re.sub(r"\bDISTINCT\b", "", segment, flags=re.IGNORECASE).strip()
        cols = self._split_top_level_csv(segment)
        if not cols:
            return 1
        return min(len(cols), 12)

    def _estimate_function_calls(self, query: str) -> int:
        if not query:
            return 0
        calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", query)
        if not calls:
            return 0
        blacklist = {
            "SELECT",
            "FROM",
            "WHERE",
            "MATCH",
            "OPTIONAL",
            "RETURN",
            "WITH",
            "CALL",
            "CASE",
            "WHEN",
            "THEN",
            "ELSE",
            "END",
        }
        valid = [name for name in calls if name.upper() not in blacklist]
        return min(len(valid), 20)

    def _estimate_structural_factors(
        self, query: str, db_type: str, is_multi_step: bool
    ) -> Dict[str, float]:
        q = str(query or "")
        db = str(db_type or "RDB").upper()
        is_rdb = db in ("RDB", "BOTH")
        is_gdb = db in ("GDB", "BOTH")

        factors = {
            "C_join_or_hop": 0.0,
            "C_filter_pred": 0.0,
            "C_subquery": 0.0,
            "C_sort_distinct": 0.0,
            "C_optional": 0.0,
            "C_path_varlen": 0.0,
            "C_func": 0.0,
            "C_proj_width": 0.0,
            "C_multi_step": 1.0 if is_multi_step else 0.0,
            "C_agg": 0.0,
        }

        if is_rdb:
            factors["C_join_or_hop"] += float(self._count_pattern(r"\bJOIN\b", q))

            where_clauses = re.findall(
                r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bUNION\b|;|$)",
                q,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for clause in where_clauses:
                factors["C_filter_pred"] += float(self._count_predicates(clause))

            sql_agg = self._count_pattern(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", q)
            sql_group = 1 if re.search(r"\bGROUP\s+BY\b", q, flags=re.IGNORECASE) else 0
            factors["C_agg"] += float(sql_agg + sql_group)

            cte_count = self._count_pattern(r"\bWITH\b\s+[A-Za-z_][A-Za-z0-9_]*\s+AS\s*\(", q)
            nested_select = self._count_pattern(r"\(\s*SELECT\b", q)
            factors["C_subquery"] += float(cte_count + nested_select)

            sql_sort = 1 if re.search(r"\bORDER\s+BY\b", q, flags=re.IGNORECASE) else 0
            sql_distinct = 1 if re.search(r"\bSELECT\s+DISTINCT\b", q, flags=re.IGNORECASE) else 0
            sql_union = self._count_pattern(r"\bUNION(?:\s+ALL)?\b", q)
            factors["C_sort_distinct"] += float(sql_sort + sql_distinct + sql_union)

            outer_join = self._count_pattern(r"\b(?:LEFT|RIGHT|FULL)\s+(?:OUTER\s+)?JOIN\b", q)
            factors["C_optional"] += float(outer_join)

            factors["C_proj_width"] += float(self._estimate_sql_projection_width(q))

        if is_gdb:
            hop_count = self._count_pattern(r"-\s*\[[^\]]*\]\s*->|<-\s*\[[^\]]*\]\s*-|-\s*\[[^\]]*\]\s*-", q)
            factors["C_join_or_hop"] += float(hop_count)

            varlen = self._count_pattern(r"\*\s*\d*\s*\.\.\s*\d*", q)
            factors["C_path_varlen"] += float(varlen)

            optional_match = self._count_pattern(r"\bOPTIONAL\s+MATCH\b", q)
            factors["C_optional"] += float(optional_match)

            where_clauses = re.findall(
                r"\bWHERE\b(.*?)(?:\bRETURN\b|\bWITH\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)",
                q,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for clause in where_clauses:
                factors["C_filter_pred"] += float(self._count_predicates(clause))

            pattern_maps = self._count_pattern(r"\{\s*[A-Za-z_][A-Za-z0-9_]*\s*:", q)
            factors["C_filter_pred"] += float(pattern_maps)

            cypher_agg = self._count_pattern(
                r"\b(?:count|sum|avg|min|max|collect|percentileCont|percentileDisc|stDev|stDevP)\s*\(",
                q,
            )
            factors["C_agg"] += float(cypher_agg)

            call_subquery = self._count_pattern(r"\bCALL\s*\{", q)
            factors["C_subquery"] += float(call_subquery)

            cypher_sort = 1 if re.search(r"\bORDER\s+BY\b", q, flags=re.IGNORECASE) else 0
            cypher_distinct = self._count_pattern(r"\bDISTINCT\b", q)
            factors["C_sort_distinct"] += float(cypher_sort + cypher_distinct)

            factors["C_proj_width"] += float(self._estimate_cypher_projection_width(q))

        factors["C_func"] = float(self._estimate_function_calls(q))
        return factors

    def _compute_enhanced_complexity(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        raw_factors_input = plan.get("complexity_factors", {})
        raw_factors = copy.deepcopy(raw_factors_input) if isinstance(raw_factors_input, dict) else {}
        query = str(plan.get("query", ""))
        db_type = str(plan.get("db_type", "RDB")).upper()
        is_multi_step = bool(plan.get("is_multi_step", False))

        c_relation_planner = max(0.0, self._safe_float(raw_factors.get("C_relation", 0.0)))
        c_filter_planner = max(0.0, self._safe_float(raw_factors.get("C_filter", 0.0)))
        c_agg_planner = max(0.0, self._safe_float(raw_factors.get("C_agg", 0.0)))
        omega_input = raw_factors.get("omega")

        legacy_omega = 1.0 + c_relation_planner + c_filter_planner + c_agg_planner
        planner_omega = self._safe_float(omega_input, legacy_omega) if isinstance(omega_input, (int, float)) else legacy_omega

        derived = self._estimate_structural_factors(query, db_type, is_multi_step)
        c_relation = max(c_relation_planner, float(derived.get("C_join_or_hop", 0.0)))
        c_filter = max(c_filter_planner, float(derived.get("C_filter_pred", 0.0)))
        c_agg = max(c_agg_planner, float(derived.get("C_agg", 0.0)))
        core_omega = 1.0 + c_relation + c_filter + c_agg

        extras = (
            0.80 * float(derived.get("C_subquery", 0.0))
            + 0.40 * float(derived.get("C_sort_distinct", 0.0))
            + 0.60 * float(derived.get("C_optional", 0.0))
            + 0.80 * float(derived.get("C_path_varlen", 0.0))
            + 0.50 * float(derived.get("C_func", 0.0))
            + 0.20 * float(derived.get("C_proj_width", 0.0))
            + 0.70 * float(derived.get("C_multi_step", 0.0))
        )
        enhanced_omega = max(planner_omega, legacy_omega, core_omega) + extras
        enhanced_omega = max(1.0, min(30.0, float(enhanced_omega)))

        merged = copy.deepcopy(raw_factors)
        merged["C_relation_planner"] = c_relation_planner
        merged["C_filter_planner"] = c_filter_planner
        merged["C_agg_planner"] = c_agg_planner
        merged["C_relation"] = float(c_relation)
        merged["C_filter"] = float(c_filter)
        merged["C_agg"] = float(c_agg)
        for key, value in derived.items():
            merged[key] = float(value)
        merged["omega_legacy"] = float(legacy_omega)
        merged["omega_planner"] = float(planner_omega)
        merged["omega_enhanced"] = float(enhanced_omega)
        merged["omega"] = float(enhanced_omega)
        return merged

    @staticmethod
    def _normalize_costs(costs: List[float]) -> List[float]:
        if not costs:
            return []
        min_c = min(costs)
        max_c = max(costs)
        if abs(max_c - min_c) <= 1e-12:
            return [0.0 for _ in costs]
        return [(c - min_c) / (max_c - min_c) for c in costs]

    def _estimate_rdb_cardinality(self, query: str) -> int:
        table_match = re.search(r"\bFROM\s+[`\"]?([A-Za-z0-9_]+)[`\"]?", query, flags=re.IGNORECASE)
        if not table_match:
            return self.default_cardinality

        table_name = table_match.group(1)
        try:
            result = self.rdb.run(f"SELECT COUNT(*) AS count FROM {table_name}")
            return max(_as_int(result, self.default_cardinality), 1)
        except Exception:
            return self.default_cardinality

    def _estimate_gdb_cardinality(self, query: str) -> int:
        label_match = re.search(r"\(\s*\w*\s*:\s*`?([A-Za-z0-9_]+)`?", query)
        if not label_match:
            return self.default_cardinality

        label = label_match.group(1)
        try:
            with self.neo4j_driver.session() as session:
                record = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS count").single()
            if record is None:
                return self.default_cardinality
            return max(_as_int(record["count"], self.default_cardinality), 1)
        except Exception:
            return self.default_cardinality

    def _estimate_cardinality(self, plan: Dict[str, Any], actual_cardinalities: Optional[Dict[str, int]]) -> int:
        plan_id = str(plan.get("plan_id", ""))
        if actual_cardinalities and plan_id in actual_cardinalities:
            return max(int(actual_cardinalities[plan_id]), 1)

        db_type = str(plan.get("db_type", "RDB")).upper()
        query = str(plan.get("query", ""))

        if db_type == "RDB":
            return self._estimate_rdb_cardinality(query)
        if db_type == "GDB":
            return self._estimate_gdb_cardinality(query)
        if db_type == "BOTH":
            rdb_guess = self._estimate_rdb_cardinality(query)
            gdb_guess = self._estimate_gdb_cardinality(query)
            return min(rdb_guess, gdb_guess)

        return self.default_cardinality

    def calculate_caqa_scores(
        self,
        plans_json: Union[Dict[str, Any], str],
        actual_cardinalities: Optional[Dict[str, int]] = None,
        predicted_risks: Optional[Dict[str, float]] = None,
        risk_lambda: float = 1.0,
        normalize_expected_cost: bool = True,
        user_query: str = "",
    ) -> Dict[str, Any]:
        plans_data = _extract_json_object(plans_json) if isinstance(plans_json, str) else copy.deepcopy(plans_json)
        if "plans" not in plans_data or not isinstance(plans_data["plans"], list):
            raise ValueError("Invalid plan input: 'plans' list is required.")

        expected_costs: List[float] = []
        for plan in plans_data["plans"]:
            factors = self._compute_enhanced_complexity(plan)
            omega = float(factors.get("omega", 1.0) or 1.0)
            cardinality = self._estimate_cardinality(plan, actual_cardinalities)
            expected_cost = omega * float(cardinality)
            plan_id = str(plan.get("plan_id", ""))
            provided_risk = None
            if predicted_risks and plan_id in predicted_risks:
                provided_risk = float(predicted_risks[plan_id])
            predicted_error_risk = self._estimate_error_risk(plan, provided_risk, user_query=user_query)
            semantic_penalty = self._semantic_selection_penalty(plan, user_query=user_query)

            factors["omega"] = omega
            plan["complexity_factors"] = factors
            plan["calculated_cardinality"] = cardinality
            plan["expected_cost"] = expected_cost
            plan["estimated_cost"] = expected_cost
            plan["caqa_score"] = expected_cost  # legacy compatibility
            plan["predicted_error_risk"] = predicted_error_risk
            plan["semantic_penalty"] = float(semantic_penalty)
            expected_costs.append(expected_cost)

        normalized_costs = self._normalize_costs(expected_costs) if normalize_expected_cost else expected_costs
        for idx, plan in enumerate(plans_data["plans"]):
            expected_cost = float(plan.get("expected_cost", 0.0))
            cost_for_objective = float(normalized_costs[idx]) if normalize_expected_cost else expected_cost
            predicted_error_risk = float(plan.get("predicted_error_risk", 0.0))
            semantic_penalty = float(plan.get("semantic_penalty", 0.0))
            risk_aware_score = cost_for_objective + float(risk_lambda) * predicted_error_risk + semantic_penalty

            plan["expected_cost_normalized"] = float(cost_for_objective)
            plan["risk_lambda"] = float(risk_lambda)
            plan["risk_aware_caqa_score"] = float(risk_aware_score)
            plan["selection_objective"] = (
                "expected_cost_normalized + risk_lambda * predicted_error_risk + semantic_penalty"
                if normalize_expected_cost
                else "expected_cost + risk_lambda * predicted_error_risk + semantic_penalty"
            )

        plans_data["caqa_config"] = {
            "risk_lambda": float(risk_lambda),
            "normalize_expected_cost": bool(normalize_expected_cost),
            "selection_objective": (
                "expected_cost_normalized + risk_lambda * predicted_error_risk + semantic_penalty"
                if normalize_expected_cost
                else "expected_cost + risk_lambda * predicted_error_risk + semantic_penalty"
            ),
        }

        return plans_data

    def select_best_plan(self, plans_json: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        plans_data = _extract_json_object(plans_json) if isinstance(plans_json, str) else plans_json
        plans = plans_data.get("plans", [])
        if not plans:
            raise ValueError("No plans found for selection.")

        score_key = "risk_aware_caqa_score"
        valid_plans = [p for p in plans if isinstance(p.get(score_key), (int, float))]
        if not valid_plans:
            score_key = "caqa_score"
            valid_plans = [p for p in plans if isinstance(p.get(score_key), (int, float))]
        if not valid_plans:
            raise ValueError("No valid plan has a numeric caqa_score.")

        best_plan = min(valid_plans, key=lambda p: p[score_key])
        return {
            "selected_plan": best_plan,
            "selected_plan_id": best_plan.get("plan_id"),
            "selected_plan_db_type": best_plan.get("db_type"),
            "selected_plan_query": best_plan.get("query"),
            "selected_plan_caqa_score": best_plan.get(score_key),
            "selected_plan_expected_cost": best_plan.get("expected_cost", best_plan.get("estimated_cost")),
            "selected_plan_predicted_error_risk": best_plan.get("predicted_error_risk"),
            "selected_score_key": score_key,
            "all_plans": plans,
        }


class QueryExecutor:
    """Execute SQL/Cypher query based on selected plan."""

    def __init__(self, rdb: Any, neo4j_driver: Any) -> None:
        self.rdb = rdb
        self.neo4j_driver = neo4j_driver

    @staticmethod
    def _normalize_step_db_type(raw: str) -> str:
        token = str(raw or "").strip().upper()
        if token in ("RDB", "SQL", "MYSQL"):
            return "RDB"
        if token in ("GDB", "CYPHER", "GRAPH", "NEO4J"):
            return "GDB"
        return token

    @staticmethod
    def _has_explicit_step_markers(query_text: str) -> bool:
        text = str(query_text or "")
        labeled = re.search(
            r"(?:^|\n)\s*(?:Step\s*\d+\)?\s*(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)\s*:|\d+\)\s*(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)\s*:)",
            text,
            flags=re.IGNORECASE,
        )
        if labeled:
            return True
        # Unlabeled Step blocks are treated as multi-step only when at least 2 blocks exist.
        return len(re.findall(r"\bStep\s*\d+\)?\b", text, flags=re.IGNORECASE)) >= 2

    def execute_rdb_query(self, query: str) -> Dict[str, Any]:
        query = self._normalize_query_text(query)
        query = self._strip_step_prefix(query)
        query, truncated = self._truncate_after_following_step_marker(query)
        started = time.time()
        try:
            result = self.rdb.run(query)
            payload = {
                "status": "success",
                "db_type": "RDB",
                "query": query,
                "result": result,
                "execution_time_sec": round(time.time() - started, 6),
            }
            if truncated:
                payload["auto_fix_applied"] = "single_path_trimmed_following_step_marker"
            return payload
        except Exception as exc:
            payload = {
                "status": "failed",
                "db_type": "RDB",
                "query": query,
                "error": str(exc),
                "execution_time_sec": round(time.time() - started, 6),
            }
            if truncated:
                payload["auto_fix_applied"] = "single_path_trimmed_following_step_marker"
            return payload

    @staticmethod
    def _quote_numeric_patient_id_literals(query: str) -> str:
        """Patch common Cypher patterns where patientId numeric literals need quotes."""
        patched = query

        # Pattern A: map literal -> {patientId: 123}
        patched = re.sub(
            r"(\bpatientId\s*:\s*)(-?\d+)\b",
            r"\1'\2'",
            patched,
            flags=re.IGNORECASE,
        )

        # Pattern B: comparison -> p.patientId = 123
        patched = re.sub(
            r"(\.\s*patientId\s*=\s*)(-?\d+)\b",
            r"\1'\2'",
            patched,
            flags=re.IGNORECASE,
        )

        # Pattern C: IN list -> p.patientId IN [1,2,3]
        def _replace_in_list(match: re.Match) -> str:
            prefix = match.group(1)
            raw_items = match.group(2)
            items = [x.strip() for x in raw_items.split(",") if x.strip()]
            normalized = []
            for item in items:
                if re.fullmatch(r"-?\d+", item):
                    normalized.append(f"'{item}'")
                else:
                    normalized.append(item)
            return f"{prefix}[{', '.join(normalized)}]"

        patched = re.sub(
            r"(?i)(\.\s*patientId\s*IN\s*)\[\s*([^\]]*?)\s*\]",
            _replace_in_list,
            patched,
        )
        return patched

    @staticmethod
    def _patch_apoc_date_parse_string_cast(query: str) -> str:
        """
        Patch APOC date parse calls that frequently fail with Long->String coercion.
        Example: apoc.date.parse(p.birthDate, 'ms', 'yyyy-MM-dd')
              -> apoc.date.parse(toString(p.birthDate), 'ms', 'yyyy-MM-dd')
        """

        def _replace(match: re.Match) -> str:
            arg = match.group(1).strip()
            if re.match(r"(?i)^toString\s*\(", arg):
                return match.group(0)
            return f"apoc.date.parse(toString({arg}),"

        return re.sub(
            r"(?i)apoc\.date\.parse\(\s*([^,]+?)\s*,",
            _replace,
            query,
        )

    @staticmethod
    def _try_run_cypher(neo4j_driver: Any, query: str) -> List[Dict[str, Any]]:
        with neo4j_driver.session() as session:
            return session.run(query).data()

    def _build_gdb_retry_queries(self, query: str, error_text: str) -> List[Dict[str, str]]:
        """Create ordered retry candidates for known Cypher runtime failures."""
        retries: List[Dict[str, str]] = []

        if "Can't coerce `Long" in error_text and "to String" in error_text:
            q1 = self._quote_numeric_patient_id_literals(query)
            if q1 != query:
                retries.append({"fix_name": "quote_numeric_patientId", "query": q1})

            q2 = self._patch_apoc_date_parse_string_cast(query)
            if q2 != query:
                retries.append({"fix_name": "apoc_date_parse_tostring", "query": q2})

            q3 = self._patch_apoc_date_parse_string_cast(self._quote_numeric_patient_id_literals(query))
            if q3 not in (query, q1, q2):
                retries.append({"fix_name": "quote_patientId_and_apoc_cast", "query": q3})

        return retries

    def execute_gdb_query(self, query: str) -> Dict[str, Any]:
        query = self._normalize_query_text(query)
        query = self._strip_step_prefix(query)
        query, truncated = self._truncate_after_following_step_marker(query)
        started = time.time()
        attempt_queue: List[Dict[str, str]] = [{"fix_name": "none", "query": query}]
        attempted_queries: Set[str] = {query}
        attempted_fixes: List[str] = []
        original_error: Optional[str] = None
        last_error: str = ""

        idx = 0
        while idx < len(attempt_queue):
            attempt = attempt_queue[idx]
            idx += 1

            fix_name = attempt["fix_name"]
            attempt_query = attempt["query"]

            try:
                result = self._try_run_cypher(self.neo4j_driver, attempt_query)
                payload: Dict[str, Any] = {
                    "status": "success",
                    "db_type": "GDB",
                    "query": attempt_query,
                    "result": result,
                    "execution_time_sec": round(time.time() - started, 6),
                }
                if fix_name != "none":
                    payload["auto_fix_applied"] = fix_name
                elif truncated:
                    payload["auto_fix_applied"] = "single_path_trimmed_following_step_marker"
                if attempted_fixes:
                    payload["attempted_fixes"] = attempted_fixes
                if original_error:
                    payload["original_error"] = original_error
                return payload
            except Exception as exc:
                error_text = str(exc)
                last_error = error_text
                if original_error is None:
                    original_error = error_text

                if fix_name != "none":
                    attempted_fixes.append(fix_name)

                for retry in self._build_gdb_retry_queries(attempt_query, error_text):
                    retry_query = retry["query"]
                    if retry_query in attempted_queries:
                        continue
                    attempted_queries.add(retry_query)
                    attempt_queue.append(retry)

        return {
            "status": "failed",
            "db_type": "GDB",
            "query": query,
            "error": last_error or "Unknown GDB execution error",
            "execution_time_sec": round(time.time() - started, 6),
            "attempted_fixes": attempted_fixes,
            "original_error": original_error,
            "auto_fix_applied": "single_path_trimmed_following_step_marker" if truncated else "",
        }

    @staticmethod
    def _normalize_query_text(query_text: str) -> str:
        text = str(query_text or "")
        # Some planner outputs contain escaped newlines ("\\n") instead of real newlines.
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\n", "\n")
        return text.replace("\r\n", "\n").strip()

    @staticmethod
    def _strip_step_prefix(query_text: str) -> str:
        q = str(query_text or "").strip()
        q = re.sub(r"^\s*```(?:sql|cypher)?\s*", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s*```\s*$", "", q, flags=re.IGNORECASE)
        q = re.sub(
            r"^\s*Step\s*\d+\)?\s*(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)\s*:\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(
            r"^\s*\d+\)\s*(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)\s*:\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(
            r"^\s*(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)\s*:\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(r"^\s*Step\s*\d+\)?\s*", "", q, flags=re.IGNORECASE)
        return q.strip().rstrip(";")

    @classmethod
    def _parse_multi_step(cls, query_text: str) -> List[Dict[str, Any]]:
        text = cls._normalize_query_text(query_text)
        if not text:
            return []

        label_tokens = r"(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)"
        numbered_patterns = [
            re.compile(
                rf"(?:^|\n|;)\s*Step\s*(\d+)\)?\s*({label_tokens})\s*:\s*(.*?)(?=(?:\n|;|\s+)(?:Step\s*\d+\)?\s*{label_tokens}\s*:)|\Z)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                rf"(?:^|\n|;)\s*(\d+)\)\s*({label_tokens})\s*:\s*(.*?)(?=(?:\n|;|\s+)(?:\d+\)\s*{label_tokens}\s*:)|\Z)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
        ]
        for pattern in numbered_patterns:
            steps: List[Dict[str, Any]] = []
            for match in pattern.finditer(text):
                normalized_db_type = cls._normalize_step_db_type(match.group(2))
                step_query = cls._strip_step_prefix(match.group(3))
                if not step_query:
                    continue
                steps.append(
                    {
                        "step_number": int(match.group(1)),
                        "db_type": normalized_db_type,
                        "query": step_query,
                    }
                )
            if steps:
                return sorted(steps, key=lambda s: s["step_number"])

        # Fallback 1: unlabeled but numbered blocks ("Step 1) ... Step 2) ...")
        unlabeled_pattern = re.compile(
            r"(?:^|[\n;])\s*Step\s*(\d+)\)?\s*(.*?)(?=(?:[\n;]\s*Step\s*\d+\)?|\s+Step\s*\d+\)?)|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        unlabeled_steps: List[Dict[str, Any]] = []
        for match in unlabeled_pattern.finditer(text):
            raw_query = cls._strip_step_prefix(match.group(2))
            if not raw_query:
                continue
            inferred = cls._infer_single_step_db_type(raw_query)
            unlabeled_steps.append(
                {
                    "step_number": int(match.group(1)),
                    "db_type": inferred or "UNKNOWN",
                    "query": raw_query,
                }
            )
        if len(unlabeled_steps) >= 2:
            return sorted(unlabeled_steps, key=lambda s: s["step_number"])

        # Fallback 2: label-only blocks ("RDB: ... GDB: ...")
        label_only_pattern = re.compile(
            rf"(?:^|\n)\s*({label_tokens})\s*:\s*(.*?)(?=(?:\n\s*{label_tokens}\s*:)|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        label_only_steps: List[Dict[str, Any]] = []
        for idx, match in enumerate(label_only_pattern.finditer(text), start=1):
            step_query = cls._strip_step_prefix(match.group(2))
            if not step_query:
                continue
            label_only_steps.append(
                {
                    "step_number": idx,
                    "db_type": cls._normalize_step_db_type(match.group(1)),
                    "query": step_query,
                }
            )
        if len(label_only_steps) >= 2:
            return label_only_steps

        return []

    @staticmethod
    def _truncate_after_following_step_marker(query_text: str) -> Tuple[str, bool]:
        text = str(query_text or "").strip()
        if not text:
            return text, False

        marker_pattern = re.compile(
            r"(?:Step\s*\d+\)?\s*(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)?\s*:|\d+\)\s*(?:RDB|GDB|SQL|MYSQL|CYPHER|GRAPH|NEO4J)\s*:)",
            flags=re.IGNORECASE,
        )

        for match in marker_pattern.finditer(text):
            if match.start() == 0:
                continue
            prefix = text[: match.start()].strip().rstrip(";")
            if not prefix:
                continue
            return prefix, True

        return text, False

    @staticmethod
    def _infer_single_step_db_type(query: str) -> Optional[str]:
        """Infer single-step DB type from query text when planner output is inconsistent."""
        q = (query or "").strip()
        if not q:
            return None

        has_sql = bool(re.search(r"\bSELECT\b|\bFROM\b|\bJOIN\b", q, flags=re.IGNORECASE))
        has_cypher = bool(re.search(r"\bMATCH\b|\bOPTIONAL\s+MATCH\b|-\s*\[:", q, flags=re.IGNORECASE))

        if has_cypher and not has_sql:
            return "GDB"
        if has_sql and not has_cypher:
            return "RDB"
        if has_cypher and has_sql:
            # If mixed signals exist, prefer query-start token.
            if re.match(r"^\s*MATCH\b", q, flags=re.IGNORECASE):
                return "GDB"
            if re.match(r"^\s*SELECT\b", q, flags=re.IGNORECASE):
                return "RDB"
        return None

    def _execute_multi_steps(self, query: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        step_results: List[Dict[str, Any]] = []
        for step in steps:
            step_number = int(step.get("step_number", len(step_results) + 1))
            step_db_type = str(step.get("db_type", "")).upper()
            step_query_raw = str(step.get("query", ""))
            step_query = self._strip_step_prefix(step_query_raw)

            if not step_query:
                result = {
                    "status": "failed",
                    "db_type": step_db_type or "UNKNOWN",
                    "query": step_query_raw,
                    "error": "Empty query in parsed multi-step segment.",
                }
            elif step_db_type == "RDB":
                result = self.execute_rdb_query(step_query)
            elif step_db_type == "GDB":
                result = self.execute_gdb_query(step_query)
            else:
                inferred = self._infer_single_step_db_type(step_query)
                if inferred == "RDB":
                    result = self.execute_rdb_query(step_query)
                    result["auto_fix_applied"] = "multi_step_step_db_inferred_rdb"
                elif inferred == "GDB":
                    result = self.execute_gdb_query(step_query)
                    result["auto_fix_applied"] = "multi_step_step_db_inferred_gdb"
                else:
                    result = {
                        "status": "failed",
                        "db_type": step_db_type or "UNKNOWN",
                        "query": step_query,
                        "error": f"Unsupported multi-step step db_type: {step_db_type}",
                    }

            step_results.append({"step_number": step_number, **result})

        failed_step = next((item for item in step_results if item.get("status") != "success"), None)
        payload: Dict[str, Any] = {
            "status": "failed" if failed_step else "success",
            "execution_mode": "multi_step",
            "query": query,
            "step_results": step_results,
        }
        if failed_step:
            payload["error"] = failed_step.get("error", "Unknown multi-step execution error")
            payload["failed_step_number"] = failed_step.get("step_number")
            payload["failed_step_db_type"] = failed_step.get("db_type")
            payload["failed_step_query"] = failed_step.get("query")
        return payload

    def execute_plan(self, selected_plan: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query_text(str(selected_plan.get("query", "")))
        db_type = str(selected_plan.get("db_type", "RDB")).upper()
        is_multi_step = bool(selected_plan.get("is_multi_step", False))
        has_step_prefix = self._has_explicit_step_markers(query)

        if not query:
            return {
                "status": "failed",
                "execution_mode": "single",
                "db_type": db_type,
                "query": query,
                "error": "Empty query in selected plan.",
            }

        if db_type in ("RDB", "GDB") and not is_multi_step:
            single_query = self._strip_step_prefix(query)
            single_query, trimmed = self._truncate_after_following_step_marker(single_query)
            if trimmed:
                query = single_query
                has_step_prefix = False

        if is_multi_step or has_step_prefix:
            steps = self._parse_multi_step(query)
            if not steps:
                inferred = db_type if db_type in ("RDB", "GDB") else self._infer_single_step_db_type(query)
                if inferred == "RDB":
                    result = self.execute_rdb_query(self._strip_step_prefix(query))
                    result["execution_mode"] = "single"
                    result["auto_fix_applied"] = "multi_step_flag_without_steps_inferred_rdb"
                    return result
                if inferred == "GDB":
                    result = self.execute_gdb_query(self._strip_step_prefix(query))
                    result["execution_mode"] = "single"
                    result["auto_fix_applied"] = "multi_step_flag_without_steps_inferred_gdb"
                    return result
                return {
                    "status": "failed",
                    "execution_mode": "single",
                    "db_type": db_type,
                    "query": query,
                    "error": "Plan marked as multi-step, but no valid steps were found.",
                }
            return self._execute_multi_steps(query, steps)

        if db_type == "RDB":
            return self.execute_rdb_query(self._strip_step_prefix(query))
        if db_type == "GDB":
            return self.execute_gdb_query(self._strip_step_prefix(query))
        if db_type == "BOTH":
            steps = self._parse_multi_step(query)
            if not steps:
                inferred = self._infer_single_step_db_type(query)
                if inferred == "RDB":
                    result = self.execute_rdb_query(self._strip_step_prefix(query))
                    result["execution_mode"] = "single"
                    result["auto_fix_applied"] = "both_single_step_inferred_rdb"
                    return result
                if inferred == "GDB":
                    result = self.execute_gdb_query(self._strip_step_prefix(query))
                    result["execution_mode"] = "single"
                    result["auto_fix_applied"] = "both_single_step_inferred_gdb"
                    return result
                return {
                    "status": "failed",
                    "execution_mode": "single",
                    "db_type": "BOTH",
                    "query": query,
                    "error": "db_type=BOTH without valid multi-step format and no single-step DB could be inferred.",
                }
            return self._execute_multi_steps(query, steps)

        return {
            "status": "failed",
            "execution_mode": "single",
            "db_type": db_type,
            "query": query,
            "error": f"Unsupported db_type: {db_type}",
        }


class BaseHybridAgent:
    """Main agent: MetaKG -> Schema injection -> Plan -> CAQA -> Execute."""

    def __init__(
        self,
        rdb: Any,
        neo4j_driver: Any,
        llm: OpenAILLM,
        complexity_constants: Optional[ComplexityConstants] = None,
        risk_lambda: float = 1.0,
        normalize_expected_cost: bool = True,
        enable_recovery: bool = True,
        recovery_policy: str = "uncertainty",  # uncertainty | on_failure | always | never
        recovery_threshold: float = 0.5,
        max_recovery_attempts: int = 3,
        debug: bool = True,
    ) -> None:
        self.rdb = rdb
        self.neo4j_driver = neo4j_driver
        self.llm = llm
        self.risk_lambda = float(risk_lambda)
        self.normalize_expected_cost = bool(normalize_expected_cost)
        self.enable_recovery = bool(enable_recovery)
        self.recovery_policy = recovery_policy
        self.recovery_threshold = float(recovery_threshold)
        self.max_recovery_attempts = max(1, int(max_recovery_attempts))
        self.debug = bool(debug)

        self.meta_kg_generator = MetaKnowledgeGraphGenerator(rdb, neo4j_driver, llm)
        self.query_planner = QueryPlanner(llm, complexity_constants)
        self.caqa_evaluator = CAQAEvaluator(rdb, neo4j_driver)
        self.query_executor = QueryExecutor(rdb, neo4j_driver)

    def refresh_meta_kg(self) -> Dict[str, Any]:
        """Force-refresh cached metaKG and return the new snapshot."""
        return self.meta_kg_generator.generate_meta_kg(force_refresh=True)

    def _debug_print(self, message: str) -> None:
        if self.debug:
            print(f"[debug] {message}")

    @staticmethod
    def _shorten(text: Any, max_len: int = 300) -> str:
        raw = str(text)
        if len(raw) <= max_len:
            return raw
        return raw[: max_len - 3] + "..."

    @staticmethod
    def _extract_execution_error(execution_result: Dict[str, Any]) -> str:
        if not isinstance(execution_result, dict):
            return ""
        direct_error = execution_result.get("error")
        if direct_error:
            return str(direct_error)
        for step in execution_result.get("step_results", []):
            if isinstance(step, dict) and step.get("status") != "success":
                query = str(step.get("query", ""))
                error = str(step.get("error", "unknown step error"))
                return f"step_query={query} | error={error}"
        return ""

    @staticmethod
    def _safe_llm_snapshot(llm: Any) -> Optional[Dict[str, float]]:
        snapshot_fn = getattr(llm, "snapshot", None)
        if not callable(snapshot_fn):
            return None
        try:
            raw = snapshot_fn()
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        return {str(k): float(v or 0.0) for k, v in raw.items()}

    @staticmethod
    def _safe_llm_delta(
        llm: Any,
        after: Optional[Dict[str, float]],
        before: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        zero = {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "llm_calls": 0.0,
            "llm_latency_sec": 0.0,
        }
        if after is None or before is None:
            return zero
        delta_fn = getattr(llm, "delta", None)
        if callable(delta_fn):
            try:
                raw = delta_fn(after, before)
                if isinstance(raw, dict):
                    out = zero.copy()
                    out.update({str(k): float(v or 0.0) for k, v in raw.items()})
                    return out
            except Exception:
                pass
        out = zero.copy()
        for key in out.keys():
            out[key] = float(after.get(key, 0.0) - before.get(key, 0.0))
        return out

    @staticmethod
    def _build_error_guided_repair_hints(error_log: str) -> str:
        err = str(error_log or "")
        lower = err.lower()
        hints: List[str] = []

        if "plan marked as multi-step, but no valid steps were found" in lower:
            hints.append(
                "Multi-step parse failure: if db_type=BOTH, use exact headers "
                "'Step 1) RDB:' and 'Step 2) GDB:' on separate lines."
            )
            hints.append(
                "If you cannot produce valid BOTH step headers, output a single-source plan (RDB or GDB) with one executable statement."
            )

        if "expected exactly one statement per query but got" in lower:
            hints.append("Single-statement failure: return exactly one statement per step/query. Do not concatenate statements with ';'.")

        if "you have an error in your sql syntax" in lower:
            hints.append("MySQL syntax failure: avoid invalid CTE forms. Use 'WITH cte AS (SELECT ... ) SELECT ...'.")
            hints.append("Do not use 'WITH (SELECT ... ) AS alias' or 'WITH Table AS alias SELECT ...'.")

        if "near '(selec" in lower or "near 'diag" in lower:
            hints.append("Current SQL appears malformed around WITH/SELECT. Rewrite from scratch with explicit FROM/JOIN and valid alias declarations.")

        if "variable `" in err and "not defined" in lower:
            var_match = re.search(r"Variable\s+`([^`]+)`\s+not\s+defined", err, flags=re.IGNORECASE)
            var_name = var_match.group(1) if var_match else "a variable"
            hints.append(
                f"Cypher scope failure: `{var_name}` is referenced before binding. "
                "Any variable used in WHERE/RETURN/ORDER BY must be bound in current MATCH/WITH scope."
            )
            hints.append("When using WITH, carry forward every variable needed later (e.g., WITH p, a, d).")

        if "unknown column" in lower:
            hints.append("Unknown column failure: use only table.column names that explicitly exist in SCHEMA_CONTEXT for that table.")
            if "rowid" in lower:
                hints.append("MIMIC RDB failure: medi.rowId/m.rowId is not valid. Use COUNT(*), COUNT(DISTINCT medication), or a schema-validated composite key from existing columns.")

        if "subquery returns more than 1 row" in lower:
            hints.append(
                "MySQL scalar subquery failure: a SELECT-list subquery must return exactly one row. "
                "Do not place top-3 GROUP BY ... LIMIT 3 subqueries as scalar columns."
            )
            hints.append("Rewrite top-k outputs with a CTE plus GROUP_CONCAT/JSON_ARRAYAGG, or return the top-k rows directly.")

        if "cannot select datetime from" in lower:
            hints.append(
                "Cypher temporal failure: the value is date-only, so do not call datetime(...) on it. "
                "Use date(a.admitDate)/date(a.dischargeDate), duration.inDays(...).days, or prefer RDB admin.los for length-of-stay."
            )

        if not hints:
            hints.append("Repair by simplifying query structure while preserving intent, and strictly validate schema names from SCHEMA_CONTEXT.")

        return "\n".join(f"- {h}" for h in hints)

    @staticmethod
    def _extract_cypher_node_labels(query: str) -> Set[str]:
        return set(
            re.findall(
                r"\(\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*)?:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
                query or "",
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _extract_cypher_relationship_types(query: str) -> Set[str]:
        return set(
            re.findall(
                r"\[\s*[^]]*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
                query or "",
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _extract_cypher_alias_to_labels(query: str) -> Dict[str, Set[str]]:
        alias_map: Dict[str, Set[str]] = {}
        pattern = re.compile(
            r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
            flags=re.IGNORECASE,
        )
        for alias, label in pattern.findall(query or ""):
            alias_map.setdefault(alias, set()).add(label)
        return alias_map

    @staticmethod
    def _extract_cypher_alias_properties(query: str) -> List[Dict[str, str]]:
        refs: List[Dict[str, str]] = []
        seen: Set[str] = set()
        pattern = re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
            flags=re.IGNORECASE,
        )
        for alias, prop in pattern.findall(query or ""):
            key = f"{alias}.{prop}".lower()
            if key in seen:
                continue
            seen.add(key)
            refs.append({"alias": alias, "property": prop})
        return refs

    @staticmethod
    def _extract_cypher_node_map_properties(query: str) -> List[Dict[str, str]]:
        refs: List[Dict[str, str]] = []
        seen: Set[str] = set()
        node_map_pattern = re.compile(
            r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\{([^}]*)\}",
            flags=re.IGNORECASE | re.DOTALL,
        )
        map_key_pattern = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*:")
        for _, label, map_text in node_map_pattern.findall(query or ""):
            for prop in map_key_pattern.findall(map_text or ""):
                key = f"{label}.{prop}".lower()
                if key in seen:
                    continue
                seen.add(key)
                refs.append({"label": label, "property": prop})
        return refs

    @staticmethod
    def _extract_rdb_columns_from_schema_text(schema_text: Any) -> Set[str]:
        text = str(schema_text or "")
        if not text.strip():
            return set()

        columns: Set[str] = set()
        skip_prefixes = (
            "CREATE ",
            "PRIMARY ",
            "KEY ",
            "UNIQUE ",
            "CONSTRAINT ",
            "FOREIGN ",
            "INDEX ",
            ")",
        )

        for raw_line in text.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            upper = line.upper()
            if upper.startswith(skip_prefixes):
                continue
            match = re.match(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s+[A-Za-z]", line)
            if match:
                columns.add(match.group(1))
        return columns

    def _build_rdb_table_columns(self, meta_kg: Dict[str, Any]) -> Dict[str, Set[str]]:
        rdb = meta_kg.get("rdb", {}) if isinstance(meta_kg, dict) else {}
        table_columns: Dict[str, Set[str]] = {}
        for table_name, table_info in rdb.items():
            columns: Set[str] = set()
            if isinstance(table_info, dict):
                columns |= self._extract_rdb_columns_from_schema_text(table_info.get("schema"))
                sample_data = table_info.get("sample_data")
                if isinstance(sample_data, list):
                    for row in sample_data:
                        if isinstance(row, dict):
                            columns |= set(str(k) for k in row.keys())
            table_columns[str(table_name)] = columns
        return table_columns

    @staticmethod
    def _extract_sql_cte_names(query: str) -> Set[str]:
        text = str(query or "")
        if not re.match(r"^\s*WITH\b", text, flags=re.IGNORECASE):
            return set()
        return set(
            name
            for name in re.findall(
                r"(?:^\s*WITH|,)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?\s+AS\s*\(",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )

    @staticmethod
    def _extract_rdb_table_aliases(query: str) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        if not query:
            return alias_map

        pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+[`\"]?([A-Za-z0-9_]+)[`\"]?(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
            flags=re.IGNORECASE,
        )
        banned_alias = {
            "ON",
            "USING",
            "WHERE",
            "GROUP",
            "ORDER",
            "LIMIT",
            "LEFT",
            "RIGHT",
            "FULL",
            "INNER",
            "OUTER",
            "JOIN",
            "UNION",
        }

        for match in pattern.finditer(query):
            table = str(match.group(1))
            alias = match.group(2)
            alias_clean = str(alias).strip() if alias else table
            if alias_clean.upper() in banned_alias:
                alias_clean = table
            alias_map[alias_clean] = table
            alias_map[table] = table
        return alias_map

    @staticmethod
    def _parse_plan_steps_for_schema_guard(plan: Dict[str, Any]) -> List[Dict[str, str]]:
        db_type = str(plan.get("db_type", "RDB")).upper()
        query = str(plan.get("query", ""))
        if "\\n" in query and "\n" not in query:
            query = query.replace("\\n", "\n")

        if db_type != "BOTH":
            return [{"db_type": db_type, "query": query}]

        parsed_steps = QueryExecutor._parse_multi_step(query)
        if parsed_steps:
            return [
                {
                    "step_number": str(step.get("step_number", idx)),
                    "db_type": str(step.get("db_type", "UNKNOWN")),
                    "query": str(step.get("query", "")).strip().rstrip(";"),
                }
                for idx, step in enumerate(parsed_steps, start=1)
            ]

        has_sql = bool(re.search(r"\bSELECT\b|\bFROM\b|\bJOIN\b", query, flags=re.IGNORECASE))
        has_cypher = bool(re.search(r"\bMATCH\b|\bOPTIONAL\s+MATCH\b|-\s*\[:", query, flags=re.IGNORECASE))
        if has_sql and not has_cypher:
            return [{"db_type": "RDB", "query": query}]
        if has_cypher and not has_sql:
            return [{"db_type": "GDB", "query": query}]
        return [{"db_type": "BOTH", "query": query}]

    def _validate_rdb_schema_refs(self, query: str, meta_kg: Dict[str, Any]) -> Dict[str, Any]:
        rdb = meta_kg.get("rdb", {}) if isinstance(meta_kg, dict) else {}
        known_tables = set(str(t) for t in rdb.keys())
        table_columns = self._build_rdb_table_columns(meta_kg)

        alias_map = self._extract_rdb_table_aliases(query)
        cte_names = self._extract_sql_cte_names(query)
        cte_names_lc = {name.lower() for name in cte_names}
        referenced_tables = sorted(set(alias_map.values()))
        unknown_tables = sorted(
            t for t in referenced_tables
            if t not in known_tables and t.lower() not in cte_names_lc
        )

        unknown_columns: Set[str] = set()
        qualified_cols = re.findall(
            r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
            query or "",
            flags=re.IGNORECASE,
        )
        for alias, column in qualified_cols:
            table = alias_map.get(alias)
            if not table:
                continue
            if table not in known_tables:
                continue
            known_cols = table_columns.get(table, set())
            if not known_cols:
                continue
            if column not in known_cols:
                unknown_columns.add(f"{alias}.{column}")

        return {
            "referenced_tables": referenced_tables,
            "unknown_tables": unknown_tables,
            "unknown_columns": sorted(unknown_columns),
            "alias_map": alias_map,
            "cte_names": sorted(cte_names),
            "is_valid": not unknown_tables and not unknown_columns,
        }


    def _validate_gdb_timeline_topology(self, query: str) -> Dict[str, Any]:
        """Validate MIMIC-specific graph topology for event queries."""
        used_labels = self._extract_cypher_node_labels(query)
        used_relationships = self._extract_cypher_relationship_types(query)

        normalized_rels = {str(r).upper() for r in used_relationships}
        admission_event_rels = {"HAS_DIAG"}
        day_event_rels = {"HAS_LAB", "HAS_MEDI", "HAS_VITAL", "HAS_OPERATION", "HAS_PHYSICAL"}
        event_rels = admission_event_rels | day_event_rels
        has_event_rel = bool(normalized_rels & event_rels)
        has_admission_event_rel = bool(normalized_rels & admission_event_rels)
        has_day_event_rel = bool(normalized_rels & day_event_rels)

        has_day_label = any(str(lbl).lower() == "day" for lbl in used_labels) or bool(
            re.search(r":\s*`?Day`?", query or "", flags=re.IGNORECASE)
        )
        has_admission_label = any(str(lbl).lower() == "admission" for lbl in used_labels) or bool(
            re.search(r":\s*`?Admission`?", query or "", flags=re.IGNORECASE)
        )
        has_patient_label = any(str(lbl).lower() == "patient" for lbl in used_labels) or bool(
            re.search(r":\s*`?Patient`?", query or "", flags=re.IGNORECASE)
        )
        has_next_rel = "NEXT" in normalized_rels or bool(
            re.search(r"\[\s*[^]]*:\s*`?NEXT`?", query or "", flags=re.IGNORECASE)
        )
        has_has_day_rel = "HAS_DAY" in normalized_rels or bool(
            re.search(r"\[\s*[^]]*:\s*`?HAS_DAY`?", query or "", flags=re.IGNORECASE)
        )
        has_admission_rel = "HAS_ADMISSION" in normalized_rels or bool(
            re.search(r"\[\s*[^]]*:\s*`?HAS_ADMISSION`?", query or "", flags=re.IGNORECASE)
        )

        direct_patient_event_edge = bool(
            re.search(
                r"\([^)]*:\s*`?Patient`?[^)]*\)\s*-\s*\[[^]]*:\s*`?HAS_(?:DIAGNOSIS|LAB_TEST|MEDICATION|VITAL|OPERATION|PHYSICAL)`?[^]]*\]\s*->",
                query or "",
                flags=re.IGNORECASE,
            )
        )
        patient_to_admission_edge = bool(
            re.search(
                r"\([^)]*:\s*`?Patient`?[^)]*\)\s*-\s*\[[^]]*:\s*`?HAS_ADMISSION`?[^]]*\]\s*->\s*\([^)]*:\s*`?Admission`?",
                query or "",
                flags=re.IGNORECASE,
            )
        )
        admission_to_diagnosis_edge = bool(
            re.search(
                r"\([^)]*:\s*`?Admission`?[^)]*\)\s*-\s*\[[^]]*:\s*`?HAS_DIAG`?[^]]*\]\s*->\s*\([^)]*:\s*`?Diagnosis`?",
                query or "",
                flags=re.IGNORECASE,
            )
        )
        day_to_event_edge = bool(
            re.search(
                r"\([^)]*:\s*`?Day`?[^)]*\)\s*-\s*\[[^]]*:\s*`?HAS_(?:LAB_TEST|MEDICATION|VITAL|OPERATION|PHYSICAL)`?[^]]*\]\s*->",
                query or "",
                flags=re.IGNORECASE,
            )
        )

        violations: List[str] = []
        if has_day_event_rel and has_has_day_rel:
            violations.append("uses_has_day_for_day_event_query")
        if has_event_rel:
            if not has_patient_label:
                violations.append("event_query_missing_patient_label")
            if not has_admission_label:
                violations.append("event_query_missing_admission_label")
            if not has_admission_rel:
                violations.append("event_query_missing_has_admission_relationship")
            if not patient_to_admission_edge:
                violations.append("event_query_missing_patient_to_admission_path")
            if direct_patient_event_edge:
                violations.append("direct_patient_to_event_edge_not_allowed")

        if has_day_event_rel:
            if not has_day_label:
                violations.append("day_event_query_missing_day_label")
            if not has_next_rel:
                violations.append("day_event_query_missing_next_relationship")
            if not day_to_event_edge:
                relaxed_day_path = bool(has_day_label and has_next_rel and has_day_event_rel)
                if not relaxed_day_path:
                    violations.append("day_event_query_missing_day_to_event_path")

        if has_admission_event_rel and not admission_to_diagnosis_edge:
            relaxed_diagnosis_path = bool(has_admission_label and has_admission_event_rel)
            if not relaxed_diagnosis_path:
                violations.append("diagnosis_query_missing_admission_to_diagnosis_path")

        return {
            "requires_day_timeline": bool(has_day_event_rel),
            "has_day_label": bool(has_day_label),
            "has_admission_label": bool(has_admission_label),
            "has_patient_label": bool(has_patient_label),
            "has_has_admission_relationship": bool(has_admission_rel),
            "has_next_relationship": bool(has_next_rel),
            "has_has_day_relationship": bool(has_has_day_rel),
            "has_patient_to_admission_edge": bool(patient_to_admission_edge),
            "has_day_to_event_edge": bool(day_to_event_edge),
            "has_admission_to_diagnosis_edge": bool(admission_to_diagnosis_edge),
            "direct_patient_event_edge": bool(direct_patient_event_edge),
            "topology_violations": violations,
            "is_valid": not violations,
        }

    def _validate_plan_schema_refs(self, plan: Dict[str, Any], meta_kg: Dict[str, Any]) -> Dict[str, Any]:
        step_reports: List[Dict[str, Any]] = []
        unknown_labels: Set[str] = set()
        unknown_relationships: Set[str] = set()
        unknown_node_properties: Set[str] = set()
        unknown_tables: Set[str] = set()
        unknown_columns: Set[str] = set()
        topology_violations: Set[str] = set()

        for idx, step in enumerate(self._parse_plan_steps_for_schema_guard(plan), start=1):
            step_db_type = str(step.get("db_type", "RDB")).upper()
            step_query = str(step.get("query", ""))
            entry: Dict[str, Any] = {
                "step_number": idx,
                "db_type": step_db_type,
            }

            if step_db_type in ("GDB", "BOTH"):
                gdb_report = self._validate_gdb_schema_refs(step_query, meta_kg)
                entry["gdb_report"] = gdb_report
                unknown_labels.update(gdb_report.get("unknown_labels", []))
                unknown_relationships.update(gdb_report.get("unknown_relationships", []))
                unknown_node_properties.update(gdb_report.get("unknown_node_properties", []))
                timeline_report = self._validate_gdb_timeline_topology(step_query)
                entry["timeline_report"] = timeline_report
                topology_violations.update(timeline_report.get("topology_violations", []))

            if step_db_type in ("RDB", "BOTH"):
                rdb_report = self._validate_rdb_schema_refs(step_query, meta_kg)
                entry["rdb_report"] = rdb_report
                unknown_tables.update(rdb_report.get("unknown_tables", []))
                unknown_columns.update(rdb_report.get("unknown_columns", []))

            step_reports.append(entry)

        report = {
            "step_reports": step_reports,
            "unknown_labels": sorted(unknown_labels),
            "unknown_relationships": sorted(unknown_relationships),
            "unknown_node_properties": sorted(unknown_node_properties),
            "unknown_tables": sorted(unknown_tables),
            "unknown_columns": sorted(unknown_columns),
            "topology_violations": sorted(topology_violations),
        }
        report["is_valid"] = not (
            report["unknown_labels"]
            or report["unknown_relationships"]
            or report["unknown_node_properties"]
            or report["unknown_tables"]
            or report["unknown_columns"]
            or report["topology_violations"]
        )
        return report

    @staticmethod
    def _contains_unresolved_runtime_placeholder(query: str) -> bool:
        text = str(query or "")
        patterns = (
            r"\?[A-Za-z_][A-Za-z0-9_]*",  # SQL-style named placeholder
            r"\$[A-Za-z_][A-Za-z0-9_]*",  # Cypher parameter placeholder
            r"<\s*[A-Za-z_][A-Za-z0-9_]*\s*>",  # template token like <pid>
            r"\*\s*\d+\s*\.\.\s*[A-Za-z_][A-Za-z0-9_]*",  # Cypher var-length path upper bound like *0..N
            r"\*\s*[A-Za-z_][A-Za-z0-9_]*\s*\.\.\s*\d+",  # Cypher var-length path lower bound like *N..10
        )
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _has_multiple_statements(query: str) -> bool:
        """
        Detect likely multi-statement text separated by semicolons.
        A trailing single semicolon is allowed.
        """
        text = str(query or "")
        chunks = [seg.strip() for seg in text.split(";") if seg.strip()]
        return len(chunks) > 1

    @staticmethod
    def _compile_rdb_query_sanity(query: str) -> List[str]:
        errors: List[str] = []
        q = str(query or "")
        if re.search(r"\bUNNEST\s*\(", q, flags=re.IGNORECASE):
            errors.append("uses_unnest_not_supported_in_mysql")
        if re.search(r"\bFROM\s*\(\s*VALUES\b", q, flags=re.IGNORECASE):
            errors.append("uses_values_table_constructor_not_supported")
        if re.search(r"\bMATCH\b|\bOPTIONAL\s+MATCH\b|-\s*\[:", q, flags=re.IGNORECASE):
            errors.append("contains_cypher_tokens")
        if re.search(r"\breplace\s+with\s+actual\b", q, flags=re.IGNORECASE):
            errors.append("contains_narrative_placeholder_text")
        # Project rule: patientId should be treated as a quoted string key in SQL.
        if re.search(
            r"\b(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?patientId\s*=\s*-?\d+\b",
            q,
            flags=re.IGNORECASE,
        ):
            errors.append("patientid_literal_must_be_quoted")
        if re.search(
            r"\b(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?patientId\s+IN\s*\(\s*-?\d+(?:\s*,\s*-?\d+)*\s*\)",
            q,
            flags=re.IGNORECASE,
        ):
            errors.append("patientid_in_literals_must_be_quoted")
        if BaseHybridAgent._has_multiple_statements(q):
            errors.append("multiple_statements_not_allowed")
        return errors

    @staticmethod
    def _compile_gdb_query_sanity(query: str) -> List[str]:
        errors: List[str] = []
        q = str(query or "")
        if re.search(r"\bSELECT\b|\bFROM\b|\bJOIN\b", q, flags=re.IGNORECASE):
            errors.append("contains_sql_tokens")
        if re.search(r"\bUNNEST\s*\(", q, flags=re.IGNORECASE):
            errors.append("contains_sql_unnest_token")
        if re.search(r"\breplace\s+with\s+actual\b", q, flags=re.IGNORECASE):
            errors.append("contains_narrative_placeholder_text")
        if re.search(r"\*\s*\d+\s*\.\.\s*[A-Za-z_][A-Za-z0-9_]*", q):
            errors.append("contains_non_numeric_path_bound")
        if re.search(r"\*\s*[A-Za-z_][A-Za-z0-9_]*\s*\.\.\s*\d+", q):
            errors.append("contains_non_numeric_path_bound")
        if BaseHybridAgent._has_multiple_statements(q):
            errors.append("multiple_statements_not_allowed")
        return errors

    def _compile_plan_for_execution(
        self,
        plan: Dict[str, Any],
        meta_kg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compile plan query text before scoring/execution:
        - normalize formatting
        - reject unresolved placeholders and non-executable narrative text
        - apply SQL/Cypher sanity checks
        - run schema/topology checks
        """
        compiled = copy.deepcopy(plan)
        compile_errors: List[str] = []
        compile_warnings: List[str] = []

        raw_query = str(compiled.get("query", "") or "")
        query = self.query_executor._normalize_query_text(raw_query)
        compiled["query"] = query

        db_type = str(compiled.get("db_type", "RDB")).upper()
        if db_type not in ("RDB", "GDB", "BOTH"):
            compile_errors.append(f"unsupported_db_type_{db_type.lower()}")

        has_step_prefix = QueryExecutor._has_explicit_step_markers(query)

        if db_type in ("RDB", "GDB"):
            compiled["is_multi_step"] = False
            if has_step_prefix:
                stripped = self.query_executor._strip_step_prefix(query)
                if stripped:
                    compiled["query"] = stripped
                    query = stripped
                    compile_warnings.append("single_step_prefix_stripped")

            if self._contains_unresolved_runtime_placeholder(query):
                compile_errors.append("contains_unresolved_runtime_placeholder")

            if db_type == "RDB":
                for err in self._compile_rdb_query_sanity(query):
                    compile_errors.append(f"rdb_{err}")
            else:
                for err in self._compile_gdb_query_sanity(query):
                    compile_errors.append(f"gdb_{err}")

        elif db_type == "BOTH":
            compiled["is_multi_step"] = True
            steps = self.query_executor._parse_multi_step(query)
            if not steps:
                compile_errors.append("both_missing_explicit_step_blocks")
            else:
                canonical_steps: List[str] = []
                for idx, step in enumerate(steps, start=1):
                    step_db_type = str(step.get("db_type", "")).upper()
                    step_query = self.query_executor._strip_step_prefix(str(step.get("query", "")))
                    if not step_query:
                        compile_errors.append(f"step{idx}_empty_query")
                        continue

                    if self._contains_unresolved_runtime_placeholder(step_query):
                        compile_errors.append(f"step{idx}_contains_unresolved_runtime_placeholder")

                    if step_db_type == "RDB":
                        for err in self._compile_rdb_query_sanity(step_query):
                            compile_errors.append(f"step{idx}_rdb_{err}")
                    elif step_db_type == "GDB":
                        for err in self._compile_gdb_query_sanity(step_query):
                            compile_errors.append(f"step{idx}_gdb_{err}")
                    else:
                        inferred = self.query_executor._infer_single_step_db_type(step_query)
                        if inferred:
                            step_db_type = inferred
                            compile_warnings.append(f"step{idx}_db_type_inferred_{inferred.lower()}")
                        else:
                            compile_errors.append(f"step{idx}_unsupported_db_type")

                    canonical_steps.append(f"Step {idx}) {step_db_type}:\n{step_query}")

                if canonical_steps:
                    compiled["query"] = "\n\n".join(canonical_steps)

        schema_report = self._validate_plan_schema_refs(compiled, meta_kg)
        if not schema_report.get("is_valid", True):
            compile_errors.append("schema_or_topology_validation_failed")

        dedup_errors = sorted(set(str(e) for e in compile_errors if str(e).strip()))
        dedup_warnings = sorted(set(str(w) for w in compile_warnings if str(w).strip()))

        compiled["schema_guard_report"] = schema_report
        compiled["schema_compiler_report"] = {
            "is_valid": bool(schema_report.get("is_valid", True)) and len(dedup_errors) == 0,
            "errors": dedup_errors,
            "warnings": dedup_warnings,
        }
        return compiled

    def _validate_gdb_schema_refs(self, query: str, meta_kg: Dict[str, Any]) -> Dict[str, Any]:
        gdb = meta_kg.get("gdb", {}) if isinstance(meta_kg, dict) else {}
        known_labels = set(gdb.get("node_labels", []))
        known_relationships = set(gdb.get("relationship_types", []))
        # The generated MetaKG can omit low-frequency MIMIC graph elements even
        # when they are present in Neo4j. Keep schema_guard aligned with the
        # validated MIMIC topology used by the benchmark notebooks.
        known_labels.update({"Patient", "Admission", "Day", "Diagnosis", "Lab", "Medi", "Operation", "Physical", "Vital"})
        known_relationships.update({
            "HAS_ADMISSION",
            "HAS_DIAG",
            "HAS_DAY",
            "NEXT",
            "HAS_LAB",
            "HAS_MEDI",
            "HAS_OPERATION",
            "HAS_PHYSICAL",
            "HAS_VITAL",
        })
        known_props_by_label: Dict[str, Set[str]] = {
            str(label): set(props or [])
            for label, props in (gdb.get("node_properties", {}) or {}).items()
            if isinstance(props, list)
        }

        used_labels = self._extract_cypher_node_labels(query)
        used_relationships = self._extract_cypher_relationship_types(query)
        alias_to_labels = self._extract_cypher_alias_to_labels(query)
        alias_property_refs = self._extract_cypher_alias_properties(query)
        node_map_property_refs = self._extract_cypher_node_map_properties(query)

        unknown_labels = sorted(label for label in used_labels if label not in known_labels)
        unknown_relationships = sorted(rel for rel in used_relationships if rel not in known_relationships)
        unknown_node_properties: Set[str] = set()

        for ref in alias_property_refs:
            alias = str(ref.get("alias", ""))
            prop = str(ref.get("property", ""))
            labels = alias_to_labels.get(alias, set())
            if not labels or not prop:
                continue
            valid = False
            for label in labels:
                if prop in known_props_by_label.get(label, set()):
                    valid = True
                    break
            if not valid:
                label_text = "|".join(sorted(labels))
                unknown_node_properties.add(f"{label_text}.{prop}")

        for ref in node_map_property_refs:
            label = str(ref.get("label", ""))
            prop = str(ref.get("property", ""))
            if not label or not prop:
                continue
            if prop not in known_props_by_label.get(label, set()):
                unknown_node_properties.add(f"{label}.{prop}")

        return {
            "used_labels": sorted(used_labels),
            "used_relationships": sorted(used_relationships),
            "unknown_labels": unknown_labels,
            "unknown_relationships": unknown_relationships,
            "unknown_node_properties": sorted(unknown_node_properties),
            "is_valid": not unknown_labels and not unknown_relationships and not unknown_node_properties,
        }

    def _apply_schema_guard_to_candidate_plans(
        self,
        candidate_plans: Dict[str, Any],
        meta_kg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Filter/penalize plans that reference non-existent schema elements.
        Covers:
        - GDB labels/relationship types
        - GDB label-specific properties
        - RDB tables/qualified columns
        """
        plans = list(candidate_plans.get("plans", []))
        if not plans:
            return candidate_plans

        valid_plans: List[Dict[str, Any]] = []
        invalid_plans: List[Dict[str, Any]] = []

        for plan in plans:
            compiled_plan = self._compile_plan_for_execution(plan, meta_kg)
            report = compiled_plan.get("schema_guard_report", {})
            compiler_report = compiled_plan.get("schema_compiler_report", {})
            is_valid = bool(compiler_report.get("is_valid", report.get("is_valid", True)))

            if is_valid:
                valid_plans.append(compiled_plan)
                continue

            # Keep invalid plans only as fallback with a very high risk.
            penalized = copy.deepcopy(compiled_plan)
            base_risk = penalized.get("predicted_error_risk", 0.5)
            if not isinstance(base_risk, (int, float)):
                base_risk = 0.5
            penalized["predicted_error_risk"] = max(0.98, float(base_risk))
            penalized["schema_guard_report"] = report
            penalized["schema_guard_penalized"] = True
            invalid_plans.append(penalized)
            self._debug_print(
                "schema guard reject: "
                f"plan_id={plan.get('plan_id')} "
                f"unknown_labels={report.get('unknown_labels')} "
                f"unknown_relationships={report.get('unknown_relationships')} "
                f"unknown_node_properties={report.get('unknown_node_properties')} "
                f"unknown_tables={report.get('unknown_tables')} "
                f"unknown_columns={report.get('unknown_columns')} "
                f"topology_violations={report.get('topology_violations')} "
                f"compile_errors={compiler_report.get('errors', [])}"
            )

        # If any valid plan exists, drop invalid plans to avoid avoidable runtime errors.
        if valid_plans:
            if invalid_plans:
                self._debug_print(
                    "schema guard: dropped invalid plans "
                    f"count={len(invalid_plans)} (schema/topology violations)."
                )
            guarded = copy.deepcopy(candidate_plans)
            guarded["plans"] = valid_plans
            return guarded

        # If all plans are invalid, keep penalized versions so the run can still proceed.
        self._debug_print("schema guard: all candidate plans violate schema/topology checks; using penalized fallback.")
        guarded = copy.deepcopy(candidate_plans)
        guarded["plans"] = invalid_plans if invalid_plans else plans
        return guarded

    def _is_plan_schema_valid(self, plan: Dict[str, Any], meta_kg: Dict[str, Any]) -> bool:
        compiled = self._compile_plan_for_execution(plan, meta_kg)
        compiler_report = compiled.get("schema_compiler_report", {})
        return bool(compiler_report.get("is_valid", True))

    @staticmethod
    def _infer_candidate_entities_from_query(
        user_query: str,
        meta_kg: Dict[str, Any],
        max_candidates: int = 6,
    ) -> List[str]:
        """Infer likely schema entities for staged/detailed injection."""
        tokens = set(re.findall(r"[A-Za-z_]+", (user_query or "").lower()))

        rdb_tables = list((meta_kg.get("rdb") or {}).keys())
        gdb_nodes = list((meta_kg.get("gdb") or {}).get("node_labels", []))
        universe = rdb_tables + gdb_nodes

        candidates: List[str] = []
        for name in universe:
            lowered = name.lower()
            parts = [p for p in re.split(r"[_\s]+", lowered) if p]
            if lowered in tokens or lowered.rstrip("s") in tokens or any(part in tokens for part in parts):
                candidates.append(name)

        keyword_map = {
            "patient": ["Patient", "sample_patient"],
            "admission": ["Admission", "AdminRecord", "sample_admin"],
            "day": ["Day"],
            "diagnosis": ["Diagnosis", "sample_diag"],
            "lab": ["Lab", "Lab", "sample_lab"],
            "medication": ["Medi", "Medi", "sample_medi"],
            "medi": ["Medi", "sample_medi"],
            "prescription": ["Medi", "sample_medi"],
            "drug": ["Medi", "sample_medi"],
            "sodium": ["Medi", "sample_medi"],
            "chloride": ["Medi", "sample_medi"],
            "vital": ["Vital", "sample_vital"],
            "operation": ["Operation", "sample_operation"],
            "physical": ["Physical", "sample_physical"],
            "visit": ["AdminRecord", "Admission", "sample_admin"],
            "confirmed": ["AdminRecord", "sample_admin"],
            "inpatient": ["AdminRecord", "sample_admin"],
            "outpatient": ["AdminRecord", "sample_admin"],
            "stay": ["AdminRecord", "Admission", "sample_admin"],
            "department": ["AdminRecord", "Admission", "sample_admin"],
        }

        q_low = (user_query or "").lower()
        for key, mapped in keyword_map.items():
            if key in q_low:
                for item in mapped:
                    if item in universe:
                        candidates.append(item)

        # Project rule: event-centric queries in GDB should traverse Day timeline.
        event_keywords = [
            "lab",
            "diagnosis",
            "medication",
            "medi",
            "prescription",
            "drug",
            "vital",
            "operation",
            "physical",
            "timeline",
            "event",
        ]
        if any(k in q_low for k in event_keywords):
            for day_name in ("Day", "day"):
                if day_name in universe:
                    candidates.append(day_name)

        unique: List[str] = []
        seen: Set[str] = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:max_candidates]

    def _build_schema_context(
        self,
        user_query: str,
        meta_kg: Dict[str, Any],
        table_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Build schema context with summary + optional staged/detailed injection.
        If caller does not provide table_names, infer likely entities from user query.
        """
        summary_context = self.meta_kg_generator.inject_schema_context(user_query, table_names=None)

        used_entities = list(table_names) if table_names else self._infer_candidate_entities_from_query(user_query, meta_kg)
        if not used_entities:
            return {
                "schema_context": summary_context,
                "injection_mode": "summary_only",
                "injected_entities": [],
            }

        detailed_context = self.meta_kg_generator.inject_schema_context(user_query, table_names=used_entities)
        return {
            "schema_context": summary_context + "\n\n" + detailed_context,
            "injection_mode": "summary_plus_detailed",
            "injected_entities": used_entities,
        }

    @staticmethod
    def _extract_execution_payload(execution_result: Dict[str, Any]) -> Any:
        if not isinstance(execution_result, dict):
            return None
        if execution_result.get("execution_mode") == "multi_step":
            payload: List[Dict[str, Any]] = []
            for step in execution_result.get("step_results", []):
                if not isinstance(step, dict):
                    continue
                payload.append(
                    {
                        "step_number": step.get("step_number"),
                        "db_type": step.get("db_type"),
                        "query": step.get("query"),
                        "result": step.get("result"),
                        "status": step.get("status"),
                    }
                )
            return payload
        return execution_result.get("result")

    @staticmethod
    def _is_empty_like_result(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            token = value.strip().lower()
            return token in {"", "none", "null", "nan", "[]", "{}"}
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return True
            return all(BaseHybridAgent._is_empty_like_result(item) for item in value)
        if isinstance(value, dict):
            if not value:
                return True
            return all(BaseHybridAgent._is_empty_like_result(v) for v in value.values())
        return False

    def _is_insufficient_payload(self, payload: Any) -> bool:
        """
        Decide whether execution payload is effectively empty/insufficient.
        This is used as an additional recovery trigger even when execution status is success.
        """
        # Multi-step payload shape from _extract_execution_payload:
        # [{"step_number":..., "db_type":..., "query":..., "result":..., "status":...}, ...]
        if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
            if any("result" in item for item in payload):
                step_results = [item.get("result") for item in payload if isinstance(item, dict)]
                return len(step_results) == 0 or all(self._is_empty_like_result(r) for r in step_results)
        return self._is_empty_like_result(payload)

    @staticmethod
    def _compact_payload_for_answer(payload: Any, max_rows: int = 24, max_fields: int = 16, max_chars: int = 5000) -> str:
        def _trim(obj: Any, depth: int = 0) -> Any:
            if depth > 4:
                return "<truncated_depth>"
            if isinstance(obj, dict):
                keys = list(obj.keys())[:max_fields]
                return {str(k): _trim(obj.get(k), depth + 1) for k in keys}
            if isinstance(obj, list):
                return [_trim(item, depth + 1) for item in obj[:max_rows]]
            return obj

        compact_obj = _trim(_to_json_serializable(payload))
        text = json.dumps(compact_obj, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 16] + "...<truncated>"

    @staticmethod
    def _is_insufficient_answer(answer: str) -> bool:
        text = str(answer or "").strip().lower()
        if not text:
            return True
        if text == "i could not find enough data to answer.":
            return True
        return "could not find enough data" in text

    def _generate_natural_answer(
        self,
        user_query: str,
        selected_plan: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> str:
        """Generate concise natural-language answer from executed query result."""
        if not isinstance(execution_result, dict) or execution_result.get("status") != "success":
            return ""

        payload = self._extract_execution_payload(execution_result)
        payload_json = self._compact_payload_for_answer(payload, max_rows=24, max_fields=16, max_chars=5000)
        db_type = str(selected_plan.get("db_type", "RDB"))

        prompt = f"""
You are a clinical data QA assistant.
Given the user question and execution payload, produce one concise factual answer.

USER_QUESTION:
{user_query}

SELECTED_DB_TYPE:
{db_type}

EXECUTION_RESULT_JSON:
{payload_json}

Rules:
1) Answer using only information present in EXECUTION_RESULT_JSON.
2) If result is empty/insufficient, return exactly: "I could not find enough data to answer."
3) Do not mention SQL/Cypher internals.
4) Keep answer short and factual.
5) Copy numeric values exactly; do not estimate or round.

Return JSON only:
{{
  "natural_answer": "..."
}}
"""
        try:
            raw = self.llm.invoke(prompt, json_mode=True)
            parsed = _extract_json_object(raw)
            answer = str(parsed.get("natural_answer", "")).strip()
            if answer:
                return answer
        except Exception:
            pass

        return self._shorten(payload_json, max_len=500)

    def _build_selection_from_plan(self, plan: Dict[str, Any], all_plans: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "selected_plan": plan,
            "selected_plan_id": plan.get("plan_id"),
            "selected_plan_db_type": plan.get("db_type"),
            "selected_plan_query": plan.get("query"),
            "selected_plan_caqa_score": plan.get("risk_aware_caqa_score", plan.get("caqa_score")),
            "selected_plan_expected_cost": plan.get("expected_cost", plan.get("estimated_cost")),
            "selected_plan_predicted_error_risk": plan.get("predicted_error_risk"),
            "selected_score_key": "risk_aware_caqa_score",
            "all_plans": all_plans,
        }

    def _regenerate_plan_from_error(
        self,
        user_query: str,
        schema_context: str,
        failed_plan: Dict[str, Any],
        error_log: str,
        recovery_attempt: int,
    ) -> Optional[Dict[str, Any]]:
        error_guided_hints = self._build_error_guided_repair_hints(error_log)
        recovery_prompt = f"""
You are a query repair assistant for a hybrid RDB/GDB agent.
Repair the failed query using schema context and runtime error.
Return JSON only.

USER_QUERY:
{user_query}

FAILED_PLAN_DB_TYPE:
{failed_plan.get("db_type", "RDB")}

FAILED_QUERY:
{failed_plan.get("query", "")}

ERROR_LOG:
{error_log}

ERROR_GUIDED_HINTS:
{error_guided_hints}

SCHEMA_CONTEXT:
{schema_context}

Repair rules:
1) Use only names in SCHEMA_CONTEXT. Never invent schema elements.
2) Return one executable SQL/Cypher query (or valid BOTH Step format only if truly needed).
3) No placeholders ($x, ?x, <x>) and no narrative text in query.
4) patientId literals must be quoted strings in SQL/Cypher.
5) For GDB event queries, use Patient-[:HAS_ADMISSION]->Admission, then Admission/Day NEXT traversal, followed by Day-[:HAS_*]->Event. Do not repair to Admission-[:HAS_DAY]->Day in this MIMIC runtime.
6) For day-level timeline repairs, first select latest Admission using schema-proven date field(s),
    then anchor Day rows with patientId + admissionId when available.
7) For sequence questions, chain d1 -> NEXT -> d2 (avoid restarting from Admission for second hop).
8) For aggregate repairs, avoid fan-out inflation; use staged aggregation and COUNT(DISTINCT key).
9) In GDB repairs, do not add event.admissionId filters by default. The NEXT path preserves admission scope; add event.admissionId only when schema and value format alignment are confirmed.
10) If "admissions with medication" returns zero unexpectedly, prefer Day-mediated medication existence path
     when direct Admission-[:HAS_MEDI]->Medi edge is not clearly confirmed.
11) If error is empty/insufficient result, relax brittle predicates while preserving semantics.
12) Cypher scope rule: every variable in WHERE/RETURN/ORDER BY must remain bound after each WITH.
13) Single statement rule: do not concatenate multiple statements with ';' in one query/step.
14) If db_type=BOTH, use exact step headers and line breaks:
    Step 1) RDB:
    <single SQL statement>
    Step 2) GDB:
    <single Cypher statement>
15) If BOTH format cannot be made executable, switch to one reliable single-source query instead of malformed multi-step text.
16) For Cypher variable-length paths, use concrete numeric bounds only (e.g., *0..100), never symbolic bounds like *0..N.
17) In repairs, re-check MetaKG concept-name fields before changing a predicate: diagnosis text uses diagnosisName, lab names use testName, medication names use genericName, operation text uses operationName, and physical/vital measurement names use label.
18) Do not repair an unknown column by changing only letter case or by substituting a generic name field. The replacement must be explicitly listed under the same table/label in SCHEMA_CONTEXT.

Return JSON only:
{{
  "db_type": "RDB or GDB or BOTH",
  "is_multi_step": false,
  "query": "fixed executable query",
  "fix_rationale": "short reason",
  "predicted_error_risk": 0.2
}}
"""
        try:
            raw = self.llm.invoke(recovery_prompt, json_mode=True)
            data = _extract_json_object(raw)
            query = str(data.get("query", "")).strip()
            if not query:
                return None

            db_type = str(data.get("db_type", failed_plan.get("db_type", "RDB"))).upper()
            if db_type not in ("RDB", "GDB", "BOTH"):
                db_type = str(failed_plan.get("db_type", "RDB")).upper()
            if db_type not in ("RDB", "GDB", "BOTH"):
                db_type = "RDB"

            risk = data.get("predicted_error_risk", failed_plan.get("predicted_error_risk", 0.5))
            if not isinstance(risk, (int, float)):
                risk = 0.5
            predicted_error_risk = max(0.0, min(1.0, float(risk)))

            expected_cost = float(failed_plan.get("expected_cost", failed_plan.get("estimated_cost", 1000.0)))
            regenerated_plan = {
                "plan_id": f"RECOVERY_{recovery_attempt}",
                "description": f"LLM repaired plan from runtime error (attempt {recovery_attempt})",
                "db_type": db_type,
                "query": query,
                "is_multi_step": bool(data.get("is_multi_step", False)),
                "query_rationale": str(data.get("fix_rationale", "Runtime error guided query repair")),
                "complexity_factors": copy.deepcopy(failed_plan.get("complexity_factors", {"omega": 1})),
                "expected_cost": expected_cost,
                "estimated_cost": expected_cost,
                "caqa_score": expected_cost,
                "predicted_error_risk": predicted_error_risk,
                "risk_lambda": self.risk_lambda,
                "risk_aware_caqa_score": expected_cost + self.risk_lambda * predicted_error_risk,
                "selection_objective": "expected_cost + risk_lambda * predicted_error_risk",
            }
            return regenerated_plan
        except Exception as exc:
            self._debug_print(f"recovery regeneration failed: {self._shorten(exc)}")
            return None

    def run(
        self,
        user_query: str,
        table_names: Optional[List[str]] = None,
        decomposed_info: str = "",
        actual_cardinalities: Optional[Dict[str, int]] = None,
        predicted_risks: Optional[Dict[str, float]] = None,
        force_refresh_meta_kg: bool = False,
    ) -> Dict[str, Any]:
        if not user_query or not user_query.strip():
            raise ValueError("user_query must not be empty.")

        run_started = time.time()
        self._debug_print("=== Agent Run Started ===")
        self._debug_print(f"user_query: {user_query}")
        self._debug_print(
            "config: "
            f"risk_lambda={self.risk_lambda}, "
            f"normalize_expected_cost={self.normalize_expected_cost}, "
            f"enable_recovery={self.enable_recovery}, "
            f"recovery_policy={self.recovery_policy}, "
            f"recovery_threshold={self.recovery_threshold}, "
            f"max_recovery_attempts={self.max_recovery_attempts}"
        )
        recovery_usage_started_at: Optional[Dict[str, float]] = None

        had_cached_meta_kg = bool(self.meta_kg_generator.meta_kg)
        meta_kg = self.meta_kg_generator.generate_meta_kg(force_refresh=force_refresh_meta_kg)
        if force_refresh_meta_kg:
            meta_kg_source = "refreshed"
        else:
            meta_kg_source = "cache" if had_cached_meta_kg else "fresh"
        self._debug_print(
            "meta_kg loaded: "
            f"source={meta_kg_source}, "
            f"rdb_tables={len(meta_kg.get('rdb', {}))}, "
            f"gdb_nodes={len(meta_kg.get('gdb', {}).get('node_labels', []))}, "
            f"cross_source_keys={len(meta_kg.get('cross_source_keys', [])) if isinstance(meta_kg.get('cross_source_keys'), list) else 0}"
        )
        schema_pack = self._build_schema_context(user_query, meta_kg, table_names=table_names)
        schema_context = str(schema_pack.get("schema_context", ""))
        injection_mode = str(schema_pack.get("injection_mode", "summary_only"))
        injected_entities = list(schema_pack.get("injected_entities", []))
        self._debug_print(f"schema_context length: {len(schema_context)}")
        self._debug_print(
            "schema injection mode: "
            f"{injection_mode}, injected_entities={injected_entities if injected_entities else '[]'}"
        )
        has_available_tables = '"available_tables"' in schema_context
        has_available_nodes = '"available_nodes"' in schema_context
        has_available_relationships = '"available_relationships"' in schema_context
        has_detailed_schema = ("Detailed" in schema_context) or ("detailed" in schema_context)
        self._debug_print(
            "schema_context flags: "
            f"available_tables={has_available_tables}, "
            f"available_nodes={has_available_nodes}, "
            f"available_relationships={has_available_relationships}, "
            f"detailed_schema={has_detailed_schema}"
        )
        self._debug_print(f"schema_context preview: {self._shorten(schema_context, max_len=500)}")

        candidate_plans = self.query_planner.generate_plans(user_query, schema_context, decomposed_info)
        candidate_count = len(candidate_plans.get("plans", []))
        self._debug_print(f"candidate plans generated: {candidate_count}")
        candidate_plans = self._apply_schema_guard_to_candidate_plans(candidate_plans, meta_kg)
        guarded_count = len(candidate_plans.get("plans", []))
        if guarded_count != candidate_count:
            self._debug_print(
                "candidate plans after schema guard: "
                f"{guarded_count} (from {candidate_count})"
            )

        scored_plans = self.caqa_evaluator.calculate_caqa_scores(
            candidate_plans,
            actual_cardinalities=actual_cardinalities,
            predicted_risks=predicted_risks,
            risk_lambda=self.risk_lambda,
            normalize_expected_cost=self.normalize_expected_cost,
            user_query=user_query,
        )
        for plan in scored_plans.get("plans", []):
            factors = plan.get("complexity_factors", {}) if isinstance(plan.get("complexity_factors"), dict) else {}
            compiler_report = plan.get("schema_compiler_report", {}) if isinstance(plan.get("schema_compiler_report"), dict) else {}
            self._debug_print(
                "plan: "
                f"id={plan.get('plan_id')}, "
                f"db={plan.get('db_type')}, "
                f"omega={factors.get('omega')}, "
                f"join_or_hop={factors.get('C_join_or_hop')}, "
                f"filter_pred={factors.get('C_filter_pred')}, "
                f"agg={factors.get('C_agg')}, "
                f"expected_cost={plan.get('expected_cost')}, "
                f"risk={plan.get('predicted_error_risk')}, "
                f"score={plan.get('risk_aware_caqa_score', plan.get('caqa_score'))}"
            )
            if compiler_report.get("errors") or compiler_report.get("warnings"):
                self._debug_print(
                    "plan compiler: "
                    f"id={plan.get('plan_id')}, "
                    f"errors={compiler_report.get('errors', [])}, "
                    f"warnings={compiler_report.get('warnings', [])}"
                )

        selection = self.caqa_evaluator.select_best_plan(scored_plans)
        selected_plan = selection["selected_plan"]
        self._debug_print(
            "selected plan: "
            f"id={selection.get('selected_plan_id')}, "
            f"db={selection.get('selected_plan_db_type')}, "
            f"score={selection.get('selected_plan_caqa_score')}, "
            f"risk={selection.get('selected_plan_predicted_error_risk')}"
        )
        self._debug_print(f"selected query: {self._shorten(selected_plan.get('query', ''))}")

        try:
            execution_result = self.query_executor.execute_plan(selection["selected_plan"])
        except Exception as exc:
            execution_result = {
                "status": "failed",
                "execution_mode": "single",
                "db_type": str(selection["selected_plan"].get("db_type", "UNKNOWN")),
                "query": str(selection["selected_plan"].get("query", "")),
                "error": f"execute_plan_exception: {exc}",
            }
        self._debug_print(
            "primary execution: "
            f"status={execution_result.get('status')}, "
            f"mode={execution_result.get('execution_mode', 'single')}"
        )
        if execution_result.get("status") != "success":
            self._debug_print(f"primary execution error: {self._shorten(execution_result.get('error'))}")
        elif execution_result.get("auto_fix_applied"):
            self._debug_print(f"primary execution auto_fix: {execution_result.get('auto_fix_applied')}")

        recovery_invoked = False
        recovery_selected = None
        recovery_result = None
        recovery_attempts = 0
        tried_plan_ids: Set[str] = {str(selected_plan.get("plan_id"))}
        primary_payload = self._extract_execution_payload(execution_result)
        primary_insufficient_payload = self._is_insufficient_payload(primary_payload)
        if primary_insufficient_payload:
            self._debug_print("primary execution payload is empty/insufficient")

        if self.enable_recovery:
            selected_plan = selection["selected_plan"]
            selected_risk = float(selected_plan.get("predicted_error_risk", 0.0) or 0.0)
            exec_ok = execution_result.get("status") == "success"

            should_recover = False
            if self.recovery_policy == "always":
                should_recover = True
            elif self.recovery_policy == "on_failure":
                should_recover = (not exec_ok) or primary_insufficient_payload
            elif self.recovery_policy == "uncertainty":
                # Uncertainty mode also recovers on hard execution failure or insufficient payload.
                should_recover = (selected_risk >= self.recovery_threshold) or (not exec_ok) or primary_insufficient_payload
            elif self.recovery_policy == "never":
                should_recover = False

            self._debug_print(
                "recovery check: "
                f"exec_ok={exec_ok}, selected_risk={selected_risk}, "
                f"insufficient_payload={primary_insufficient_payload}, should_recover={should_recover}"
            )

            if should_recover:
                recovery_invoked = True
                if recovery_usage_started_at is None:
                    recovery_usage_started_at = self._safe_llm_snapshot(self.llm)
                last_error = self._extract_execution_error(execution_result)
                while recovery_attempts < self.max_recovery_attempts:
                    recovery_attempts += 1
                    self._debug_print(f"recovery attempt {recovery_attempts}/{self.max_recovery_attempts}")

                    candidate = None
                    candidate_source = ""

                    if last_error:
                        regenerated = self._regenerate_plan_from_error(
                            user_query=user_query,
                            schema_context=schema_context,
                            failed_plan=selection["selected_plan"],
                            error_log=last_error,
                            recovery_attempt=recovery_attempts,
                        )
                        if regenerated:
                            candidate = regenerated
                            candidate_source = "llm_repair"

                    if candidate is None:
                        fallback_candidates = [
                            p for p in scored_plans.get("plans", []) if str(p.get("plan_id")) not in tried_plan_ids
                        ]
                        if not fallback_candidates:
                            self._debug_print("no fallback candidates left for recovery")
                            break
                        candidate = min(
                            fallback_candidates,
                            key=lambda p: float(p.get("risk_aware_caqa_score", p.get("caqa_score", 1e18))),
                        )
                        candidate_source = "ranked_fallback"
                    tried_plan_ids.add(str(candidate.get("plan_id")))

                    recovery_selected = candidate
                    self._debug_print(
                        "recovery selected plan: "
                        f"source={candidate_source}, id={candidate.get('plan_id')}, db={candidate.get('db_type')}, "
                        f"score={candidate.get('risk_aware_caqa_score', candidate.get('caqa_score'))}"
                    )
                    self._debug_print(f"recovery query: {self._shorten(candidate.get('query', ''))}")

                    if not self._is_plan_schema_valid(candidate, meta_kg):
                        report = self._validate_plan_schema_refs(candidate, meta_kg)
                        last_error = (
                            "schema_guard_reject: "
                            f"unknown_labels={report.get('unknown_labels')} "
                            f"unknown_relationships={report.get('unknown_relationships')} "
                            f"unknown_tables={report.get('unknown_tables')} "
                            f"unknown_columns={report.get('unknown_columns')} "
                            f"topology_violations={report.get('topology_violations')}"
                        )
                        self._debug_print(self._shorten(last_error))
                        tried_plan_ids.add(str(candidate.get("plan_id")))
                        continue

                    try:
                        recovery_result = self.query_executor.execute_plan(candidate)
                    except Exception as exc:
                        recovery_result = {
                            "status": "failed",
                            "execution_mode": "single",
                            "db_type": str(candidate.get("db_type", "UNKNOWN")),
                            "query": str(candidate.get("query", "")),
                            "error": f"execute_plan_exception: {exc}",
                        }
                    self._debug_print(
                        "recovery execution: "
                        f"status={recovery_result.get('status')}, "
                        f"mode={recovery_result.get('execution_mode', 'single')}"
                    )

                    if recovery_result.get("status") == "success":
                        candidate_payload = self._extract_execution_payload(recovery_result)
                        if self._is_insufficient_payload(candidate_payload):
                            last_error = "insufficient_result_payload"
                            self._debug_print("recovery execution returned insufficient payload; continuing recovery attempts")
                            continue
                        selection = self._build_selection_from_plan(candidate, scored_plans.get("plans", []))
                        execution_result = recovery_result
                        self._debug_print("recovery succeeded and replaced current selection")
                        break

                    last_error = self._extract_execution_error(recovery_result)
                    self._debug_print(f"recovery execution error: {self._shorten(last_error or 'unknown error')}")

        self._debug_print(
            "final execution: "
            f"status={execution_result.get('status')}, "
            f"selected_plan_id={selection.get('selected_plan_id')}"
        )
        if execution_result.get("status") == "success":
            preview = execution_result.get("result")
            if preview is None and execution_result.get("step_results"):
                preview = execution_result.get("step_results")
            self._debug_print(f"final result preview: {self._shorten(preview)}")
        else:
            self._debug_print(f"final error: {self._shorten(execution_result.get('error'))}")

        generated_natural_answer = self._generate_natural_answer(
            user_query=user_query,
            selected_plan=selection.get("selected_plan", {}),
            execution_result=execution_result,
        )
        if generated_natural_answer:
            self._debug_print(f"generated natural answer: {self._shorten(generated_natural_answer)}")

        # If final answer explicitly indicates insufficient data, try additional fallback recovery.
        if (
            self.enable_recovery
            and self.recovery_policy != "never"
            and self._is_insufficient_answer(generated_natural_answer)
            and recovery_attempts < self.max_recovery_attempts
        ):
            recovery_invoked = True
            if recovery_usage_started_at is None:
                recovery_usage_started_at = self._safe_llm_snapshot(self.llm)
            self._debug_print(
                "answer-based recovery trigger: "
                "final answer indicates insufficient data, trying fallback plans"
            )
            while recovery_attempts < self.max_recovery_attempts:
                recovery_attempts += 1
                self._debug_print(
                    f"answer-based recovery attempt {recovery_attempts}/{self.max_recovery_attempts}"
                )
                fallback_candidates = [
                    p for p in scored_plans.get("plans", []) if str(p.get("plan_id")) not in tried_plan_ids
                ]
                if not fallback_candidates:
                    self._debug_print("no fallback candidates left for answer-based recovery")
                    break

                candidate = min(
                    fallback_candidates,
                    key=lambda p: float(p.get("risk_aware_caqa_score", p.get("caqa_score", 1e18))),
                )
                tried_plan_ids.add(str(candidate.get("plan_id")))
                recovery_selected = candidate
                self._debug_print(
                    "answer-based recovery selected fallback: "
                    f"id={candidate.get('plan_id')}, db={candidate.get('db_type')}, "
                    f"score={candidate.get('risk_aware_caqa_score', candidate.get('caqa_score'))}"
                )

                if not self._is_plan_schema_valid(candidate, meta_kg):
                    self._debug_print("answer-based recovery candidate rejected by schema guard")
                    continue

                try:
                    candidate_exec = self.query_executor.execute_plan(candidate)
                except Exception as exc:
                    candidate_exec = {
                        "status": "failed",
                        "execution_mode": "single",
                        "db_type": str(candidate.get("db_type", "UNKNOWN")),
                        "query": str(candidate.get("query", "")),
                        "error": f"execute_plan_exception: {exc}",
                    }

                self._debug_print(
                    "answer-based recovery execution: "
                    f"status={candidate_exec.get('status')}, "
                    f"mode={candidate_exec.get('execution_mode', 'single')}"
                )
                if candidate_exec.get("status") != "success":
                    self._debug_print(
                        f"answer-based recovery execution error: "
                        f"{self._shorten(self._extract_execution_error(candidate_exec) or candidate_exec.get('error'))}"
                    )
                    continue

                candidate_payload = self._extract_execution_payload(candidate_exec)
                if self._is_insufficient_payload(candidate_payload):
                    self._debug_print("answer-based recovery candidate payload still insufficient")
                    continue

                candidate_answer = self._generate_natural_answer(
                    user_query=user_query,
                    selected_plan=candidate,
                    execution_result=candidate_exec,
                )
                if self._is_insufficient_answer(candidate_answer):
                    self._debug_print("answer-based recovery candidate answer still insufficient")
                    continue

                selection = self._build_selection_from_plan(candidate, scored_plans.get("plans", []))
                execution_result = candidate_exec
                recovery_result = candidate_exec
                generated_natural_answer = candidate_answer
                self._debug_print("answer-based recovery succeeded and replaced current selection")
                break

        execution_payload = self._extract_execution_payload(execution_result)
        recovery_usage_delta = self._safe_llm_delta(
            self.llm,
            self._safe_llm_snapshot(self.llm),
            recovery_usage_started_at,
        )
        self._debug_print(
            "recovery usage: "
            f"calls={int(recovery_usage_delta.get('llm_calls', 0.0))}, "
            f"tokens={int(recovery_usage_delta.get('total_tokens', 0.0))}"
        )

        self._debug_print(f"run elapsed: {round(time.time() - run_started, 3)} sec")
        self._debug_print("=== Agent Run Finished ===")

        return {
            "user_query": user_query,
            "meta_kg": meta_kg,
            "schema_context": schema_context,
            "schema_injection_mode": injection_mode,
            "schema_injected_entities": injected_entities,
            "candidate_plans": candidate_plans,
            "scored_plans": scored_plans,
            "selected_plan": selection,
            "execution_result": execution_result,
            "execution_result_payload": execution_payload,
            "generated_natural_answer": generated_natural_answer,
            "recovery_invoked": recovery_invoked,
            "recovery_attempts": recovery_attempts,
            "recovery_selected_plan": recovery_selected,
            "recovery_result": recovery_result,
            "recovery_token_usage": recovery_usage_delta,
            "recovery_prompt_tokens": int(recovery_usage_delta.get("prompt_tokens", 0.0)),
            "recovery_completion_tokens": int(recovery_usage_delta.get("completion_tokens", 0.0)),
            "recovery_total_tokens": int(recovery_usage_delta.get("total_tokens", 0.0)),
            "recovery_llm_calls": int(recovery_usage_delta.get("llm_calls", 0.0)),
            "recovery_llm_latency_sec": float(recovery_usage_delta.get("llm_latency_sec", 0.0)),
        }


# =====================================================================
# ReAct wrapper (merged from the former react_agent.py; single-file).
# =====================================================================
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

_HARDENING_TAG = "REACT_PROMPT_HARDENING_V260225"
_METAKG_HARDENING_BLOCK = (
    f"\n[{_HARDENING_TAG}] Use only exact introspected schema names; emit cross-source keys only for high-confidence ID matches; omit uncertain mappings.\n"
)


_MIMIC_QUERY_SAFETY_TAG = "MIMIC_QUERY_SAFETY_V260429"
_MIMIC_QUERY_SAFETY_BLOCK = (
    f"\n[{_MIMIC_QUERY_SAFETY_TAG}] Exact CamelCase RDB tables; reach all days via Day{{admissionKey:a.admissionKey}} (not (a)-[:NEXT]->Day); diagnoses on Day{{index:0}} (seq_num 1-5); toFloat() numeric props, no datetime() rewrap; one statement per step; quote string IDs; COUNT(DISTINCT) to avoid fan-out.\n"
)

if isinstance(PROMPTS, dict):
    base_meta_prompt = str(PROMPTS.get("meta_kg", ""))
    if _HARDENING_TAG not in base_meta_prompt:
        PROMPTS["meta_kg"] = base_meta_prompt + _METAKG_HARDENING_BLOCK

    for _prompt_key in (
        "planner_rules",
        "schema_validation_checklist",
        "schema_context",
        "sql_generation",
        "cypher_generation",
        "query_generation",
        "answer_synthesis",
    ):
        _base_prompt = str(PROMPTS.get(_prompt_key, ""))
        if _base_prompt and _MIMIC_QUERY_SAFETY_TAG not in _base_prompt:
            PROMPTS[_prompt_key] = _base_prompt + _MIMIC_QUERY_SAFETY_BLOCK


class ToolSpec:
    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        self.name = name
        self.description = description
        self.handler = handler


class ReActHybridAgent(BaseHybridAgent):
    """Tool-driven ReAct wrapper around the existing hybrid agent core."""

    def __init__(
        self,
        rdb: Any,
        neo4j_driver: Any,
        llm: OpenAILLM,
        complexity_constants: Optional[ComplexityConstants] = None,
        risk_lambda: float = 1.0,
        normalize_expected_cost: bool = True,
        enable_recovery: bool = True,
        recovery_policy: str = "uncertainty",
        recovery_threshold: float = 0.5,
        max_recovery_attempts: int = 3,
        max_react_iterations: int = 12,
        react_print: bool = True,
        debug: bool = True,
    ) -> None:
        super().__init__(
            rdb=rdb,
            neo4j_driver=neo4j_driver,
            llm=llm,
            complexity_constants=complexity_constants,
            risk_lambda=risk_lambda,
            normalize_expected_cost=normalize_expected_cost,
            enable_recovery=enable_recovery,
            recovery_policy=recovery_policy,
            recovery_threshold=recovery_threshold,
            max_recovery_attempts=max_recovery_attempts,
            debug=debug,
        )
        self.max_react_iterations = max(1, int(max_react_iterations))
        self.react_print = bool(react_print)
        self._planner_hardening = (
            f"[{_HARDENING_TAG}] Return exactly 2 executable plans, schema names only, no placeholders; BOTH = 'Step 1) RDB: ...' + 'Step 2) GDB: ...'; one statement each; quote string IDs; prefer the simpler valid plan.\n"
        )
        self._planner_hardening += _MIMIC_QUERY_SAFETY_BLOCK
        self._recovery_hardening = (
            f"[{_HARDENING_TAG}] Repair to one executable statement; never invent names; if BOTH cannot run, fall back to a reliable single-source query; quote string IDs.\n"
        )
        self._recovery_hardening += _MIMIC_QUERY_SAFETY_BLOCK
        self.tools = self._build_tool_registry()

    def _react_print(self, message: str) -> None:
        if not self.react_print:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"[react][{ts}] {message}")

    def _build_tool_registry(self) -> Dict[str, ToolSpec]:
        return {
            "meta_kg": ToolSpec(
                name="meta_kg",
                description="Build or load MetaKG snapshot.",
                handler=self.tool_meta_kg,
            ),
            "schema_injection": ToolSpec(
                name="schema_injection",
                description="Inject summary+detailed schema context from MetaKG.",
                handler=self.tool_schema_injection,
            ),
            "planner": ToolSpec(
                name="planner",
                description="Generate candidate plans from query and schema context.",
                handler=self.tool_planner,
            ),
            "schema_guard": ToolSpec(
                name="schema_guard",
                description="Compile/schema/topology guard on candidate plans.",
                handler=self.tool_schema_guard,
            ),
            "raqa": ToolSpec(
                name="raqa",
                description="Compute RAQA score and annotate plans.",
                handler=self.tool_raqa,
            ),
            "select_plan": ToolSpec(
                name="select_plan",
                description="Select best plan by RAQA score.",
                handler=self.tool_select_plan,
            ),
            "execute_query": ToolSpec(
                name="execute_query",
                description="Execute selected SQL/Cypher/BOTH plan.",
                handler=self.tool_execute_query,
            ),
            "recovery": ToolSpec(
                name="recovery",
                description="Policy-based recovery with max retry budget.",
                handler=self.tool_recovery,
            ),
            "answer_generation": ToolSpec(
                name="answer_generation",
                description="Generate natural language answer from execution payload.",
                handler=self.tool_answer_generation,
            ),
        }

    def _next_tool(self, state: Dict[str, Any]) -> str:
        if "meta_kg" not in state:
            return "meta_kg"
        if "schema_context" not in state:
            return "schema_injection"
        if "candidate_plans" not in state:
            return "planner"
        if not state.get("schema_guard_done", False):
            return "schema_guard"
        if "scored_plans" not in state:
            return "raqa"
        if "selection" not in state:
            return "select_plan"
        if "execution_result" not in state:
            return "execute_query"
        if not state.get("recovery_done", False):
            return "recovery"
        if "generated_natural_answer" not in state:
            return "answer_generation"
        return "finish"

    def _think(self, iteration: int, tool_name: str, state: Dict[str, Any]) -> str:
        # ReAct think step: an LLM-generated natural-language rationale emitted before each act.
        # The tool *schedule* is fixed (deterministic); this is the reasoning trace, not the controller.
        if tool_name == "finish":
            return "All required stages are complete; finalize the answer."
        try:
            prompt = (
                "You are a hybrid RDB/GDB EHR-QA agent operating in a think-act-observe loop "
                "with a fixed tool schedule. "
                f"Completed stages: {sorted(state.keys())}. The next tool is '{tool_name}'. "
                f"In ONE concise sentence, give the reasoning for invoking '{tool_name}' at this step."
            )
            rationale = self.llm.invoke(prompt)
            if isinstance(rationale, str) and rationale.strip():
                return rationale.strip()
        except Exception:
            pass
        return f"Iteration {iteration}: invoke `{tool_name}`."

    def _act(self, tool_name: str, state: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        tool = self.tools[tool_name]
        self._react_print(f"tool_start={tool_name}")
        return tool.handler(state, params or {})

    def _observe(self, tool_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "tool": tool_name,
            "status": observation.get("status", "ok"),
        }
        for key in (
            "meta_kg_source",
            "candidate_count",
            "candidate_count_before",
            "candidate_count_after",
            "selected_plan_id",
            "selected_plan_db_type",
            "execution_status",
            "recovery_invoked",
            "recovery_attempts",
            "answer_len",
        ):
            if key in observation:
                summary[key] = observation[key]
        return summary

    def tool_meta_kg(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        force_refresh = bool(params.get("force_refresh_meta_kg", False))
        had_cached = bool(self.meta_kg_generator.meta_kg)
        meta_kg = self.meta_kg_generator.generate_meta_kg(force_refresh=force_refresh)
        if force_refresh:
            source = "refreshed"
        else:
            source = "cache" if had_cached else "fresh"

        state["meta_kg"] = meta_kg
        state["meta_kg_source"] = source
        return {
            "status": "ok",
            "meta_kg_source": source,
            "rdb_tables": len(meta_kg.get("rdb", {})),
            "gdb_nodes": len(meta_kg.get("gdb", {}).get("node_labels", [])),
            "cross_source_keys": (
                len(meta_kg.get("cross_source_keys", []))
                if isinstance(meta_kg.get("cross_source_keys", []), list)
                else 0
            ),
        }

    def tool_schema_injection(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        user_query = str(state["user_query"])
        meta_kg = state["meta_kg"]
        table_names = params.get("table_names", state.get("table_names"))

        schema_pack = self._build_schema_context(user_query, meta_kg, table_names=table_names)
        schema_context = str(schema_pack.get("schema_context", ""))
        injection_mode = str(schema_pack.get("injection_mode", "summary_only"))
        injected_entities = list(schema_pack.get("injected_entities", []))

        state["schema_context"] = schema_context
        state["schema_injection_mode"] = injection_mode
        state["schema_injected_entities"] = injected_entities

        return {
            "status": "ok",
            "schema_context_chars": len(schema_context),
            "injection_mode": injection_mode,
            "injected_entities": injected_entities,
        }

    def tool_planner(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        user_query = str(state["user_query"])
        schema_context = str(state["schema_context"])
        decomposed_info = str(params.get("decomposed_info", state.get("decomposed_info", "")))
        if decomposed_info:
            decomposed_info = decomposed_info + "\n\n" + self._planner_hardening
        else:
            decomposed_info = self._planner_hardening

        candidate_plans = self.query_planner.generate_plans(user_query, schema_context, decomposed_info)
        state["candidate_plans"] = candidate_plans
        candidate_count = len(candidate_plans.get("plans", []))
        return {
            "status": "ok",
            "candidate_count": candidate_count,
        }

    def tool_schema_guard(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        candidate_plans = state["candidate_plans"]
        meta_kg = state["meta_kg"]
        before = len(candidate_plans.get("plans", []))
        guarded = self._apply_schema_guard_to_candidate_plans(candidate_plans, meta_kg)
        after = len(guarded.get("plans", []))
        state["candidate_plans"] = guarded
        state["schema_guard_done"] = True
        return {
            "status": "ok",
            "candidate_count_before": before,
            "candidate_count_after": after,
        }

    def tool_raqa(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        user_query = str(state["user_query"])
        candidate_plans = state["candidate_plans"]
        actual_cardinalities = params.get("actual_cardinalities", state.get("actual_cardinalities"))
        predicted_risks = params.get("predicted_risks", state.get("predicted_risks"))

        scored = self.caqa_evaluator.calculate_caqa_scores(
            candidate_plans,
            actual_cardinalities=actual_cardinalities,
            predicted_risks=predicted_risks,
            risk_lambda=self.risk_lambda,
            normalize_expected_cost=self.normalize_expected_cost,
            user_query=user_query,
        )
        state["scored_plans"] = scored

        plans = scored.get("plans", [])
        min_score = None
        if plans:
            values = [float(p.get("risk_aware_caqa_score", p.get("caqa_score", 1e18))) for p in plans]
            min_score = min(values)

        return {
            "status": "ok",
            "candidate_count": len(plans),
            "min_risk_aware_score": min_score,
        }

    def tool_select_plan(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        scored = state["scored_plans"]
        try:
            selection = self.caqa_evaluator.select_best_plan(scored)
        except Exception as exc:
            plans = scored.get("plans", [])
            if not plans:
                raise ValueError(f"No selectable plan exists: {exc}") from exc
            fallback = plans[0]
            selection = self._build_selection_from_plan(fallback, plans)

        state["selection"] = selection
        return {
            "status": "ok",
            "selected_plan_id": selection.get("selected_plan_id"),
            "selected_plan_db_type": selection.get("selected_plan_db_type"),
            "selected_plan_score": selection.get("selected_plan_caqa_score"),
        }

    def tool_execute_query(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        selection = state["selection"]
        selected_plan = selection["selected_plan"]

        try:
            execution_result = self.query_executor.execute_plan(selected_plan)
        except Exception as exc:
            execution_result = {
                "status": "failed",
                "execution_mode": "single",
                "db_type": str(selected_plan.get("db_type", "UNKNOWN")),
                "query": str(selected_plan.get("query", "")),
                "error": f"execute_plan_exception: {exc}",
            }

        payload = self._extract_execution_payload(execution_result)
        insufficient_payload = self._is_insufficient_payload(payload)
        state["execution_result"] = execution_result
        state["execution_result_payload"] = payload
        state["primary_insufficient_payload"] = insufficient_payload

        return {
            "status": "ok",
            "execution_status": execution_result.get("status"),
            "execution_mode": execution_result.get("execution_mode", "single"),
            "primary_insufficient_payload": insufficient_payload,
        }

    def tool_recovery(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        selection = state["selection"]
        execution_result = state["execution_result"]
        scored_plans = state["scored_plans"]
        user_query = str(state["user_query"])
        schema_context = str(state["schema_context"])
        meta_kg = state["meta_kg"]

        recovery_invoked = False
        recovery_selected = None
        recovery_result = None
        recovery_attempts = int(state.get("recovery_attempts", 0))
        tried_plan_ids: Set[str] = {str(selection["selected_plan"].get("plan_id"))}
        recovery_usage_started_at = None
        primary_payload = self._extract_execution_payload(execution_result)
        primary_insufficient_payload = self._is_insufficient_payload(primary_payload)

        if self.enable_recovery:
            selected_plan = selection["selected_plan"]
            selected_risk = float(selected_plan.get("predicted_error_risk", 0.0) or 0.0)
            exec_ok = execution_result.get("status") == "success"

            should_recover = False
            if self.recovery_policy == "always":
                should_recover = True
            elif self.recovery_policy == "on_failure":
                should_recover = (not exec_ok) or primary_insufficient_payload
            elif self.recovery_policy == "uncertainty":
                should_recover = (selected_risk >= self.recovery_threshold) or (not exec_ok) or primary_insufficient_payload
            elif self.recovery_policy == "never":
                should_recover = False

            if should_recover:
                recovery_invoked = True
                recovery_usage_started_at = self._safe_llm_snapshot(self.llm)
                last_error = self._extract_execution_error(execution_result)
                self._react_print(
                    "recovery_triggered: "
                    f"policy={self.recovery_policy}, "
                    f"risk={selected_risk:.3f}, exec_ok={exec_ok}, insufficient_payload={primary_insufficient_payload}"
                )

                while recovery_attempts < self.max_recovery_attempts:
                    recovery_attempts += 1
                    candidate = None
                    self._react_print(
                        f"recovery_attempt={recovery_attempts}/{self.max_recovery_attempts}, "
                        f"last_error={str(last_error)[:120]}"
                    )

                    if last_error:
                        repair_error_log = (
                            str(last_error) + "\n\n" + self._recovery_hardening
                        )
                        regenerated = self._regenerate_plan_from_error(
                            user_query=user_query,
                            schema_context=schema_context,
                            failed_plan=selection["selected_plan"],
                            error_log=repair_error_log,
                            recovery_attempt=recovery_attempts,
                        )
                        if regenerated:
                            candidate = regenerated

                    if candidate is None:
                        fallback_candidates = [
                            p for p in scored_plans.get("plans", []) if str(p.get("plan_id")) not in tried_plan_ids
                        ]
                        if not fallback_candidates:
                            break
                        candidate = min(
                            fallback_candidates,
                            key=lambda p: float(p.get("risk_aware_caqa_score", p.get("caqa_score", 1e18))),
                        )

                    tried_plan_ids.add(str(candidate.get("plan_id")))
                    recovery_selected = candidate

                    if not self._is_plan_schema_valid(candidate, meta_kg):
                        report = self._validate_plan_schema_refs(candidate, meta_kg)
                        last_error = (
                            "schema_guard_reject: "
                            f"unknown_labels={report.get('unknown_labels')} "
                            f"unknown_relationships={report.get('unknown_relationships')} "
                            f"unknown_tables={report.get('unknown_tables')} "
                            f"unknown_columns={report.get('unknown_columns')} "
                            f"topology_violations={report.get('topology_violations')}"
                        )
                        self._react_print("recovery_candidate_rejected_by_schema_guard")
                        continue

                    try:
                        recovery_result = self.query_executor.execute_plan(candidate)
                    except Exception as exc:
                        recovery_result = {
                            "status": "failed",
                            "execution_mode": "single",
                            "db_type": str(candidate.get("db_type", "UNKNOWN")),
                            "query": str(candidate.get("query", "")),
                            "error": f"execute_plan_exception: {exc}",
                        }

                    if recovery_result.get("status") == "success":
                        candidate_payload = self._extract_execution_payload(recovery_result)
                        if self._is_insufficient_payload(candidate_payload):
                            last_error = "insufficient_result_payload"
                            self._react_print("recovery_execution_success_but_insufficient_payload")
                            continue

                        selection = self._build_selection_from_plan(candidate, scored_plans.get("plans", []))
                        execution_result = recovery_result
                        self._react_print("recovery_succeeded_replaced_selection")
                        break

                    last_error = self._extract_execution_error(recovery_result)
                    self._react_print("recovery_execution_failed")

        generated_natural_answer = self._generate_natural_answer(
            user_query=user_query,
            selected_plan=selection.get("selected_plan", {}),
            execution_result=execution_result,
        )

        # Additional answer-based fallback recovery.
        if (
            self.enable_recovery
            and self.recovery_policy != "never"
            and self._is_insufficient_answer(generated_natural_answer)
            and recovery_attempts < self.max_recovery_attempts
        ):
            recovery_invoked = True
            if recovery_usage_started_at is None:
                recovery_usage_started_at = self._safe_llm_snapshot(self.llm)
            self._react_print("answer_based_recovery_triggered")

            while recovery_attempts < self.max_recovery_attempts:
                recovery_attempts += 1
                fallback_candidates = [
                    p for p in scored_plans.get("plans", []) if str(p.get("plan_id")) not in tried_plan_ids
                ]
                if not fallback_candidates:
                    self._react_print("answer_based_recovery_no_fallback_candidates")
                    break

                candidate = min(
                    fallback_candidates,
                    key=lambda p: float(p.get("risk_aware_caqa_score", p.get("caqa_score", 1e18))),
                )
                tried_plan_ids.add(str(candidate.get("plan_id")))
                recovery_selected = candidate

                if not self._is_plan_schema_valid(candidate, meta_kg):
                    self._react_print("answer_based_recovery_schema_reject")
                    continue

                try:
                    candidate_exec = self.query_executor.execute_plan(candidate)
                except Exception:
                    continue

                if candidate_exec.get("status") != "success":
                    self._react_print("answer_based_recovery_exec_failed")
                    continue
                if self._is_insufficient_payload(self._extract_execution_payload(candidate_exec)):
                    self._react_print("answer_based_recovery_insufficient_payload")
                    continue

                candidate_answer = self._generate_natural_answer(
                    user_query=user_query,
                    selected_plan=candidate,
                    execution_result=candidate_exec,
                )
                if self._is_insufficient_answer(candidate_answer):
                    self._react_print("answer_based_recovery_insufficient_answer")
                    continue

                selection = self._build_selection_from_plan(candidate, scored_plans.get("plans", []))
                execution_result = candidate_exec
                recovery_result = candidate_exec
                generated_natural_answer = candidate_answer
                self._react_print("answer_based_recovery_succeeded")
                break

        recovery_usage_delta = self._safe_llm_delta(
            self.llm,
            self._safe_llm_snapshot(self.llm),
            recovery_usage_started_at,
        )

        state["selection"] = selection
        state["execution_result"] = execution_result
        state["execution_result_payload"] = self._extract_execution_payload(execution_result)
        state["generated_natural_answer"] = generated_natural_answer
        state["recovery_invoked"] = recovery_invoked
        state["recovery_attempts"] = recovery_attempts
        state["recovery_selected_plan"] = recovery_selected
        state["recovery_result"] = recovery_result
        state["recovery_token_usage"] = recovery_usage_delta
        state["recovery_prompt_tokens"] = int(recovery_usage_delta.get("prompt_tokens", 0.0))
        state["recovery_completion_tokens"] = int(recovery_usage_delta.get("completion_tokens", 0.0))
        state["recovery_total_tokens"] = int(recovery_usage_delta.get("total_tokens", 0.0))
        state["recovery_llm_calls"] = int(recovery_usage_delta.get("llm_calls", 0.0))
        state["recovery_llm_latency_sec"] = float(recovery_usage_delta.get("llm_latency_sec", 0.0))
        state["recovery_done"] = True

        return {
            "status": "ok",
            "recovery_invoked": recovery_invoked,
            "recovery_attempts": recovery_attempts,
            "execution_status": execution_result.get("status"),
        }

    def tool_answer_generation(self, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("generated_natural_answer"):
            generated_natural_answer = self._generate_natural_answer(
                user_query=str(state["user_query"]),
                selected_plan=state.get("selection", {}).get("selected_plan", {}),
                execution_result=state.get("execution_result", {}),
            )
            state["generated_natural_answer"] = generated_natural_answer

        self._react_print(
            f"final_answer={self._shorten(state.get('generated_natural_answer', ''), max_len=220)}"
        )
        return {
            "status": "ok",
            "answer_len": len(str(state.get("generated_natural_answer", ""))),
        }

    def run_react(
        self,
        user_query: str,
        table_names: Optional[List[str]] = None,
        decomposed_info: str = "",
        actual_cardinalities: Optional[Dict[str, int]] = None,
        predicted_risks: Optional[Dict[str, float]] = None,
        force_refresh_meta_kg: bool = False,
    ) -> Dict[str, Any]:
        if not user_query or not user_query.strip():
            raise ValueError("user_query must not be empty.")

        run_started = time.time()
        self._react_print(f"run_start query={self._shorten(user_query, max_len=120)}")
        self._debug_print("=== ReAct Agent Run Started ===")
        self._debug_print(
            "config: "
            f"risk_lambda={self.risk_lambda}, "
            f"normalize_expected_cost={self.normalize_expected_cost}, "
            f"enable_recovery={self.enable_recovery}, "
            f"recovery_policy={self.recovery_policy}, "
            f"recovery_threshold={self.recovery_threshold}, "
            f"max_recovery_attempts={self.max_recovery_attempts}, "
            f"max_react_iterations={self.max_react_iterations}"
        )

        state: Dict[str, Any] = {
            "user_query": user_query,
            "table_names": table_names,
            "decomposed_info": decomposed_info,
            "actual_cardinalities": actual_cardinalities,
            "predicted_risks": predicted_risks,
            "force_refresh_meta_kg": force_refresh_meta_kg,
        }
        trace: List[Dict[str, Any]] = []

        for iteration in range(1, self.max_react_iterations + 1):
            tool_name = self._next_tool(state)
            thought = self._think(iteration, tool_name, state)
            self._debug_print(f"[react][{iteration}] thought: {thought}")
            self._react_print(f"iter={iteration} next_tool={tool_name}")

            if tool_name == "finish":
                trace.append(
                    {
                        "iteration": iteration,
                        "thought": thought,
                        "action": "finish",
                        "observation": {"status": "done"},
                    }
                )
                break

            params: Dict[str, Any] = {}
            if tool_name == "meta_kg":
                params["force_refresh_meta_kg"] = force_refresh_meta_kg
            if tool_name == "planner":
                params["decomposed_info"] = decomposed_info
            if tool_name == "raqa":
                params["actual_cardinalities"] = actual_cardinalities
                params["predicted_risks"] = predicted_risks

            started = time.time()
            observation = self._act(tool_name, state, params=params)
            elapsed = round(time.time() - started, 3)
            observed = self._observe(tool_name, observation)
            observed["elapsed_sec"] = elapsed
            self._debug_print(f"[react][{iteration}] observe: {observed}")
            self._react_print(f"iter={iteration} done tool={tool_name} elapsed={elapsed}s status={observed.get('status')}")

            trace.append(
                {
                    "iteration": iteration,
                    "thought": thought,
                    "action": tool_name,
                    "action_input": params,
                    "observation": observed,
                }
            )
        else:
            self._debug_print("react loop reached max iterations; forcing answer_generation.")
            if "generated_natural_answer" not in state:
                self.tool_answer_generation(state, {})

        self._debug_print(f"run elapsed: {round(time.time() - run_started, 3)} sec")
        self._debug_print("=== ReAct Agent Run Finished ===")
        self._react_print(
            f"final_answer={self._shorten(state.get('generated_natural_answer', ''), max_len=220)}"
        )
        self._react_print(
            "run_end "
            f"elapsed={round(time.time() - run_started, 3)}s, "
            f"selected_plan_id={state.get('selection', {}).get('selected_plan_id') if isinstance(state.get('selection'), dict) else None}, "
            f"exec_status={state.get('execution_result', {}).get('status') if isinstance(state.get('execution_result'), dict) else None}"
        )

        return {
            "user_query": user_query,
            "meta_kg": state.get("meta_kg", {}),
            "schema_context": state.get("schema_context", ""),
            "schema_injection_mode": state.get("schema_injection_mode", "summary_only"),
            "schema_injected_entities": state.get("schema_injected_entities", []),
            "candidate_plans": state.get("candidate_plans", {"plans": []}),
            "scored_plans": state.get("scored_plans", {"plans": []}),
            "selected_plan": state.get("selection"),
            "execution_result": state.get("execution_result", {}),
            "execution_result_payload": state.get("execution_result_payload"),
            "generated_natural_answer": state.get("generated_natural_answer", ""),
            "recovery_invoked": bool(state.get("recovery_invoked", False)),
            "recovery_attempts": int(state.get("recovery_attempts", 0)),
            "recovery_selected_plan": state.get("recovery_selected_plan"),
            "recovery_result": state.get("recovery_result"),
            "recovery_token_usage": state.get("recovery_token_usage", {}),
            "recovery_prompt_tokens": int(state.get("recovery_prompt_tokens", 0)),
            "recovery_completion_tokens": int(state.get("recovery_completion_tokens", 0)),
            "recovery_total_tokens": int(state.get("recovery_total_tokens", 0)),
            "recovery_llm_calls": int(state.get("recovery_llm_calls", 0)),
            "recovery_llm_latency_sec": float(state.get("recovery_llm_latency_sec", 0.0)),
            "react_trace": trace,
            "react_iterations_used": len(trace),
            "react_max_iterations": self.max_react_iterations,
        }

    def run(
        self,
        user_query: str,
        table_names: Optional[List[str]] = None,
        decomposed_info: str = "",
        actual_cardinalities: Optional[Dict[str, int]] = None,
        predicted_risks: Optional[Dict[str, float]] = None,
        force_refresh_meta_kg: bool = False,
    ) -> Dict[str, Any]:
        return self.run_react(
            user_query=user_query,
            table_names=table_names,
            decomposed_info=decomposed_info,
            actual_cardinalities=actual_cardinalities,
            predicted_risks=predicted_risks,
            force_refresh_meta_kg=force_refresh_meta_kg,
        )
