from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from .services.preference_engine import enrich_jd_dataframe, calculate_preference_scores

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "boarding_sbert_model"
RESUME_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "resume_embeddings.npy"
JD_EMB_PATH = PROJECT_ROOT / "data" / "processed" / "jd_embeddings.npy"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "boarding_dataset_index.parquet"
CONFIG_PATH = PROJECT_ROOT / "models" / "matching_config.joblib"


TOP_K_DEFAULT = 5

app = FastAPI(title="Boarding Resume-JD Matching API")


class WeightedMatchRequest(BaseModel):
    text: str
    top_k: int = TOP_K_DEFAULT
    semantic_weight: float = 0.8
    overlap_weight: float = 0.2
    preference_weight: float = 0.0
    keyword_list: list[str] | None = None
    hard_filters: dict[str, Any] | None = None
    student_preferences: dict[str, Any] | None = None


class MatchResponse(BaseModel):
    rank: int
    match_score: float
    role: str
    text: str
    label: Any | None = None
    reason: Any | None = None


def clean_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def compute_text_overlap_score(text_a: str, text_b: str, keywords: list[str] | None = None) -> float:
    words_a = set(clean_text(text_a).split())
    words_b = set(clean_text(text_b).split())
    if keywords:
        keyword_set = {clean_text(k) for k in keywords if clean_text(k)}
        if not keyword_set:
            return 0.0
        overlap = words_a & words_b & keyword_set
        return float(len(overlap) / len(keyword_set))

    union = words_a | words_b
    if not union:
        return 0.0
    overlap = words_a & words_b
    return float(len(overlap) / len(union))


def apply_hard_filters(df_candidates: pd.DataFrame, hard_filters: dict[str, Any] | None) -> pd.DataFrame:
    if hard_filters is None:
        return df_candidates

    filtered = df_candidates.copy()
    for key, value in hard_filters.items():
        if key not in filtered.columns:
            continue
        if isinstance(value, (list, tuple, set)):
            filtered = filtered[filtered[key].isin(value)]
        else:
            filtered = filtered[filtered[key].astype(str).str.contains(str(value), case=False, na=False)]
    return filtered


def load_artifacts() -> tuple[SentenceTransformer, torch.Tensor, torch.Tensor, pd.DataFrame]:
    model = SentenceTransformer(str(MODEL_DIR))
    resume_embeddings = torch.from_numpy(np.load(RESUME_EMB_PATH))
    jd_embeddings = torch.from_numpy(np.load(JD_EMB_PATH))
    df = pd.read_parquet(DATA_PATH)
    df = enrich_jd_dataframe(df)
    return model, resume_embeddings, jd_embeddings, df


model, resume_embeddings, jd_embeddings, df = load_artifacts()

ROLE_COL = "Role"
RESUME_COL = "Resume"
JD_COL = "Job_Description"
LABEL_COL = "label"
REASON_COL = "Reason_for_decision"


def build_response_rows(results: pd.DataFrame, text_field: str) -> list[MatchResponse]:
    return [
        MatchResponse(
            rank=int(row["rank"]),
            match_score=float(row["match_score"]),
            role=row.get(ROLE_COL, ""),
            text=row.get(text_field, ""),
            label=row.get(LABEL_COL),
            reason=row.get(REASON_COL),
        )
        for _, row in results.iterrows()
    ]


def weighted_match(
    query: str,
    candidates: pd.Series,
    candidate_embeddings: torch.Tensor,
    top_k: int,
    semantic_weight: float,
    overlap_weight: float,
    preference_weight: float = 0.0,
    keyword_list: list[str] | None = None,
    hard_filters: dict[str, Any] | None = None,
    student_preferences: dict[str, Any] | None = None,
    candidate_text_column: str = JD_COL,
) -> pd.DataFrame:
    cleaned = clean_text(query)
    query_emb = model.encode(cleaned, convert_to_tensor=True)
    semantic_scores = util.cos_sim(query_emb, candidate_embeddings)[0].cpu().numpy()
                     
    total_weight = semantic_weight + overlap_weight + preference_weight
    if total_weight > 0:
        semantic_weight /= total_weight
        overlap_weight /= total_weight
        preference_weight /= total_weight

    scores = semantic_scores.copy()
    
    overlap_scores = np.zeros_like(scores)
    if overlap_weight > 0.0:
        overlap_scores = np.array(
            [compute_text_overlap_score(cleaned, candidate_text, keyword_list) for candidate_text in candidates.astype(str).tolist()]
        )

    pref_scores = np.zeros_like(scores)
    passes_personality_filters = pd.Series(True, index=df.index)
    
    if preference_weight > 0.0 and student_preferences and candidate_text_column == JD_COL:
        passes_personality_filters, pref_series = calculate_preference_scores(student_preferences, df)
        pref_scores = pref_series.to_numpy()

    combined_scores = (
        semantic_weight * semantic_scores +
        overlap_weight * overlap_scores +
        preference_weight * pref_scores
    )

    results = df.copy()
    results = results.assign(combined_score=combined_scores)

    if hard_filters is not None:
        results = apply_hard_filters(results, hard_filters)
        
    if preference_weight > 0.0 and student_preferences and candidate_text_column == JD_COL:
        results = results[passes_personality_filters.loc[results.index]]

    if results.empty:
        return pd.DataFrame(columns=["rank", "match_score", ROLE_COL, candidate_text_column, LABEL_COL, REASON_COL])
    results = results.loc[results.index].sort_values(by="combined_score", ascending=False).head(top_k)
    results = results[[ROLE_COL, candidate_text_column, LABEL_COL, REASON_COL, "combined_score"]].copy()
    results["match_score"] = (results["combined_score"] + 0.18).clip(upper=1.0).round(4)
    results["rank"] = range(1, len(results) + 1)
    results = results.reset_index(drop=True)
    return results[["rank", "match_score", ROLE_COL, candidate_text_column, LABEL_COL, REASON_COL]]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "model": str(MODEL_DIR.name)}


@app.post("/match/resume")
def match_resume(request: WeightedMatchRequest) -> dict[str, list[MatchResponse]]:
    results = weighted_match(
        query=request.text,
        candidates=df[JD_COL],
        candidate_embeddings=jd_embeddings,
        top_k=request.top_k,
        semantic_weight=request.semantic_weight,
        overlap_weight=request.overlap_weight,
        preference_weight=request.preference_weight,
        keyword_list=request.keyword_list,
        hard_filters=request.hard_filters,
        student_preferences=request.student_preferences,
        candidate_text_column=JD_COL,
    )
    return {"matches": build_response_rows(results, JD_COL)}


@app.post("/match/jd")
def match_jd(request: WeightedMatchRequest) -> dict[str, list[MatchResponse]]:
    results = weighted_match(
        query=request.text,
        candidates=df[RESUME_COL],
        candidate_embeddings=resume_embeddings,
        top_k=request.top_k,
        semantic_weight=request.semantic_weight,
        overlap_weight=request.overlap_weight,
        keyword_list=request.keyword_list,
        hard_filters=request.hard_filters,
        candidate_text_column=RESUME_COL,
    )
    return {"matches": build_response_rows(results, RESUME_COL)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
