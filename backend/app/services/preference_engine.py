import re
import pandas as pd
import numpy as np

def enrich_jd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans each Job Description dynamically to extract virtual metadata tags 
    (Location, English requirements, Duration, Company Size, Atmosphere, Mission Type)
    in less than 0.1 seconds, avoiding any hard disk parquet modifications.
    """
    enriched = df.copy()
    
    locations = []
    english_levels = []
    durations = []
    company_styles = []
    mission_types = []
    atmospheres = []
    
    # We will look at both 'job_description' or fallback columns in dataframe
    jd_col = "job_description" if "job_description" in df.columns else ("Job_Description" if "Job_Description" in df.columns else None)
    role_col = "role" if "role" in df.columns else ("Role" if "Role" in df.columns else None)
    
    for idx, row in enriched.iterrows():
        jd_text = str(row.get(jd_col, "")).lower() if jd_col else ""
        role_text = str(row.get(role_col, "")).lower() if role_col else ""
        
        # 1. Location extraction
        if any(kw in jd_text for kw in ["south africa", "afrique du sud", "cape town", "le cap", "sa "]):
            loc = "South Africa"
        elif any(kw in jd_text for kw in ["maroc", "morocco", "casablanca", "rabat", "tanger", "marrakech", "moroccan"]):
            loc = "Morocco"
        else:
            loc = "Open"
        locations.append(loc)
        
        # 2. English requirement level
        if any(kw in jd_text for kw in ["english fluent", "fluent english", "anglais courant", "anglais bilingue", "advanced english", "must speak english"]):
            eng = "Advanced"
        elif any(kw in jd_text for kw in ["anglais intermédiaire", "intermediate english", "good english", "notions d'anglais"]):
            eng = "Intermediate"
        else:
            # If South Africa is detected, English is implicitly at least Intermediate
            if loc == "South Africa":
                eng = "Intermediate"
            else:
                eng = "Beginner"
        english_levels.append(eng)
        
        # 3. Duration (in months)
        duration_match = re.search(r"(\d+)\s*(mois|months)", jd_text)
        if duration_match:
            dur = int(duration_match.group(1))
        else:
            # Fallback based on typical contract markers
            if any(kw in jd_text for kw in ["cdi", "temps plein", "permanent", "full-time"]):
                dur = 6
            else:
                dur = 3  # default standard internship
        durations.append(dur)
        
        # 4. Company Style & Vibe
        if any(kw in jd_text for kw in ["startup", "start-up", "jeune pousse", "creativity", "innovation", "agile"]):
            style = "Start-up cool"
            atmos = "Libre et flexible"
        elif any(kw in jd_text for kw in ["multinationale", "grand groupe", "corporate", "grande entreprise", "structured", "cabinet"]):
            style = "Grande boîte"
            atmos = "Calme et structuré"
        else:
            style = "Petite entreprise locale"
            atmos = "Dynamique et rythmé"
        company_styles.append(style)
        atmospheres.append(atmos)
        
        # 5. Mission Type
        # Check counts for technical keywords
        tech_score = sum(kw in jd_text for kw in ["python", "java", "sql", "code", "dev", "tech", "logiciel", "frontend", "backend", "it ", "ingénieur", "algorithme"])
        # Check counts for creative keywords
        creative_score = sum(kw in jd_text for kw in ["design", "communication", "marketing", "créati", "motion", "video", "redac", "social", "reels", "tiktok", "visual"])
        # Check counts for relational keywords
        rel_score = sum(kw in jd_text for kw in ["humain", "relationnel", "client", "support", "sales", "hr", "rh", "ventes", "community", "social", "partenaire"])
        
        scores = {
            "Technique": tech_score,
            "Créatif": creative_score,
            "Humain / relationnel": rel_score
        }
        
        max_type = max(scores, key=scores.get)
        if scores[max_type] == 0:
            mission_types.append("Un peu de tout, je suis open")
        else:
            mission_types.append(max_type)
            
    enriched["detected_location"] = locations
    enriched["detected_english"] = english_levels
    enriched["detected_duration"] = durations
    enriched["detected_company_style"] = company_styles
    enriched["detected_atmosphere"] = atmospheres
    enriched["detected_mission_type"] = mission_types
    
    return enriched

def calculate_preference_scores(student_pref: dict, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Takes student form answers and calculates:
    1. A boolean Series (passes_hard_filters) indicating if they pass language/availability constraints.
    2. A float Series (pref_scores) from 0.0 to 1.0 representing cultural compatibility.
    """
    passes_hard_filters = pd.Series(True, index=df.index)
    
    # --- HARD FILTERS ---
    
    # Destination constraint
    target_dest = student_pref.get("travel_destination")
    if target_dest == "🇲🇦 Maroc (ambiance francophone)":
        passes_hard_filters &= (df["detected_location"] == "Morocco") | (df["detected_location"] == "Open")
    elif target_dest == "🇿🇦 Afrique du Sud (ambiance anglophone & internationale)":
        passes_hard_filters &= (df["detected_location"] == "South Africa") | (df["detected_location"] == "Open")
        
    # English level vs Company Requirement
    student_eng = student_pref.get("english_level", "Débutant")
    eng_ranks = {"Débutant": 0, "Intermédiaire": 1, "Avancé": 2}
    jd_eng_ranks = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    
    student_rank = eng_ranks.get(student_eng, 0)
    for idx, row in df.iterrows():
        jd_req_rank = jd_eng_ranks.get(row.get("detected_english", "Beginner"), 0)
        if student_rank < jd_req_rank:
            passes_hard_filters.loc[idx] = False
            
    # Duration constraint
    student_duration = student_pref.get("duration") # "1 mois", "2-3 mois", "6 mois ou plus"
    if student_duration == "1 mois":
        passes_hard_filters &= df["detected_duration"] <= 2
    elif student_duration == "2-3 mois":
        passes_hard_filters &= df["detected_duration"] <= 4
    elif student_duration == "6 mois ou plus":
        passes_hard_filters &= df["detected_duration"] >= 5

    # --- SOFT PREFERENCES SCORE ---
    pref_scores = pd.Series(0.0, index=df.index)
    
    # Domain Mapping
    desired_domain = student_pref.get("desired_domain")
    domain_mapping = {
        "Marketing / Com’": "marketing",
        "Dév / IT": "dev",
        "Tourisme / Hôtellerie": "tourisme",
        "Éducation / Social": "social",
        "ONG / Humanitaire": "ong",
        "Finance / Gestion": "finance"
    }
    
    preferred_style = student_pref.get("company_style") # Startup, PME, Large
    preferred_mission = student_pref.get("mission_type") # Technical, Creative, Relational, Open
    preferred_atmos = student_pref.get("ideal_atmosphere") # Calm, Dynamic, Flexible
    work_style = student_pref.get("work_style") # Team vs Solo
    
    role_col = "role" if "role" in df.columns else ("Role" if "Role" in df.columns else None)
    jd_col = "job_description" if "job_description" in df.columns else ("Job_Description" if "Job_Description" in df.columns else None)
    
    for idx, row in df.iterrows():
        score = 0.0
        max_score = 0.0
        
        # 1. Domain match (+0.25)
        if desired_domain and desired_domain in domain_mapping:
            keyword = domain_mapping[desired_domain]
            role_text = str(row.get(role_col, "")).lower() if role_col else ""
            jd_text = str(row.get(jd_col, "")).lower() if jd_col else ""
            
            if keyword in role_text or keyword in jd_text:
                score += 0.25
            max_score += 0.25
            
        # 2. Company style match (+0.20)
        if preferred_style:
            if row.get("detected_company_style") == preferred_style:
                score += 0.20
            max_score += 0.20
            
        # 3. Mission type match (+0.20)
        if preferred_mission and preferred_mission != "Un peu de tout, je suis open":
            if row.get("detected_mission_type") == preferred_mission:
                score += 0.20
            max_score += 0.20
            
        # 4. Atmosphere match (+0.15)
        if preferred_atmos:
            if row.get("detected_atmosphere") == preferred_atmos:
                score += 0.15
            max_score += 0.15
            
        # 5. Work style team vs solo (+0.20)
        if work_style:
            jd_text = str(row.get(jd_col, "")).lower() if jd_col else ""
            requires_team = any(kw in jd_text for kw in ["team", "équipe", "collaborat", "partage", "coopér"])
            
            if work_style == "En équipe" and requires_team:
                score += 0.20
            elif work_style == "Seul(e) tranquille" and not requires_team:
                score += 0.20
            elif work_style == "Ça dépend de l’ambiance":
                score += 0.15  # partial compatibility
            else:
                score += 0.05  # slight mismatch
            max_score += 0.20
            
        pref_scores.loc[idx] = (score / max_score) if max_score > 0 else 0.5
        
    return passes_hard_filters, pref_scores
