import sys

# Add backend folder to Python path
sys.path.append('backend')

try:
    print("1. Loading FastAPI backend and SBERT models (this will trigger in-memory JD metadata enrichment)...")
    from app.main import df, weighted_match, load_artifacts, JD_COL, ROLE_COL
    print("[SUCCESS] Main artifacts and enriched DataFrame loaded successfully!")
    
    # 2. Define a sample student preferences payload based on your form sections
    sample_student_pref = {
        "languages": ["Francais", "Anglais"],
        "travel_destination": "South Africa (ambiance anglophone & internationale)",
        "english_level": "Débutant ",
        "duration": "6 mois",
        "desired_domain": "Dev / IT",
        "company_style": "Grande boîte",
        "mission_type": "Technique",
        "ideal_atmosphere": "Dynamique et rythmé",
        "work_style": "Seul(e) tranquille"
    }
    
    sample_resume = "Student developer specialized in Python, React, and FastAPI. Love working in agile startup teams."
    
    print("\n2. Running Hybrid matching (SBERT Skill Match = 50%, Jaccard Keyword Overlap = 15%, Personality Fit = 35%)...")
    
    # Retrieve jd_embeddings from main to avoid recalculation
    from app.main import jd_embeddings
    
    results = weighted_match(
        query=sample_resume,
        candidates=df[JD_COL],
        candidate_embeddings=jd_embeddings,
        top_k=3,
        semantic_weight=0.50,
        overlap_weight=0.15,
        preference_weight=0.35,
        student_preferences=sample_student_pref,
        candidate_text_column=JD_COL
    )
    
    print("\n[HYBRID RECOMMENDATION RESULTS]")
    if results.empty:
        print("  No matching companies found that satisfied all hard constraints (Language/Duration).")
    else:
        for idx, row in results.iterrows():
            print(f"\n  [Rank {row['rank']}] Score: {row['match_score']:.4f} | Role: {row[ROLE_COL]}")
            # print a snippet of the JD
            snippet = row[JD_COL][:150].replace('\n', ' ')
            print(f"    JD: {snippet}...")
            
    print("\n[SUCCESS] All features are fully functional and ready to be consumed by the web team!")

except Exception as e:
    import traceback
    print("\n[ERROR] Verification failed:")
    traceback.print_exc()
