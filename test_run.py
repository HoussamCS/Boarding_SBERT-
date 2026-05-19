import sys
import traceback

try:
    # Add backend folder to path
    sys.path.append('backend')
    
    print("Importing load_artifacts from app.main...")
    from app.main import load_artifacts
    
    print("Loading SBERT model, Parquet dataset, and NumPy embeddings...")
    model, resume, jd, df = load_artifacts()
    print("\n[SUCCESS] All artifacts loaded successfully!")
    print(f"  - Dataset Rows: {df.shape[0]}")
    print(f"  - Dataset Columns: {list(df.columns)}")
    print(f"  - Resume Embeddings shape: {resume.shape}")
    print(f"  - Job Description Embeddings shape: {jd.shape}")

except Exception as e:
    print("\n[ERROR] Error occurred during artifact loading:")
    traceback.print_exc()
