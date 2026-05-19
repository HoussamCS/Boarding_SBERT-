import uvicorn

if __name__ == "__main__":
    # Start the FastAPI server on port 8000 with auto-reload enabled
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
