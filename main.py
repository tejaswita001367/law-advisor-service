from fastapi import FastAPI, Query
from typing import Optional
from law_data import law_database

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running"}

@app.get("/search-law")
def search_law(query: str = Query(..., description="Enter your legal query like 'murder' or 'theft'")):
    keyword = query.lower()

    for law_keyword, law_info in law_database.items():
        if law_keyword in keyword:
            return {
                "keyword": law_keyword,
                "section": law_info["section"],
                "description": law_info["description"],
                "punishment": law_info["punishment"]
            }

    return {"message": "No matching law found for your query"}
