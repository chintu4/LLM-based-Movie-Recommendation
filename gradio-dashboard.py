import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import gradio as gr

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()

DATA_PATH = os.getenv("MOVIES_CSV", "movies_with_emotions.csv")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_movies_db")

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv(DATA_PATH)

# Poster_Url handling (keep your existing behavior, but be defensive)
if "Poster_Url" in df.columns:
    df["large_Poster_Url"] = df["Poster_Url"].astype(str) + "&fife=w800"
    df["large_Poster_Url"] = np.where(
        df["Poster_Url"].isna() | (df["Poster_Url"].astype(str).str.strip() == ""),
        "cover-not-found.jpg",
        df["large_Poster_Url"],
    )
else:
    df["large_Poster_Url"] = "cover-not-found.jpg"

# -----------------------
# Figure out key columns safely
# -----------------------
def pick_first(existing_cols, candidates):
    for c in candidates:
        if c in existing_cols:
            return c
    return None

# Print columns to make dataset mismatches obvious in hosted envs
print(f"Loaded CSV: {DATA_PATH}")
print("CSV columns:", list(df.columns))

id_col = pick_first(df.columns, ["movieId", "movie_id", "id", "tmdb_id", "imdb_id", "isbn13", "index"])

# Notebooks in this repo use Title (capital T), Overview, Genre
# so include those common variants too.
title_col = pick_first(df.columns, [
    "title",
    "Title",
    "movie_title",
    "name",
    "original_title",
    "primaryTitle",
    "title_and_subtitle",
])

desc_col = pick_first(df.columns, [
    "description",
    "plot",
    "overview",
    "Overview",
    "tagline",
    "summary",
    "Synopsis",
    "tagged_description",
])

genre_col = pick_first(df.columns, [
    "simple_categories",
    "simple_Genre",
    "genre",
    "genres",
    "Genre",
])

# Emotion columns (optional)
emotion_cols = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}
available_emotions = {k: v for k, v in emotion_cols.items() if v in df.columns}

# If no usable ID column, create one from row index
if id_col is None:
    df = df.reset_index().rename(columns={"index": "row_id"})
    id_col = "row_id"

if title_col is None:
    # Must have something to show in UI
    raise ValueError(
        f"Could not find a title column. CSV has columns: {list(df.columns)}"
    )

if desc_col is None:
    # We can still do semantic search, but results will be weaker; create an empty description
    df["description"] = ""
    desc_col = "description"

# -----------------------
# Build / load vector DB from the CSV itself
# (no tagged_description.txt needed)
# -----------------------
def build_documents(dataframe: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in dataframe.iterrows():
        movie_id = row.get(id_col)
        title = str(row.get(title_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip()

        # Add optional genre/category into the indexed text
        genre_val = str(row.get(genre_col, "")).strip() if genre_col else ""

        text = f"{title}\n{genre_val}\n{desc}".strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"movie_id": str(movie_id)},
            )
        )
    return docs

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

# Persist so you don't re-embed every run
if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    docs = build_documents(df)
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)

# -----------------------
# Recommender
# -----------------------
def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    if not query or not query.strip():
        return df.head(0)

    recs = db.similarity_search(query, k=initial_top_k)

    movie_ids = []
    for r in recs:
        mid = r.metadata.get("movie_id")
        if mid is not None:
            movie_ids.append(str(mid))

    # Filter matching rows
    # (convert both sides to string to avoid dtype mismatch)
    working = df[df[id_col].astype(str).isin(movie_ids)].copy()

    # Category filter (only if we have a category column)
    if category != "All" and genre_col is not None:
        working = working[working[genre_col].astype(str) == str(category)].copy()

    # Tone sorting (only if emotion columns exist)
    if tone in available_emotions:
        emotion_col = available_emotions[tone]
        # Use assignment (avoid inplace warnings)
        working = working.sort_values(by=emotion_col, ascending=False)

    return working.head(final_top_k)


def recommend_movies(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        title = str(row.get(title_col, "Untitled"))
        desc = str(row.get(desc_col, "") or "")
        desc_words = desc.split()
        truncated_description = " ".join(desc_words[:30]) + (
            "..." if len(desc_words) > 30 else ""
        )

        caption_parts = [title]
        if genre_col is not None:
            caption_parts.append(f"({row.get(genre_col, '')})")
        if truncated_description.strip():
            caption_parts.append(f": {truncated_description}")

        caption = " ".join([p for p in caption_parts if str(p).strip()])
        results.append((row.get("large_Poster_Url", "cover-not-found.jpg"), caption))

    return results


# -----------------------
# UI
# -----------------------
categories = ["All"]
if genre_col is not None:
    categories += sorted(
        [c for c in df[genre_col].dropna().astype(str).unique() if c.strip()]
    )

tones = ["All"] + list(available_emotions.keys())

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic movie recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe a movie you want:",
            placeholder="e.g., A suspenseful story about survival and betrayal",
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category/genre:",
            value="All",
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All",
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended movies", columns=4, rows=4)

    submit_button.click(
        fn=recommend_movies,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()
