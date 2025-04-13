import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import uuid
import uvicorn

from fastapi import FastAPI, HTTPException, Body, Path, Query as FastQuery # Alias Query
from pydantic import BaseModel, Field, validator, HttpUrl
from slugify import slugify # For creating slugs
from dotenv import load_dotenv

# --- LangChain & Vector Store Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import chromadb # Import ChromaDB client library directly for client setup

# --- Pydantic Models (Matching Prisma Schema) ---
from enum import Enum

class ArticleStatus(str, Enum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"

# Base model for tags (simple example)
class TagInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)

# Model for creating an article (input)
class ArticleInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=255)
    content: str = Field(..., min_length=50)
    excerpt: Optional[str] = Field(None, max_length=500)
    featuredImage: Optional[HttpUrl] = Field(None, description="URL to the main image") # Use HttpUrl for validation
    status: ArticleStatus = ArticleStatus.DRAFT
    publishedAt: Optional[datetime] = None
    authorId: str = Field(..., description="ID of the User who wrote the article")
    categoryId: str = Field(..., description="ID of the Category")
    tags: List[str] = Field([], description="List of tag names associated with the article") # Store tags as list of strings in metadata

    @validator('publishedAt', pre=True, always=True)
    def set_published_at_on_publish(cls, v, values):
        # Automatically set publishedAt if status is PUBLISHED and publishedAt is not set
        if values.get('status') == ArticleStatus.PUBLISHED and v is None:
            return datetime.now(timezone.utc)
        # Ensure datetime is timezone-aware (UTC) if provided
        if v and v.tzinfo is None:
             # You might want to decide on a default timezone policy
             # Here, we assume naive datetimes are UTC
             return v.replace(tzinfo=timezone.utc)
        return v

# Model representing a stored/retrieved article (includes generated fields)
class ArticleStored(ArticleInput):
    id: str = Field(..., description="Unique article ID (CUID-like, here UUID)")
    slug: str = Field(..., description="URL-friendly identifier")
    createdAt: datetime = Field(...)
    updatedAt: datetime = Field(...)

    class Config:
        from_attributes = True # Allows mapping from ORM objects (though we use dicts here)
        json_encoders = {
            datetime: lambda v: v.isoformat(), # Ensure ISO format for JSON output
        }


# Model for search queries
class SearchQuery(BaseModel):
    query_text: str = Field(..., description="The text query for similarity search.")
    top_k: int = Field(5, ge=1, le=50, description="Number of similar articles to return.")
    # Metadata filters (examples, expand as needed)
    filter_author_id: Optional[str] = Field(None, description="Filter results by exact author ID.")
    filter_category_id: Optional[str] = Field(None, description="Filter results by exact category ID.")
    filter_status: Optional[ArticleStatus] = Field(None, description="Filter results by article status.")
    filter_tags_all: Optional[List[str]] = Field(None, description="Filter results that must contain ALL of these tags.")
    filter_tags_any: Optional[List[str]] = Field(None, description="Filter results that must contain AT LEAST ONE of these tags.")
    filter_published_after: Optional[datetime] = Field(None, description="Filter results published after this date (ISO format).")
    filter_published_before: Optional[datetime] = Field(None, description="Filter results published before this date (ISO format).")


# Model for search results
class SearchResult(BaseModel):
    article: ArticleStored
    similarity_score: float = Field(..., description="Similarity score (lower is better for distance metrics like L2/Cosine)")


# --- Configuration & Initialization ---
load_dotenv() # Load environment variables from .env file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ollama and ChromaDB Setup ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "news_articles")

if not OLLAMA_EMBEDDING_MODEL:
    logger.error("OLLAMA_EMBEDDING_MODEL environment variable not set. Please set it to your desired Ollama embedding model.")
    raise ValueError("OLLAMA_EMBEDDING_MODEL is required.")

logger.info(f"Using Ollama server at: {OLLAMA_BASE_URL}")
logger.info(f"Using Ollama embedding model: {OLLAMA_EMBEDDING_MODEL}")
logger.info(f"ChromaDB path: {CHROMA_DB_PATH}")
logger.info(f"ChromaDB collection: {CHROMA_COLLECTION_NAME}")

try:
    # Initialize Ollama Embeddings
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        # Consider adding other Ollama parameters if needed, e.g., keep_alive
    )
    logger.info("Ollama Embeddings initialized successfully.")

    # Initialize ChromaDB Client
    # Use PersistentClient to save data to disk
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    logger.info(f"ChromaDB Persistent Client initialized at path: {CHROMA_DB_PATH}")

    # Get or Create Chroma Collection using the LangChain wrapper for compatibility
    # LangChain's Chroma wrapper handles passing the embedding function automatically
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        # Optional: You can specify metadata for the collection if needed on creation
        # collection_metadata={"hnsw:space": "cosine"} # e.g., set distance metric
    )
    logger.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' accessed/created.")

except Exception as e:
    logger.error(f"Failed to initialize embeddings or ChromaDB: {e}", exc_info=True)
    # Depending on the error, you might want to exit or handle it gracefully
    raise RuntimeError(f"Initialization failed: {e}")


# --- FastAPI Application ---
app = FastAPI(
    title="News Article Vector Search API",
    description=f"API to add and search news articles using LangChain, ChromaDB, and Ollama ({OLLAMA_EMBEDDING_MODEL}).",
    version="1.0.0",
)

# --- Helper Functions ---
def create_article_document_text(article: ArticleInput) -> str:
    """Combines relevant text fields for embedding."""
    parts = [
        f"Title: {article.title}",
        f"Excerpt: {article.excerpt}" if article.excerpt else "",
        f"Content: {article.content}",
        # Optional: Include tags or category name if they add semantic value for search
        # f"Tags: {', '.join(article.tags)}" if article.tags else "",
    ]
    return "\n\n".join(filter(None, parts)) # Join non-empty parts

def article_to_metadata(article: ArticleStored) -> Dict[str, Any]:
    """Converts the full ArticleStored object into a ChromaDB-compatible metadata dictionary."""
    metadata = article.dict()
    # Convert complex types to Chroma-compatible types (string, int, float, bool)
    for key, value in metadata.items():
        if isinstance(value, datetime):
            metadata[key] = value.isoformat() # Store dates as ISO strings
        elif isinstance(value, Enum):
            metadata[key] = value.value # Store enum value as string
        elif isinstance(value, list):
             # Chroma metadata values can be lists of strings/numbers/bools
             # Ensure tags are strings (they already are in our model)
            if key == "tags":
                metadata[key] = value # Keep as list of strings
            else: # Handle other potential list types if necessary
                 metadata[key] = [str(item) for item in value]
        elif isinstance(value, HttpUrl):
             metadata[key] = str(value) # Store URL as string
        elif value is None:
             # ChromaDB might handle None, but explicit conversion avoids issues
             # Or you might choose to omit None values from metadata
             pass # Let's omit None for simplicity, or handle as needed

    # Filter out None values explicitly if desired (Chroma usually handles them)
    return {k: v for k, v in metadata.items() if v is not None}


# --- API Endpoints ---

@app.post("/articles/",
          response_model=ArticleStored,
          status_code=201,
          summary="Add a New Article",
          tags=["Articles"])
async def add_article(article_input: ArticleInput):
    """
    Adds a new news article to the ChromaDB vector store.
    Generates embeddings for the text content and stores other fields as metadata.
    """
    logger.info(f"Received request to add article: {article_input.title}")

    # 1. Generate ID and Slug, set timestamps
    article_id = str(uuid.uuid4()) # Using UUID for simplicity instead of CUID
    slug_text = slugify(article_input.title)
    now = datetime.now(timezone.utc)

    # Basic check for slug uniqueness (in a real app, query DB first)
    # For simplicity here, we might append part of the ID if needed, but skipping for now.
    slug = f"{slug_text}-{article_id[:4]}" # Add part of ID for basic uniqueness

    stored_article_data = article_input.dict()
    stored_article_data.update({
        "id": article_id,
        "slug": slug,
        "createdAt": now,
        "updatedAt": now,
        # Ensure publishedAt is set correctly based on status
        "publishedAt": article_input.publishedAt if article_input.status != ArticleStatus.PUBLISHED or article_input.publishedAt else now
    })
    # Create the full ArticleStored object for metadata generation
    article_stored = ArticleStored(**stored_article_data)


    # 2. Prepare text for embedding and metadata dictionary
    document_text = create_article_document_text(article_input)
    metadata = article_to_metadata(article_stored)

    # Ensure essential metadata for retrieval is present
    if 'id' not in metadata or metadata['id'] != article_id:
        metadata['id'] = article_id # Make sure ID is correctly in metadata

    logger.debug(f"Document text for embedding (first 100 chars): {document_text[:100]}...")
    logger.debug(f"Metadata being stored: {metadata}")


    # 3. Add to ChromaDB using LangChain wrapper
    try:
        vectorstore.add_texts(
            texts=[document_text],
            metadatas=[metadata],
            ids=[article_id] # Use the generated article ID as the ChromaDB document ID
        )
        logger.info(f"Article '{article_input.title}' (ID: {article_id}) added to ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"Failed to add article ID {article_id} to ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store article in vector database: {e}")

    return article_stored # Return the fully formed article data


@app.post("/articles/search/",
          response_model=List[SearchResult],
          summary="Search Articles",
          tags=["Articles"])
async def search_articles(search_query: SearchQuery):
    """
    Performs similarity search on article content using the query text.
    Optionally filters results based on provided metadata criteria (author, category, status, tags, dates).
    """
    logger.info(f"Received search request: query='{search_query.query_text}', top_k={search_query.top_k}, filters={search_query.dict(exclude={'query_text', 'top_k'}, exclude_none=True)}")

    # 1. Construct ChromaDB 'where' filter from search query filters
    where_filter = {}
    filters_applied = []

    if search_query.filter_author_id:
        where_filter["authorId"] = search_query.filter_author_id
        filters_applied.append(f"authorId={search_query.filter_author_id}")
    if search_query.filter_category_id:
        where_filter["categoryId"] = search_query.filter_category_id
        filters_applied.append(f"categoryId={search_query.filter_category_id}")
    if search_query.filter_status:
        where_filter["status"] = search_query.filter_status.value
        filters_applied.append(f"status={search_query.filter_status.value}")

    # Tag filtering - Example using $and for 'all' and potentially $or for 'any'
    # Note: ChromaDB's $contains operator works on list elements for exact matches.
    # For 'any', you might need multiple queries or more complex logic if $or isn't directly supported in the desired way.
    # Let's implement 'all' using $and if multiple tags provided.
    if search_query.filter_tags_all:
         # Chroma's basic where doesn't directly support list containment ($contains) in complex $and/$or
         # A common workaround is multiple 'where' clauses if supported, or filtering *after* retrieval.
         # Let's try simple equality for a single tag filter_tags_all=[tag]
         # If multiple tags, complex filtering might be needed post-retrieval or via Chroma advanced features.
         if len(search_query.filter_tags_all) == 1:
              # This is approximate - really needs $contains or similar
              where_filter["tags"] = search_query.filter_tags_all[0]
              filters_applied.append(f"tags CONTAINS (approx) {search_query.filter_tags_all[0]}")
         else:
              # For multiple tags 'all', we might need $and logic if supported, or filter later
              logger.warning("Filtering by ALL multiple tags is complex with basic Chroma 'where'; filtering first tag only or consider post-filtering.")
              where_filter["tags"] = search_query.filter_tags_all[0] # Example: filter by first tag only
              filters_applied.append(f"tags CONTAINS (approx) {search_query.filter_tags_all[0]}")


    # Date filtering - Requires numeric comparison ($gte, $lte)
    # Chroma metadata allows ISO strings; comparisons might work directly or need timestamp conversion depending on version/setup.
    date_filters = {}
    if search_query.filter_published_after:
        # Assuming publishedAt is stored as ISO string
        date_filters["$gte"] = search_query.filter_published_after.isoformat()
        filters_applied.append(f"publishedAfter={search_query.filter_published_after.isoformat()}")
    if search_query.filter_published_before:
        date_filters["$lte"] = search_query.filter_published_before.isoformat()
        filters_applied.append(f"publishedBefore={search_query.filter_published_before.isoformat()}")

    if date_filters:
        if "publishedAt" in where_filter: # Check if key already exists from another filter (unlikely here)
            if isinstance(where_filter["publishedAt"], dict):
                 where_filter["publishedAt"].update(date_filters)
            else: # If simple equality was set, this logic needs refinement
                 logger.error("Conflicting filters on publishedAt")
                 raise HTTPException(status_code=400, detail="Cannot apply range filter and equality filter on publishedAt simultaneously.")
        else:
            where_filter["publishedAt"] = date_filters


    # If where_filter is empty, set it to None for the query method
    final_where = where_filter if where_filter else None
    logger.info(f"Executing ChromaDB search with where filter: {final_where}, filters applied: {', '.join(filters_applied) if filters_applied else 'None'}")


    # 2. Perform Similarity Search with Filters using LangChain wrapper
    try:
        # Use similarity_search_with_score to get scores (distances)
        results_with_scores = await vectorstore.asimilarity_search_with_score(
            query=search_query.query_text,
            k=search_query.top_k,
            filter=final_where # Pass the constructed filter dictionary
            # `filter` is the newer kwarg name replacing `where` in some Langchain/Chroma versions
            # Use `where=final_where` if `filter` doesn't work with your version.
        )
        logger.info(f"Found {len(results_with_scores)} results matching query and filters.")

    except Exception as e:
        logger.error(f"Error during ChromaDB similarity search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to perform search in vector database: {e}")

    # 3. Process Results
    final_results: List[SearchResult] = []
    if results_with_scores:
        for doc, score in results_with_scores:
            try:
                # The metadata stored should match ArticleStored structure
                # We need to handle potential type mismatches during reconstruction (e.g., datetime strings)
                metadata = doc.metadata

                # Re-parse dates from ISO strings stored in metadata
                if 'publishedAt' in metadata and isinstance(metadata['publishedAt'], str):
                    metadata['publishedAt'] = datetime.fromisoformat(metadata['publishedAt'].replace('Z', '+00:00')) # Handle Z timezone if present
                if 'createdAt' in metadata and isinstance(metadata['createdAt'], str):
                    metadata['createdAt'] = datetime.fromisoformat(metadata['createdAt'].replace('Z', '+00:00'))
                if 'updatedAt' in metadata and isinstance(metadata['updatedAt'], str):
                    metadata['updatedAt'] = datetime.fromisoformat(metadata['updatedAt'].replace('Z', '+00:00'))

                # Reconstruct the ArticleStored object from metadata
                # Ensure all required fields for ArticleStored are present in metadata
                required_fields = ArticleStored.__fields__.keys()
                if all(field in metadata for field in required_fields):
                     article_data = ArticleStored(**metadata)
                     final_results.append(SearchResult(article=article_data, similarity_score=score))
                else:
                     missing = [field for field in required_fields if field not in metadata]
                     logger.warning(f"Skipping result with ID {metadata.get('id', 'N/A')} due to missing metadata fields: {missing}. Metadata: {metadata}")

            except Exception as e:
                 # Log issue with reconstructing a specific article but continue with others
                 logger.error(f"Error processing search result item (metadata: {doc.metadata}): {e}", exc_info=True)

    return final_results


@app.get("/articles/{article_id}",
         response_model=ArticleStored,
         summary="Get Article by ID",
         tags=["Articles"])
async def get_article(article_id: str = Path(..., description="The ID of the article to retrieve.")):
    """
    Retrieves a single article from ChromaDB by its unique ID.
    """
    logger.info(f"Received request to get article by ID: {article_id}")
    try:
        # Use Chroma client's get method for direct ID lookup
        result = vectorstore.get(ids=[article_id], include=["metadatas"]) # include=['metadatas', 'documents']

        if not result or not result.get('ids') or article_id not in result['ids']:
            logger.warning(f"Article with ID {article_id} not found in ChromaDB.")
            raise HTTPException(status_code=404, detail=f"Article with ID {article_id} not found.")

        # Extract metadata for the found article
        metadata_list = result.get('metadatas', [])
        if not metadata_list:
             logger.error(f"Article ID {article_id} found, but metadata is missing.")
             raise HTTPException(status_code=500, detail="Article found but data is inconsistent.")

        metadata = metadata_list[0] # Get the metadata dict for the first (only) ID

        # Reconstruct the ArticleStored object (similar to search result processing)
        try:
            if 'publishedAt' in metadata and isinstance(metadata['publishedAt'], str):
                 metadata['publishedAt'] = datetime.fromisoformat(metadata['publishedAt'].replace('Z', '+00:00'))
            if 'createdAt' in metadata and isinstance(metadata['createdAt'], str):
                metadata['createdAt'] = datetime.fromisoformat(metadata['createdAt'].replace('Z', '+00:00'))
            if 'updatedAt' in metadata and isinstance(metadata['updatedAt'], str):
                metadata['updatedAt'] = datetime.fromisoformat(metadata['updatedAt'].replace('Z', '+00:00'))

            article_data = ArticleStored(**metadata)
            return article_data
        except Exception as e:
             logger.error(f"Error reconstructing article ID {article_id} from metadata: {e}. Metadata: {metadata}", exc_info=True)
             raise HTTPException(status_code=500, detail="Failed to reconstruct article data from storage.")

    except HTTPException as http_exc:
         raise http_exc # Re-raise FastAPI exceptions
    except Exception as e:
        logger.error(f"Error retrieving article ID {article_id} from ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve article: {e}")


# --- Run the app (for direct execution) ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001)) # Use a different port than the summarizer maybe
    host = os.getenv("HOST", "127.0.0.1")
    logger.info(f"Starting Uvicorn server on {host}:{port}")
    # Use --reload for development, but not typically in __main__ block for production
    uvicorn.run("main:app", host=host, port=port, reload=True) # Added reload=True for convenience