import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import connect_db, get_db, close_db
from app.services.embedding import get_chroma_collection

async def wipe_database():
    print("Connecting to database...")
    await connect_db()
    db = get_db()
    
    # Wipe MongoDB Candidates Collection
    print("Dropping MongoDB candidates collection...")
    await db.candidates.drop()
    
    # Wipe ChromaDB Collection
    print("Recreating ChromaDB collection...")
    chroma_collection = get_chroma_collection()
    
    # We can fetch all IDs and delete them to effectively wipe the collection
    try:
        all_docs = chroma_collection.get()
        ids = all_docs.get("ids", [])
        if ids:
            chroma_collection.delete(ids=ids)
            print(f"Deleted {len(ids)} embeddings from ChromaDB.")
        else:
            print("ChromaDB is already empty.")
    except Exception as e:
        print(f"Failed to wipe ChromaDB: {e}")

    await close_db()
    print("Database wiped successfully.")

if __name__ == "__main__":
    asyncio.run(wipe_database())
