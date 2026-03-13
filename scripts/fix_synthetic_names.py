import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import connect_db, get_db, close_db
from app.services.embedding import get_chroma_collection

async def fix_synthetic_names():
    print("Connecting to database...")
    await connect_db()
    db = get_db()
    chroma_collection = get_chroma_collection()
    
    # Get all candidates
    cursor = db.candidates.find({"source_filename": {"$exists": True}})
    candidates = await cursor.to_list(length=1000)
    print(f"Found {len(candidates)} candidates.")
    
    updated_count = 0
    
    for candidate in candidates:
        candidate_id = str(candidate["_id"])
        original_name = candidate["candidate_name"]
        filename = candidate.get("source_filename", "")
        
        target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synthetic_resumes")
        # The synthetic resumes are formatted as "First_Last_Resume.pdf"
        # Let's extract the actual name from the filename.
        if filename.endswith("_Resume.pdf"):
            real_name = filename.replace("_Resume.pdf", "").replace("_", " ")
        else:
            continue
            
        if real_name != original_name:
            print(f"Fixing name: '{original_name}' -> '{real_name}'")
            
            # Update MongoDB
            await db.candidates.update_one(
                {"_id": candidate["_id"]},
                {"$set": {"candidate_name": real_name}}
            )
            
            # Update ChromaDB Metadata
            try:
                chroma_collection.update(
                    ids=[candidate_id],
                    metadatas=[{
                        "candidate_name": real_name,
                        "email": candidate["email"]
                    }]
                )
            except Exception as e:
                print(f"Warning: Failed to update ChromaDB for {candidate_id}: {e}")
                
            updated_count += 1
            
    await close_db()
    print(f"Successfully fixed {updated_count} candidate names from the filename!")

if __name__ == "__main__":
    asyncio.run(fix_synthetic_names())
