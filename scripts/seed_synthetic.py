import asyncio
import os
import sys

from pypdf import PdfReader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import connect_db, insert_candidate, close_db
from app.services.embedding import index_resume
from app.api.routes import _extract_name_from_resume, _extract_email_from_resume

async def seed_synthetic_resumes():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synthetic_resumes")
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} not found.")
        return
        
    pdf_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.pdf')])
    print(f"Found {len(pdf_files)} PDF resumes in {target_dir}. Connecting to database...")
    
    if not pdf_files:
        print("No PDFs found to process.")
        return
    
    await connect_db()
    count = 0
    
    for i, file_name in enumerate(pdf_files):
        file_path = os.path.join(target_dir, file_name)
        try:
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                print(f"Skipping {file_name}: No text extracted.")
                continue
                
            name = _extract_name_from_resume(text)
            if name == "Unknown Candidate":
                name = f"Candidate {file_name.replace('.pdf', '')}"
                
            email = _extract_email_from_resume(text)
            if not email:
                email = f"candidate_{count}@example.com"
                
            print(f"[{i+1}/{len(pdf_files)}] Indexing: {name} ({email}) from {file_name}")
            
            candidate_id = await insert_candidate({
                "candidate_name": name,
                "email": email,
                "resume_text": text,
                "source_filename": file_name,
            })
            
            index_resume(
                candidate_id=candidate_id,
                resume_text=text,
                metadata={"candidate_name": name, "email": email},
            )
            count += 1
            
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
            
    await close_db()
    print(f"Successfully seeded {count} synthetic resumes!")

if __name__ == "__main__":
    asyncio.run(seed_synthetic_resumes())
