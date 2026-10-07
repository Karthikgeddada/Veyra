import os
import zipfile
import shutil

async def packager_agent(state: dict):
    run_id = state.get("run_id", "default_run")
    base_dir = f"generated_projects/{run_id}"
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Collect all files
    all_files = []
    all_files.extend(state.get("code_files", []))
    all_files.extend(state.get("tests_files", []))
    all_files.extend(state.get("docs_files", []))
    
    for f in all_files:
        path = f.get("path")
        content = f.get("content")
        if not path or content is None:
            continue
            
        full_path = os.path.join(base_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(content)
            
    # Create ZIP
    zip_filename = f"generated_projects/{run_id}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)
