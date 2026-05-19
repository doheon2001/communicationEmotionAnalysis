import os
import json
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
input_dir = r"c:\Users\user1\communicationEmotionAnalysis\data\raw\Selectstar_Tunip_HUMANELab_opendata"
output_file = r"c:\Users\user1\communicationEmotionAnalysis\data\raw\Selectstar_Tunip_HUMANELab_opendata\Selectstar_Tunip_HUMANELab_opendata_merged.json"

# Find all JSON files (exclude the merged output file itself if it already exists)
all_json_files = glob.glob(os.path.join(input_dir, "*.json"))
json_files = [f for f in all_json_files if os.path.basename(f) != os.path.basename(output_file)]

# Sort the files alphabetically (C000000.json, C000001.json, ... C199998.json)
# This guarantees the rows will be merged in the exact sequential order.
json_files.sort()

print(f"Found {len(json_files)} JSON files to merge. Starting merge...")

def process_file(file_path):
    """
    Reads a single JSON file with multi-stage encoding fallback.
    Returns (file_name, data_dict) or (file_name, None) on failure.
    """
    file_name = os.path.basename(file_path)
    
    # 1. Try reading with UTF-8-SIG (handles standard UTF-8 and UTF-8 with BOM)
    # strict=False allows raw control characters (like newlines/tabs) inside JSON strings
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f, strict=False)
            return file_name, data
    except (UnicodeDecodeError, json.JSONDecodeError) as e_utf8:
        # 2. Try falling back to CP949 (EUC-KR based Windows encoding)
        try:
            with open(file_path, 'r', encoding='cp949') as f:
                data = json.load(f, strict=False)
                return file_name, data
        except Exception as e_cp949:
            # 3. Try reading as raw bytes, decoding to UTF-8 while ignoring corrupt characters
            try:
                with open(file_path, 'rb') as f:
                    content_bytes = f.read()
                content_str = content_bytes.decode('utf-8', errors='ignore')
                data = json.loads(content_str, strict=False)
                return file_name, data
            except Exception as e_fallback:
                print(f"Error reading {file_name}: UTF-8: {e_utf8}, CP949: {e_cp949}, Fallback: {e_fallback}")
                return file_name, None

total_files = len(json_files)
start_time = time.time()
last_report_time = start_time

# Create a list of the same length to hold results in the correct sorted order
results = [None] * total_files

print(f"Running file merging using ThreadPoolExecutor...")

with ThreadPoolExecutor() as executor:
    # Submit all file reading tasks, mapping future to its original sorted index
    future_to_index = {executor.submit(process_file, fp): i for i, fp in enumerate(json_files)}
    
    completed_count = 0
    for future in as_completed(future_to_index):
        idx = future_to_index[future]
        file_name, data = future.result()
        results[idx] = (file_name, data)
        completed_count += 1
        
        # Report progress every 5000 files or if 5 seconds have passed
        current_time = time.time()
        if completed_count % 5000 == 0 or completed_count == total_files or (current_time - last_report_time) > 5.0:
            elapsed = current_time - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining = total_files - completed_count
            eta = remaining / rate if rate > 0 else 0
            
            percent = (completed_count / total_files) * 100
            print(f"[Progress] {completed_count}/{total_files} files processed ({percent:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | Speed: {rate:.1f} files/sec")
            last_report_time = current_time

print("All files processed. Constructing final JSON structure...")

# Prepend 'file_name' key while preserving the original dictionary keys and order exactly
final_data = []
for file_name, data in results:
    if data is not None:
        ordered_data = {"file_name": file_name}
        ordered_data.update(data)
        final_data.append(ordered_data)

print("Writing merged data to JSON file...")

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write to JSON file with UTF-8 encoding
# ensure_ascii=False ensures Korean characters are written directly in UTF-8 rather than escaped
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

total_elapsed = time.time() - start_time
print(f"Successfully merged {len(final_data)} files into {output_file}")
print(f"Total time taken: {total_elapsed:.1f} seconds.")
