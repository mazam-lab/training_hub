#!/usr/bin/env python3

import json

def scan_json_file():
    print("Scanning long_ds.jsonl for JSON parsing issues...")
    
    with open('long_ds.jsonl', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                json.loads(line)
                if line_num <= 15:
                    print(f"Line {line_num}: OK")
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON ERROR - {e}")
                print(f"Problematic content near position {e.pos}:")
                start = max(0, e.pos - 50)
                end = min(len(line), e.pos + 50)
                print(f"'{line[start:end]}'")
                print(" " * (e.pos - start) + "^")
                
                if line_num <= 20:  # Show first few errors in detail
                    print(f"Full line length: {len(line)}")
                    print("="*80)
                
                # Stop after finding first 5 errors
                break
            
            if line_num > 100:  # Don't scan the entire huge file
                print(f"Scanned first {line_num} lines successfully")
                break

if __name__ == "__main__":
    scan_json_file()






