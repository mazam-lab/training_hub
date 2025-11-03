#!/usr/bin/env python3

import pyarrow as pa
import json

def test_json_parsing():
    try:
        # Try to read the file line by line with PyArrow
        import pyarrow.json as pj
        
        # Read the problematic file
        print("Testing PyArrow JSON parsing...")
        
        # Try reading just the first 15 lines
        with open('long_ds.jsonl', 'r') as f:
            for i, line in enumerate(f, 1):
                if i > 15:
                    break
                    
                try:
                    # Test with regular JSON parser
                    json.loads(line.strip())
                    print(f"Line {i}: OK with json.loads")
                except json.JSONDecodeError as e:
                    print(f"Line {i}: FAILED with json.loads - {e}")
                
                try:
                    # Test with PyArrow
                    table = pj.read_json(pa.py_buffer(line.encode()))
                    print(f"Line {i}: OK with PyArrow")
                except Exception as e:
                    print(f"Line {i}: FAILED with PyArrow - {e}")
                    
                if i == 11:
                    print(f"Line 11 content length: {len(line)}")
                    print(f"Line 11 first 100 chars: {line[:100]}")
                    print(f"Line 11 last 100 chars: {line[-100:]}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_json_parsing()






