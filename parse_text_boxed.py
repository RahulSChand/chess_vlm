import numpy as np
import ast
import pickle
import re

def parse_text_sections(filename, delimiter=None):
    """
    Parse text file by splitting on delimiter and return list of sections.
    Automatically detects dash-based delimiters (e.g., ---, -----, ----------)
    """
    try:
        with open(filename, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    
    # Auto-detect delimiter if not provided
    if delimiter is None:
        lines = content.split('\n')
        for line in lines:
            stripped_line = line.strip()
            # Check if line consists only of dashes and is at least 3 characters long
            if len(stripped_line) >= 3 and all(c == '-' for c in stripped_line):
                delimiter = stripped_line
                print(f"Auto-detected delimiter: '{delimiter}' (length: {len(delimiter)})")
                break
        
        if delimiter is None:
            print("Warning: No dash-based delimiter found, using default")
            delimiter = "---------------"
    
    # Split by delimiter and strip whitespace
    sections = [section.strip() for section in content.split(delimiter)]
    
    # Calculate line numbers for each section
    lines = content.split('\n')
    current_line = 1
    section_line_numbers = [1]  # First section starts at line 1
    
    for line in lines:
        if delimiter in line:
            current_line += 1
            section_line_numbers.append(current_line)
        else:
            current_line += 1

    return_sections = []
    
    # Remove empty sections and apply filter with error handling
    total_len = len(sections)
    for i, section in enumerate(sections):
        line_num = section_line_numbers[i] if i < len(section_line_numbers) else "unknown"
        
        try:
            # section_filtered = filter_section(section)
            section_filtered = extract_boxed(section)[0]
            # print(section_filtered)
            print(section_filtered)
            if len(section_filtered) == "":
                print(f"ERROR in section {i+1}")

            return_sections.append(section_filtered)
        except Exception as e:
            print(f"ERROR in section {i+1} (line {line_num}): {str(e)}")
            print(f"PROBLEMATIC TEXT:")
            print("=" * 30)
            print(section[:200])  # Show first 200 chars
            print("=" * 30)
            return_sections.append("ERR")
            # Continue processing - don't add this section to results
            continue
     
    print(f"Successfully processed {len(return_sections)} sections out of {total_len}")
    return return_sections

def filter_section(text):
    """
    Filter function - for now just returns input as-is.
    You can modify this later to extract specific content.
    """
    text = text.replace("\\boxed{", "").replace("}", "")
    return text

def extract_boxed(text):
    return re.findall(r'\\boxed\{([^}]+)\}', text)

def process_file(filename):
    """
    Parse file and apply filter to each section.
    """
    sections = parse_text_sections(filename, delimiter="---------------")
    
    if sections is None:
        return None
    
    # Apply filter to each section
    
    return sections

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Main execution
if __name__ == "__main__":
    filename = "out_qwen_move_64_overfit.txt"
    
    print("Parsing text sections...")
    results = process_file(filename)
    
    if results:
        print(f"Processed {len(results)} sections")
        print(len(results))
        save_pickle(results, "out_qwen_move_64_overfit.pkl")

    else:
        print("Failed to process file")