import numpy as np
import ast
import argparse

def parse_chess_arrays(filename, output_shape=(128, 8, 8)):
    """
    Parse arrays from text file separated by '---------------' and convert to numpy array.
    """
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    
    # Split by separator but keep track of line numbers
    sections = content.split('---------------')
    
    print(f"Found {len(sections)} sections in the file")
    
    # Calculate line numbers for each section
    lines = content.split('\n')
    current_line = 1
    section_line_numbers = [1]  # First section starts at line 1
    
    for line in lines:
        if '---------------' in line:
            current_line += 1
            section_line_numbers.append(current_line)
        else:
            current_line += 1
    
    # Initialize list to store arrays
    arrays_list = []
    
    for i, section in enumerate(sections):
        section = section.strip()
        line_num = section_line_numbers[i] if i < len(section_line_numbers) else "unknown"
        
        # Debug: show section lengths
        if not section:
            print(f"Section {i+1} (line {line_num}): EMPTY - creating placeholder array")
            placeholder = np.full((8, 8), 100, dtype=int)
            arrays_list.append(placeholder)
            continue
        elif len(section) < 100:  # Probably not a full array
            print(f"Section {i+1} (line {line_num}): TOO SHORT ({len(section)} chars) - creating placeholder array")
            placeholder = np.full((8, 8), 100, dtype=int)
            arrays_list.append(placeholder)
            continue
            
        try:
            parsed_array = ast.literal_eval(section)
            np_array = np.array(parsed_array, dtype=int)
            
            if np_array.shape == (8, 8):
                arrays_list.append(np_array)
                print(f"Section {i+1} (line {line_num}): SUCCESS - Array {len(arrays_list)}")
            else:
                print(f"Section {i+1} (line {line_num}): WRONG SHAPE {np_array.shape} - creating placeholder array")
                placeholder = np.full((8, 8), 100, dtype=int)
                arrays_list.append(placeholder)
                
        except Exception as e:
            print(f"Section {i+1} (line {line_num}): ERROR - {str(e)} - creating placeholder array")
            print(f"PROBLEMATIC TEXT AT LINE {line_num}:")
            print("=" * 50)
            print(section[:300])
            print("=" * 50)
            placeholder = np.full((8, 8), 100, dtype=int)
            arrays_list.append(placeholder)
    
    print(f"\nTotal arrays processed: {len(arrays_list)}")
    
    if arrays_list:
        result_array = np.array(arrays_list)
        return result_array
    else:
        return None

def save_arrays(arrays, output_filename="chess_arrays_full.npy"):
    """Save the numpy array to a file"""
    if arrays is not None:
        np.save(output_filename, arrays)
        print(f"Arrays saved to {output_filename}")
        print(f"Final array shape: {arrays.shape}")
        
        # Count how many placeholder arrays (filled with 100s) we have
        placeholder_count = 0
        for i, arr in enumerate(arrays):
            if np.all(arr == 100):
                placeholder_count += 1
        
        print(f"Placeholder arrays (errors): {placeholder_count}")
        print(f"Successfully parsed arrays: {len(arrays) - placeholder_count}")
        
    else:
        print("No arrays to save")

def parse_args():
    parser = argparse.ArgumentParser(description='Parse chess arrays from text file and save as numpy array')
    parser.add_argument('--input', type=str, required=True,
                      help='Input text file containing chess arrays')
    parser.add_argument('--output', type=str, required=True,
                      help='Output NPY file to save the arrays')
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    args = parse_args()
    
    print("Starting to parse chess arrays...")
    arrays = parse_chess_arrays(args.input)
    
    if arrays is not None:
        print(f"\nParsing complete!")
        print(f"Final array shape: {arrays.shape}")
        
        save_arrays(arrays, output_filename=args.output)
        
        print(f"\nFirst parsed array:")
        print(arrays[0])
        
    else:
        print("Failed to parse arrays")