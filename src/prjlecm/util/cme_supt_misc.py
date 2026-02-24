import numpy as np

def print_dict_aligned(obj, st_title="", indent=0, prefix_base_indent = 0):
    """Print with proper formatting.  
    Handles: 
    - Standalone numpy arrays
    - Single-level dicts with numpy array values
    - Nested dicts with simple values
    """
    print(st_title)
    indent_str = "    " * indent
    
    # If object is a numpy array, just print it nicely
    if isinstance(obj, np.ndarray):
        array_str = np.array2string(
            obj, 
            separator=', ',
            prefix=indent_str
        )
        print(f"{indent_str}{array_str}")
        return
    
    # If object is not a dict, print normally
    if not isinstance(obj, dict):
        print(f"{indent_str}{obj}")
        return
    
    # Handle dictionary
    print(f"{indent_str}{{")
    
    items = list(obj.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        
        # Check if value is a numpy array
        if isinstance(value, np.ndarray):
            print(f"{indent_str}    '{key}':")
            array_str = np.array2string(
                value, 
                separator=', ',
                prefix=' ' * (prefix_base_indent + indent * 4)
            )
            lines = array_str.split('\n')
            for j, line in enumerate(lines):
                if j == 0:
                    print(f"{indent_str}        np.array({line}", end="")
                else:
                    print(f"\n{indent_str}        {' ' * 9}{line}", end="")
            print(f"){comma}")
        
        # Check if value is a dict
        elif isinstance(value, dict):
            print(f"{indent_str}    {key}:  {{")
            sub_items = list(value.items())
            for j, (sub_key, sub_value) in enumerate(sub_items):
                sub_comma = "," if j < len(sub_items) - 1 else ""
                print(f"{indent_str}        '{sub_key}': {sub_value}{sub_comma}")
            print(f"{indent_str}    }}{comma}")
        
        # Regular values
        else:
            print(f"{indent_str}    '{key}': {value}{comma}")
    
    print(f"{indent_str}}}")