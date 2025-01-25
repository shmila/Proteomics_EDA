import os

# Directory to scan
dir_path = r"C:\Users\eliran.shmi\Documents\Thesis"

# Output file
output_file = r"C:\Users\eliran.shmi\Documents\Thesis\dir_structure"


# Generate listing
with open(output_file, 'w') as f:
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item)
            f.write(f"{item}\n")

print(f"Simplified directory listing saved to {output_file}")
