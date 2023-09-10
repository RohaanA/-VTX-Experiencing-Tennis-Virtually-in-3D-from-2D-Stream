import re

# Read the file and retrieve the names
with open("list.txt", "r") as file:
    names = file.readlines()

# Define a function to extract the numeric part from the name
def extract_number(name):
    return int(re.search(r"_(\d+)\.", name).group(1))

# Sort the names using the extracted numbers
sorted_names = sorted(names, key=extract_number)

# Write the sorted names to a new file
with open("sorted_file.txt", "w") as file:
    file.writelines(sorted_names)