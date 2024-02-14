# Let's first read the contents of the uploaded file to understand its structure

import datetime

import pandas as pd

file_path = "/home/psa_images/SemiF-AnnotationPipeline/.batchlogs/backlog_division.txt"

# Reading the file and parsing it into a DataFrame

# Initialize variables to hold the data
data = []
current_state = None
current_season = None

# Read and parse the file
with open(file_path, "r") as file:
    for line in file:
        stripped_line = line.strip()

        # Check if line is a state category (not indented)
        if not line.startswith("  "):  # State category
            current_state = stripped_line.replace("State: ", "")
            # print(current_state)
            current_season = None
        elif line.startswith("  ") and not line.startswith("    - "):  # Season category
            current_season = stripped_line.replace(":", "")
            # print(current_season)
        else:  # Indented lines are values
            value = stripped_line.replace("- ", "")

            if current_state and current_season and value:
                data.append([current_state, current_season, value])

# Create DataFrame
df = pd.DataFrame(data, columns=["State", "Season", "Batch"])
df["Name"] = None

size = len(df) // 3
# Splitting the DataFrame into three sub-DataFrames
df.loc[:size, "Name"] = "Jordan"
df.loc[size : 2 * size, "Name"] = "Zack"
df.loc[2 * size :, "Name"] = "Courtney"
print(df.groupby(["Name"])["Batch"].nunique())

# Save DataFrame to CSV
timestamp = datetime.datetime.now().strftime("%Y%m%d")
csv_file_path = f"divided_backlog_{timestamp}.csv"
df.to_csv(csv_file_path, index=True)
