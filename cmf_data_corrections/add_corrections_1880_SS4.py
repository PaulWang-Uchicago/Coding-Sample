import pandas as pd
import os

# Load the CMF dataset
cmf = pd.read_stata("CMF/CMF_1880_SS4.dta")

# Ensure MultiIndex (if not already set)
cmf = cmf.set_index(['file_name', 'firm_number'])

# Define update function
def update_cmf(cmf, check_df, update_mapping, condition_col='transcription_error', condition_val=1):
    updates = check_df[check_df[condition_col] == condition_val]
    for _, row in updates.iterrows():
        key = (row['file_name'], row['firm_number'])
        for target_col, source_col in update_mapping.items():
            value = row[source_col]
            if pd.notna(value):
                cmf.loc[key, target_col] = value

# List of all check file paths
check_files = [
    "capital_check_1880_SS_4.csv",
    "capital&materials_value_check_1880_SS_4.csv",
    "capital&output_check_1880_SS_4.csv",
    "materials_value_check_1880_SS_4.csv",
    "materials_value&capital_check_1880_SS_4.csv",
    "materials_value&output_check_1880_SS_4.csv",
    "output_check_1880_SS_4.csv",
    "output&capital_check_1880_SS_4.csv",
    "output&materials_value_check_1880_SS_4.csv",
    "total_wages_check_1880_SS_4.csv",
    "total_wages&output_check_1880_SS_4.csv",
    "workers_adult_female_check_1880_SS_4.csv",
    "workers_adult_male_check_1880_SS_4.csv",
    "workers_adult_male&workers_adult_female&workers_children&total_wages_check_1880_SS_4.csv"
]

# Directory where the check files are stored
check_dir = "check/1880/SS4"

# Loop through and apply corrections
for file in check_files:
    path = os.path.join(check_dir, file)
    check_df = pd.read_csv(path)

    # Automatically generate update_mapping
    update_mapping = {
        col.replace("correct_", ""): col
        for col in check_df.columns
        if col.startswith("correct_")
    }

    update_cmf(cmf, check_df, update_mapping)

# Save the updated CMF dataset
cmf.reset_index(inplace=True)
output_path = "CMF/CMF_1880_SS4_updated.dta"
cmf.to_stata(output_path, write_index=False)
print(f"Updated CMF dataset saved to {output_path}")