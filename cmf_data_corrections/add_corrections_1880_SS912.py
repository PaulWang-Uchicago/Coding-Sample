import pandas as pd
import os

# 1) Load each SS<i> and set MultiIndex
ss_list = [9, 12]
cmf_files = {
    f'SS{i}': pd.read_stata(f'CMF/CMF_1880_SS{i}.dta').set_index(['file_name','firm_number'])
    for i in ss_list
}

# 2) Revised updater that searches for the key in each DF
def update_cmf(cmf_dict, check_df, update_mapping,
               condition_col='transcription_error', condition_val=1):
    updates = check_df[check_df[condition_col] == condition_val]
    for _, row in updates.iterrows():
        key = (row['file_name'], row['firm_number'])
        # search which SS‐DF contains this key
        for ss_key, cmf in cmf_dict.items():
            if key in cmf.index:
                # apply all corrections for that one DF
                for target_col, source_col in update_mapping.items():
                    val = row[source_col]
                    if pd.notna(val):
                        cmf.loc[key, target_col] = val
                break  # done—move on to next row

# 3) Loop through your merged check files
check_dir = "check/1880/SS912"
for fname in os.listdir(check_dir):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(check_dir, fname))
    mapping = {
        col.replace("correct_",""): col
        for col in df.columns
        if col.startswith("correct_")
    }
    update_cmf(cmf_files, df, mapping)

# 4) Save each corrected SS back to .dta
for ss, df in cmf_files.items():
    df.reset_index().to_stata(f"CMF/CMF_1880_{ss}_updated.dta", write_index=False)