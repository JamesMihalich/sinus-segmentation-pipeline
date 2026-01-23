"""
Symptom Score Analysis Script

Transforms symptom score data from wide format (visit columns) to long format
where each visit is a separate row with naming scheme P####.1, P####.2, etc.

Scores: NOSE, SNOT-22, ENS6Q linked to their respective visits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and clean the symptom score data."""
    # Try different encodings to handle non-UTF-8 characters
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Fallback: read with errors ignored
        df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')

    # Standardize column names
    df.columns = ['Diagnosis', 'studyID', 'NOSE1', 'NOSE2', 'SNOT22_1', 'SNOT22_2', 'ENS6Q_1', 'ENS6Q_2']

    # Clean up empty strings
    df['studyID'] = df['studyID'].replace('', np.nan)
    df['Diagnosis'] = df['Diagnosis'].replace('', np.nan)

    # Forward-fill diagnosis for continuation rows
    df['Diagnosis'] = df['Diagnosis'].ffill()

    # Remove rows without studyID (empty separator rows)
    df = df[df['studyID'].notna()]

    # Clean studyID - standardize to uppercase and strip whitespace
    df['studyID'] = df['studyID'].str.upper().str.strip()

    # Handle "POST" entries - these are follow-up visits, extract base ID
    df['is_post'] = df['studyID'].str.contains('POST', case=False, na=False)
    df['base_studyID'] = df['studyID'].str.replace(r'\s*POST\s*', '', regex=True, case=False)

    # Clean diagnosis text
    df['Diagnosis'] = df['Diagnosis'].str.strip()
    df['Diagnosis'] = df['Diagnosis'].str.replace(r'[^\x00-\x7F]+', '', regex=True)  # Remove non-ASCII

    return df


def clean_score(value):
    """Convert a score value to numeric, handling various NA representations."""
    if pd.isna(value):
        return np.nan

    val_str = str(value).strip()

    # Handle NA representations
    if val_str.upper() in ['NA', 'N/A', '-', '', 'NAN', 'NONE', 'EMAILED']:
        return np.nan

    # Handle formulas like "20*5=100" - extract the result
    match = re.search(r'=(\d+)', val_str)
    if match:
        return float(match.group(1))

    # Try to convert to numeric
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def transform_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform from wide format to long format with visit-based naming.

    Input columns: studyID, NOSE1, NOSE2, SNOT22_1, SNOT22_2, ENS6Q_1, ENS6Q_2
    Output: PatientID (P####.1, P####.2), Diagnosis, NOSE, SNOT22, ENS6Q
    """
    rows = []

    for _, row in df.iterrows():
        base_id = row['base_studyID']
        diagnosis = row['Diagnosis']
        is_post = row['is_post']

        # Clean all scores
        nose1 = clean_score(row['NOSE1'])
        nose2 = clean_score(row['NOSE2'])
        snot1 = clean_score(row['SNOT22_1'])
        snot2 = clean_score(row['SNOT22_2'])
        ens1 = clean_score(row['ENS6Q_1'])
        ens2 = clean_score(row['ENS6Q_2'])

        if is_post:
            # POST entries - scores go to visit 2 (or later)
            # The scores in the row are the post scores
            if not all(pd.isna([nose1, snot1, ens1])):
                rows.append({
                    'PatientID': f'{base_id}.2',
                    'Diagnosis': diagnosis,
                    'NOSE': nose1 if not pd.isna(nose1) else nose2,
                    'SNOT22': snot1 if not pd.isna(snot1) else snot2,
                    'ENS6Q': ens1 if not pd.isna(ens1) else ens2
                })
        else:
            # Regular entry - Visit 1 scores
            has_visit1 = not all(pd.isna([nose1, snot1, ens1]))
            has_visit2 = not all(pd.isna([nose2, snot2, ens2]))

            if has_visit1:
                rows.append({
                    'PatientID': f'{base_id}.1',
                    'Diagnosis': diagnosis,
                    'NOSE': nose1,
                    'SNOT22': snot1,
                    'ENS6Q': ens1
                })

            if has_visit2:
                rows.append({
                    'PatientID': f'{base_id}.2',
                    'Diagnosis': diagnosis,
                    'NOSE': nose2,
                    'SNOT22': snot2,
                    'ENS6Q': ens2
                })

    result_df = pd.DataFrame(rows)

    # Sort by PatientID
    result_df['sort_key'] = result_df['PatientID'].apply(
        lambda x: (int(re.search(r'P(\d+)', x).group(1)), float(x.split('.')[-1]))
        if re.search(r'P(\d+)', x) else (9999, 0)
    )
    result_df = result_df.sort_values('sort_key').drop(columns=['sort_key'])

    return result_df


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'symptom_score_list.csv'
    output_path = script_dir / 'symptom_scores_by_visit.csv'

    print("Loading and cleaning data...")
    df = load_and_clean_data(csv_path)

    print("Transforming to long format with visit-based naming...")
    result_df = transform_to_long_format(df)

    # Save output
    result_df.to_csv(output_path, index=False)

    print(f"\nOutput saved to: {output_path}")
    print(f"Total visits: {len(result_df)}")
    print(f"Unique patients: {result_df['PatientID'].str.extract(r'(P\d+)')[0].nunique()}")
    print(f"Visit 1 entries: {result_df['PatientID'].str.endswith('.1').sum()}")
    print(f"Visit 2 entries: {result_df['PatientID'].str.endswith('.2').sum()}")

    print("\nSample output:")
    print(result_df.head(10).to_string(index=False))

    print("\nDiagnosis distribution:")
    print(result_df['Diagnosis'].value_counts().head(15))


if __name__ == '__main__':
    main()
