import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def process_shelter_data(intakes_file, outcomes_file):
    print("Loading datasets...")
    intakes = pd.read_csv(intakes_file)
    outcomes = pd.read_csv(outcomes_file)

    print("Cleaning and standardizing date formats...")
    # 1. Parse Intakes DateTime (which is timezone-naive)
    intakes['DateTime'] = pd.to_datetime(intakes['DateTime'], format='mixed')
    intakes['Intake_Date'] = intakes['DateTime'] # Preserve for calculation later

    # 2. Parse Outcomes DateTime (Strip the trailing timezone offset first, e.g., "-05:00")
    # This ensures both datasets are treated as local Austin time.
    outcomes['DateTime_outcome_str'] = outcomes['DateTime'].astype(str).str.replace(r'[-+]\d{2}:\d{2}$', '', regex=True)
    outcomes['DateTime'] = pd.to_datetime(outcomes['DateTime_outcome_str'], format='mixed')
    outcomes['Outcome_Date'] = outcomes['DateTime'] # Preserve for calculation later

    print("Sorting data chronologically...")
    # 3. pd.merge_asof REQUIRES the dataframes to be sorted globally by the 'on' key (DateTime)
    intakes = intakes.sort_values(by='DateTime').reset_index(drop=True)
    outcomes = outcomes.sort_values(by='DateTime').reset_index(drop=True)

    print("Merging records to handle animals with multiple visits...")
    # 4. Merge asof correctly matches an intake with the NEXT chronological outcome for that specific Animal ID
    merged_data = pd.merge_asof(
        intakes, 
        outcomes, 
        on='DateTime', 
        by='Animal ID', 
        direction='forward', 
        suffixes=('_intake', '_outcome')
    )

    print("Calculating Length of Stay...")
    # 5. Calculate Length of Stay in continuous Days (fractional days)
    merged_data['Length_of_Stay_Days'] = (merged_data['Outcome_Date'] - merged_data['Intake_Date']).dt.total_seconds() / (24 * 3600)

    # 6. Final cleanup: Remove animals still in the shelter (null outcomes) or impossible negative stays
    final_df = merged_data.dropna(subset=['Length_of_Stay_Days'])
    final_df = final_df[final_df['Length_of_Stay_Days'] >= 0]
    
    # Save to CSV
    output_filename = 'cleaned_austin_shelter_data.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"Success! Master dataset saved as '{output_filename}' with {len(final_df)} records.")
    return final_df

# Run the function
intakes_file = "Austin_Animal_Center_Intakes__10_01_2013_to_05_05_2025_.csv"
outcomes_file = "Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv"

df = process_shelter_data(intakes_file, outcomes_file)

# Display the first few rows to verify the Length of Stay
print(df[['Animal ID', 'Intake_Date', 'Outcome_Date', 'Length_of_Stay_Days']].head())