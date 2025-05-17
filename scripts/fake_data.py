"""
fake_data.py - Synthetic Voter Data Generation Module

This module generates synthetic voter data that mimics the structure and statistical
properties of real New York State voter files without containing any actual voter
information. Useful for testing, development, and demonstration purposes.

Author: [Your Name]
Date: May 17, 2025
"""

import pandas as pd
import numpy as np
import random
import datetime
import re
import uuid
from faker import Faker
import os
from dotenv import load_dotenv

load_dotenv()

def generate_fake_voter_data(num_voters=1000, seed=42, party_enrollments=None, 
                             party_weights=None, eds=None, vote_count_distribution=None):
    """
    Generate synthetic voter data that mimics the structure of NYS voter files
    but contains no real voter information.
    
    Args:
        num_voters (int): Number of synthetic voters to generate
        seed (int): Random seed for reproducibility
        party_enrollments (list): List of party enrollment codes to use
        party_weights (list): Probability weights for each party enrollment
        eds (list): List of election districts to use
        vote_count_distribution (pd.Series): Distribution of voting frequency
    
    Returns:
        pd.DataFrame: DataFrame of synthetic voter records matching NYS format
    
    Note: 
        Generated voter records include realistic voting histories with temporal
        patterns that can be processed by the feature_eng.py module.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a Faker instance for generating names and addresses
    fake = Faker()
    
    # Define possible values for categorical fields if not provided
    if party_enrollments is None or party_weights is None:
        party_enrollments = ['DEM', 'REP', 'BLK', 'OTH', 'CON', 'WOR']
        party_weights = [0.40, 0.38, 0.15, 0.05, 0.01, 0.01]  # Approximate typical distribution
    
    other_parties = ['IND', 'GRE', 'LBT', 'OTH']
    
    voting_methods = ['P', 'A', 'E', 'M', 'F', 'O', 'D', 'T']
    voting_method_weights = [0.70, 0.15, 0.08, 0.04, 0.01, 0.01, 0.005, 0.005]
    
    status_codes = ['A', 'I']
    status_weights = [0.92, 0.08]
    
    if eds is None:
        eds = list(range(1, 30))  # EDs 1-29
    legislative_districts = list(range(1, 15))  # LDs 1-14
    
    towns = ['City Binghamton', 'Town Binghamton', 'Union', 'Vestal', 'Conklin', 
             'Kirkwood', 'Chenango', 'Colesville', 'Windsor', 'Maine', 'Lisle']
    
    zip_codes = ['13901', '13903', '13905', '13850', '13760', '13790', '13795', '13904']
    
    # Generate voting histories with realistic temporal patterns
    def generate_voting_history(reg_year, vote_count=None):
        """
        Generate realistic voting history based on registration year and voting frequency.
        Models patterns like recency effects and higher turnout for general elections.
        
        Args:
            reg_year (int): Year voter registered
            vote_count (int, optional): Target number of votes to generate
            
        Returns:
            str: Semicolon-separated voting history string
        """
        if vote_count is None and vote_count_distribution is not None:
            # Sample from the distribution if provided
            vote_count = np.random.choice(
                vote_count_distribution.index, 
                p=vote_count_distribution.values
            )
        
        consistency = random.betavariate(2, 3)  # Beta distribution for realistic consistency
            
        current_year = 2025
        max_year = current_year
        
        # Calculate which election cycles the voter could have participated in
        possible_elections = []
        
        # Add general elections (even years)
        for year in range(max(reg_year, 2000), current_year + 1, 2):
            possible_elections.append(f"{year} General Election")
            
        # Add primary elections (even years)
        for year in range(max(reg_year, 2000), current_year + 1, 2):
            if random.random() < 0.8:  # Not all election cycles have primaries
                possible_elections.append(f"{year} Primary Election")
                
        # Add presidential primaries (presidential election years)
        for year in range(max(reg_year, 2000), current_year + 1, 4):
            if year % 4 == 0:  # Presidential years
                possible_elections.append(f"{year} Presidential Primary Election")
                
        # Add local/mayoral elections (odd years)
        for year in range(max(reg_year, 2001), current_year + 1, 2):
            if year % 2 == 1:  # Odd years
                possible_elections.append(f"{year} General Election")
        
        # If we have a target vote count, use that
        if vote_count is not None:
            # Ensure vote_count doesn't exceed possible elections
            vote_count = min(vote_count, len(possible_elections))
            # Sample from possible elections
            participated_elections = random.sample(possible_elections, vote_count)
        else:
            # Otherwise use consistency
            participated_elections = []
            for election in possible_elections:
                # Create realistic participation patterns
                year = int(election.split()[0])
                
                # Voters participate more in recent elections and more in general elections
                recency_boost = (year - reg_year) / (current_year - reg_year) if current_year > reg_year else 0
                importance_boost = 0.2 if "General" in election else 0
                presidential_boost = 0.15 if year % 4 == 0 and "General" in election else 0
                
                election_probability = min(0.95, consistency + recency_boost + importance_boost + presidential_boost)
                
                if random.random() < election_probability:
                    participated_elections.append(election)
        
        # For each participated election, choose a voting method
        formatted_elections = []
        for election in participated_elections:
            # Choose a voting method based on the election
            if "2020" in election or "2022" in election:  # COVID era
                method_weights = [0.4, 0.3, 0.2, 0.08, 0.01, 0.01, 0, 0]  # More mail/early voting
            else:
                method_weights = voting_method_weights
                
            method = np.random.choice(voting_methods, p=method_weights)
            formatted_elections.append(f"{election}({method})")
                
        # Sort from most recent to oldest
        formatted_elections.sort(reverse=True, key=lambda x: int(x.split()[0]))
        
        # Join with semicolons
        return ";".join(formatted_elections)
    
    # Voter generation
    data = []
    for i in range(num_voters):
        # Generate a random registration year between 1980 and 2024
        reg_year = random.randint(1980, 2024)
        reg_date_str = f"{reg_year}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
        
        # Generate a random birth year for someone 18+ at registration
        birth_year = random.randint(reg_year - 80, reg_year - 18)
        birth_date_str = f"{birth_year}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
        
        # Generate vote count if distribution is provided
        vote_count = None
        if vote_count_distribution is not None:
            vote_count_idx = np.random.choice(
                range(len(vote_count_distribution)),
                p=vote_count_distribution.values
            )
            vote_count = vote_count_distribution.index[vote_count_idx]
        
        # Generate the voter record
        voter = {
            "LASTNAME": fake.last_name().upper(),
            "FIRSTNAME": fake.first_name().upper(),
            "MIDDLENAME": random.choice([fake.first_name()[0], ""]) if random.random() > 0.5 else "",
            "NAMESUFFIX": random.choice(["", "JR", "SR", "II", "III"]) if random.random() > 0.9 else "",
            "RADDNUMBER": str(random.randint(1, 4000)),
            "RHALFCODE": "",
            "RPREDIRECTION": "",
            "RSTREETNAME": fake.street_name().upper(),
            "RPOSTDIRECTION": "",
            "RAPARTMENTTYPE": "APT" if random.random() > 0.8 else "",
            "RAPARTMENT": str(random.randint(1, 10)) if random.random() > 0.8 else "",
            "RADDRNONSTD": "",
            "RCITY": random.choice(["BINGHAMTON", "ENDICOTT", "JOHNSON CITY", "VESTAL", "ENDWELL"]),
            "RZIP5": random.choice(zip_codes),
            "RZIP4": "",
            "MAILADD1": "",
            "MAILADD2": "",
            "MAILADD3": "",
            "MAILADD4": "",
            "DOB": birth_date_str,
            "GENDER": random.choice(["M", "F"]),
            "ENROLLMENT": np.random.choice(party_enrollments, p=party_weights),
            "OTHERPARTY": random.choice(other_parties) if random.random() > 0.9 else "",
            "COUNTYCODE": "04",  # Broome County
            "ED": str(random.choice(eds)),
            "LD": str(random.choice(legislative_districts)),
            "TOWNCITY": random.choice(towns),
            "WARD": str(random.randint(1, 15)),
            "CD": "19",  # Congressional District
            "SD": "52",  # State Senate District
            "AD": random.choice(["121", "123", "124"]),  # Assembly District
            "LASTVOTERDATE": f"{random.randint(2000, 2024)}{random.randint(1,12):02d}{random.randint(1,28):02d}",
            "PREVYEARVOTED": "",
            "PREVCOUNTY": "",
            "PREVADDRESS": "",
            "PREVNAME": "",
            "COUNTYVRNUMBER": str(random.randint(1000000, 9999999)),
            "REGDATE": reg_date_str,
            "VRSOURCE": random.choice(["CBOE", "DMV", "MAIL", "AGCY"]),
            "IDREQUIRED": random.choice(["Y", "N"]),
            "IDMET": random.choice(["Y", "N"]),
            "STATUS": np.random.choice(status_codes, p=status_weights),
            "REASONCODE": "MAIL-CHECK" if random.random() > 0.9 else "",
            "INACT_DATE": f"{random.randint(2015, 2024)}{random.randint(1,12):02d}{random.randint(1,28):02d}" if random.random() > 0.9 else "",
            "PURGE_DATE": "",
            "SBOEID": f"NY{str(uuid.uuid4())[:14]}",  # Generate a unique ID
            "voterhistory": generate_voting_history(reg_year, vote_count)
        }
        
        data.append(voter)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def create_synthetic_sample_similar_to_real(real_data_file, cols, num_voters=5000, output_file="synthetic_voter_data.csv"):
    """
    Create synthetic data with similar statistical properties to real data
    without using any actual voter information.
    
    Args:
        real_data_file (str): Path to the real data file
        cols (list): List of columns to use from the real data
        num_voters (int): Number of synthetic voters to generate
        output_file (str): Path to save the synthetic data
        
    Returns:
        pd.DataFrame: DataFrame of synthetic voter records
        
    Note:
        This function extracts statistical distributions from real data (like party
        enrollment percentages) but does not copy any individual voter records.
    """
    # Read the real data to get distributions (but not the actual values)
    real_df = pd.read_csv(real_data_file, header=None, names=cols)
    
    # Extract statistical distributions
    party_counts = real_df['ENROLLMENT'].value_counts(normalize=True)
    party_enrollments = party_counts.index.tolist()
    party_weights = party_counts.values.tolist()
    
    ed_counts = real_df['ED'].value_counts(normalize=True) 
    eds = ed_counts.index.tolist()
    
    # Get approximate voting frequency distribution
    # First filter out NaN and empty strings
    real_df = real_df[real_df['voterhistory'].notna() & (real_df['voterhistory'] != '')]
    
    # Count the number of semicolons and add 1 to get the total vote count
    real_df['vote_count'] = real_df['voterhistory'].str.count(';') + 1
    vote_counts = real_df['vote_count'].value_counts(normalize=True)
    
    # Generate the synthetic data
    df = generate_fake_voter_data(
        num_voters=num_voters, 
        party_enrollments=party_enrollments,
        party_weights=party_weights, 
        eds=eds,
        vote_count_distribution=vote_counts
    )
    
    # Export to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {num_voters} synthetic voter records in {output_file}")
    
    return df

# Example usage:
if __name__ == "__main__":
    # This would be your file path to the real data
    # You can provide either the real data file or a list of columns
    raw = os.environ.get("raw")
    
    path = raw
    # List of expected columns in your voter file
    cols = [
        "LASTNAME", "FIRSTNAME", "MIDDLENAME", "NAMESUFFIX", 
        "RADDNUMBER", "RHALFCODE", "RPREDIRECTION", "RSTREETNAME", 
        "RPOSTDIRECTION", "RAPARTMENTTYPE", "RAPARTMENT", "RADDRNONSTD", 
        "RCITY", "RZIP5", "RZIP4", "MAILADD1", "MAILADD2", "MAILADD3", 
        "MAILADD4", "DOB", "GENDER", "ENROLLMENT", "OTHERPARTY", 
        "COUNTYCODE", "ED", "LD", "TOWNCITY", "WARD", "CD", "SD", "AD", 
        "LASTVOTERDATE", "PREVYEARVOTED", "PREVCOUNTY", "PREVADDRESS", 
        "PREVNAME", "COUNTYVRNUMBER", "REGDATE", "VRSOURCE", "IDREQUIRED", 
        "IDMET", "STATUS", "REASONCODE", "INACT_DATE", "PURGE_DATE", 
        "SBOEID", "voterhistory"
    ]
    
    # Generate 25,000 synthetic voter records with similar statistical properties
    synthetic_df = create_synthetic_sample_similar_to_real(path, cols, 25000, "synthetic_voter_data.csv")