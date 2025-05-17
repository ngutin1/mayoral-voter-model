"""
feature_eng.py - Voter Feature Engineering Module

This module extracts predictive features from voter history data for mayoral election turnout modeling.
It parses raw voter history strings into structured data and generates behavioral features.

Author: Nicholas Gutin
Date: May 2025
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


def parse_voter_history(history_string):
    """
    Parse the voter history string into structured data, handling multiple formats:
    
    Format examples: 
    - '20170912 PR(P);20161108 GE(P)' (original format - date first)
    - 'GE 20241105(M);PP 20200623(A)' (election type first, then date)
    - '2004 GENERAL ELECTION(P)' (year first, then full election name)
    """
    if pd.isna(history_string) or history_string == '':
        return []
    
    # Split by semicolon
    events = history_string.split(';')
    parsed_events = []
    
    for event in events:
        # Strip any whitespace
        event = event.strip()
        if not event:
            continue
            
        # Try to extract voting method which is consistent across formats
        method_match = re.search(r'\(([A-Z])\)$', event)
        if not method_match:
            continue  # Skip if no voting method found
            
        vote_method = method_match.group(1)
        
        # Create variables to hold extracted data
        year = None
        election_name = None
        
        # Pattern 1: Date first format - "20170912 PR(P)"
        date_first_match = re.match(r'(\d{8})\s+([A-Z]+)\([A-Z]\)', event)
        
        # Pattern 2: Election type first format - "GE 20241105(M)"
        election_first_match = re.match(r'([A-Z]+)\s+(\d{8})\([A-Z]\)', event)
        
        # Pattern 3: Year + election name format - "2004 GENERAL ELECTION(P)"
        year_name_match = re.match(r'(\d{4})\s+(.+?)\([A-Z]\)', event)
        
        if date_first_match:
            # Extract date and convert to year
            date_str, election_abbr = date_first_match.groups()
            year = int(date_str[:4])
            
            # Map common abbreviations to election names
            election_map = {
                'GE': 'General Election',
                'PE': 'Primary Election',
                'PR': 'Primary Election',
                'PP': 'Presidential Primary'
            }
            election_name = election_map.get(election_abbr, election_abbr)
            
        elif election_first_match:
            # Extract election type and date
            election_abbr, date_str = election_first_match.groups()
            year = int(date_str[:4])
            
            # Map common abbreviations to election names
            election_map = {
                'GE': 'General Election',
                'PE': 'Primary Election',
                'PR': 'Primary Election',
                'PP': 'Presidential Primary'
            }
            election_name = election_map.get(election_abbr, election_abbr)
            
        elif year_name_match:
            # Extract year and election name directly
            year_str, election_name = year_name_match.groups()
            year = int(year_str)
        
        else:
            # If no pattern matches, try a more flexible approach for year extraction
            year_match = re.search(r'(\d{4})', event)
            if year_match:
                year = int(year_match.group(1))
                # Extract anything that might be the election name
                name_match = re.search(r'([A-Za-z\s]+)', event)
                if name_match:
                    election_name = name_match.group(1).strip()
                else:
                    election_name = "Unknown"
            else:
                # Cannot extract year, skip this event
                continue
        
        # Determine if this is a local-only election (odd-year general)
        is_local_only = False
        if election_name:
            is_local_only = ('General' in election_name or 'GE' == election_name) and (year % 2 == 1)
        
        # Determine if this is a primary election
        is_primary = False
        if election_name:
            is_primary = ('Primary' in election_name or 
                         'PR' == election_name or 
                         'PE' == election_name or 
                         'PP' == election_name)
        
        # Determine if this is a general election
        is_general = False
        if election_name:
            is_general = ('General' in election_name or 'GE' == election_name)
        
        # Create event dictionary with just the specified fields
        parsed_events.append({
            'year': year,
            'election_name': election_name,
            'vote_method': vote_method,
            'vote_method_desc': get_vote_method_description(vote_method),
            'is_general': is_general,
            'is_local_only': is_local_only,
            'is_primary': is_primary
        })
    
    return sorted(parsed_events, key=lambda x: x.get('year', 0), reverse=True)  # Sort by year, most recent first


def get_vote_method_description(code):
    """
    Return human-readable description of voting method code.
    
    Args:
        code (str): Single-letter voting method code
        
    Returns:
        str: Description of the voting method
    """

    methods = {
        'P': 'Election Day Poll Site',
        'A': 'Absentee',
        'F': 'Election Day Affidavit',
        'O': 'Election Day by Other Method',
        'E': 'Early Voting at Poll Site',
        'D': 'Early Voting by Affidavit',
        'T': 'Early Voting by Other Method',
        'M': 'Early Mail Voting'
    }
    return methods.get(code, 'Unknown Method')


def add_geographic_features(voter_df, district_col='ED'):
    """
    Add district-level aggregated features to voter data.
    
    Args:
        voter_df (DataFrame): Voter data
        district_col (str): Column containing district identifiers
        
    Returns:
        DataFrame: Voter data with added geographic features
    """

    # Calculate district-level metrics
    district_stats = {}
    for district in voter_df[district_col].unique():
        district_voters = voter_df[voter_df[district_col] == district]
        
        # Skip if insufficient data
        if len(district_voters) < 10:
            continue
        
        # Calculate district-level statistics
        district_stats[district] = {
            'avg_turnout': district_voters['voted_last_mayoral'].mean(),
            'voter_count': len(district_voters),
            'consistent_voter_rate': district_voters['is_consistent_voter'].mean()
        }
    
    # Add district-level features to individual voters
    for district, stats in district_stats.items():
        mask = voter_df[district_col] == district
        voter_df.loc[mask, 'district_avg_turnout'] = stats['avg_turnout']
        voter_df.loc[mask, 'district_size'] = stats['voter_count']
        voter_df.loc[mask, 'district_consistency'] = stats['consistent_voter_rate']
    
    # Fill missing values
    numeric_cols = ['district_avg_turnout', 'district_size', 'district_consistency']
    voter_df[numeric_cols] = voter_df[numeric_cols].fillna(voter_df[numeric_cols].mean())
    
    return voter_df

def calculate_recency_weighted_score(voter_history, current_year=2025):
    """
    Calculate score giving higher weight to recent voting activity.
    
    Args:
        voter_history (list): Parsed voting history
        current_year (int): Current year for recency calculation
        
    Returns:
        float: Recency-weighted participation score
    """
    
    if not voter_history:
        return 0
    
    # Calculate weights based on recency (more recent = higher weight)
    max_years_back = 10
    scores = []
    
    for event in voter_history:
        years_ago = current_year - event['year']
        if years_ago <= max_years_back:
            # Exponential decay weight
            weight = 2.0 ** (-years_ago / 3)  # Half-life of 3 years
            scores.append(weight)
    
    return sum(scores) / len(scores) if scores else 0

def normalize_by_first_election(voter_history, all_election_dates):
    """
    Calculate features normalized by first election date.
    
    Args:
        voter_history (list): Parsed voting history
        all_election_dates (list): All election dates in dataset
        
    Returns:
        dict: Normalized voting metrics
    """

    if not voter_history:
        return {
            'participation_rate': 0,
            'normalized_total_votes': 0,
            'elections_since_first': 0
        }
    
    # Find first election date for this voter
    first_election = min(event['year'] for event in voter_history)
    
    # Count elections since their first election
    eligible_elections = [date for date in all_election_dates if date >= first_election]
    elections_since_first = len(eligible_elections)
    
    # Calculate normalized metrics
    total_votes = len(voter_history)
    participation_rate = total_votes / max(1, elections_since_first)
    
    return {
        'participation_rate': participation_rate,
        'normalized_total_votes': total_votes / max(1, elections_since_first),
        'elections_since_first': elections_since_first
    }
    
def calculate_voting_trend(voter_history, recent_years=6):
    """
    Calculate trend direction of voting participation.
    
    Args:
        voter_history (list): Parsed voting history
        recent_years (int): Number of recent years to analyze
        
    Returns:
        int: Trend indicator (1=up, 0=stable, -1=down)
    """

    if len(voter_history) < 2:
        return 0  # Neutral trend
    
    # Sort by year (oldest first)
    sorted_history = sorted(voter_history, key=lambda x: x['year'])
    
    # Get years with elections they could have voted in
    all_possible_years = set()
    for event in sorted_history:
        all_possible_years.add(event['year'])
    
    # Only consider recent years
    recent_years_set = {year for year in all_possible_years 
                      if year >= max(all_possible_years) - recent_years}
    
    # Calculate early period vs recent period participation
    if len(recent_years_set) >= 4:
        midpoint = sorted(recent_years_set)[len(recent_years_set)//2]
        early_years = [y for y in recent_years_set if y < midpoint]
        later_years = [y for y in recent_years_set if y >= midpoint]
        
        # Calculate participation rates
        early_participation = sum(1 for event in voter_history 
                                if event['year'] in early_years) / len(early_years)
        later_participation = sum(1 for event in voter_history 
                               if event['year'] in later_years) / len(later_years)
        
        # Determine trend
        if later_participation > early_participation + 0.1:
            return 1  # Upward trend
        elif later_participation < early_participation - 0.1:
            return -1  # Downward trend
    
    return 0  # Stable trend

def extract_mayoral_election_patterns(voter_history, mayoral_dates):
    """
    Extract patterns specific to mayoral elections.
    
    Args:
        voter_history (list): Parsed voting history
        mayoral_dates (list): Mayoral election dates
        
    Returns:
        dict: Mayoral election metrics
    """
    
    if not voter_history or not mayoral_dates:
        return {
            'mayoral_participation_rate': 0,
            'last_mayoral_participation': 0
        }
    
    # Identify mayoral election participation
    mayoral_votes = []
    for event in voter_history:
        event_date = f"{event['year']}{event.get('month', '00'):02d}{event.get('day', '00'):02d}"
        if event_date in mayoral_dates or (event['is_local_only'] and str(event['year']) in [date[:4] for date in mayoral_dates]):
            mayoral_votes.append(event)
    
    # Calculate metrics
    mayoral_participation_rate = len(mayoral_votes) / len(mayoral_dates)
    
    # Check if they voted in last mayoral election
    last_mayoral_participation = 0
    if mayoral_dates and mayoral_votes:
        most_recent_mayoral = max(mayoral_dates)
        for vote in mayoral_votes:
            vote_date = f"{vote['year']}{vote.get('month', '00'):02d}{vote.get('day', '00'):02d}"
            if vote_date == most_recent_mayoral:
                last_mayoral_participation = 1
                break
    
    return {
        'mayoral_participation_rate': mayoral_participation_rate,
        'last_mayoral_participation': last_mayoral_participation
    }
    
def analyze_election_type_participation(voter_history):
    """
    Analyze participation across election types.
    
    Args:
        voter_history (list): Parsed voting history
        
    Returns:
        dict: Participation rates by election type
    """
    
    if not voter_history:
        return {
            'general_rate': 0,
            'primary_rate': 0, 
            'local_rate': 0,
            'participates_all_types': 0
        }
    
    # Count by type
    election_counts = {
        'general': sum(1 for e in voter_history if e.get('is_general', False)),
        'primary': sum(1 for e in voter_history if e.get('is_primary', False)),
        'local': sum(1 for e in voter_history if e.get('is_local_only', False))
    }
    
    # Get all possible elections (substitute with real data if available)
    # Example: In last 10 years, typically 5 general, 5 primary, 5 local elections
    possible_elections = {
        'general': 5,
        'primary': 5,
        'local': 5
    }
    
    # Calculate participation rates
    general_rate = election_counts['general'] / possible_elections['general']
    primary_rate = election_counts['primary'] / possible_elections['primary']
    local_rate = election_counts['local'] / possible_elections['local']
    
    # Check if participates in all types
    participates_all_types = int(all(count > 0 for count in election_counts.values()))
    
    return {
        'general_rate': general_rate,
        'primary_rate': primary_rate,
        'local_rate': local_rate,
        'participates_all_types': participates_all_types
    }

def calc_mayoral_participation(history, mayoral_years=None):
    """
    Calculate participation rate in mayoral elections.
    
    Args:
        history (list): Parsed voting history
        mayoral_years (list, optional): Years with mayoral elections
        
    Returns:
        float: Mayoral election participation rate
    """
    
    if not history:
        return 0.0
    
    # Extract all years the voter participated in
    participated_years = [event['year'] for event in history if 'year' in event]
    if not participated_years:
        return 0.0
    
    # Find first election year
    first_year = min(participated_years)
    
    # If mayoral_years not provided, derive them based on the pattern
    # Mayoral elections are in odd years after presidential elections
    if mayoral_years is None:
        # Find presidential election years (divisible by 4)
        presidential_years = [year for year in range(2000, 2028, 4)]
        # Mayoral elections are the year after presidential elections
        mayoral_years = [year + 1 for year in presidential_years]
    
    # Find eligible mayoral elections (those after first vote)
    eligible_mayoral_years = [year for year in mayoral_years if year >= first_year]
    
    if not eligible_mayoral_years:
        return 0.0
    
    # Count mayoral elections participated in
    mayoral_votes = 0
    for year in eligible_mayoral_years:
        # Check if voter participated in a general election in this mayoral year
        for event in history:
            if ('year' in event and event['year'] == year and 
                event.get('is_general', False)):
                mayoral_votes += 1
                break
    
    # Calculate participation rate
    participation_rate = mayoral_votes / len(eligible_mayoral_years)
    return participation_rate

def calc_local_participation(history, local_years=None):
    """
    Calculate participation rate in local (non-mayoral) elections
    
    Args:
        history: Parsed voter history list
        local_years: Optional list of years when local elections occurred
                    (if None, will use odd years after midterms)
        
    Returns:
        Local election participation rate
    """
    if not history:
        return 0.0
    
    # Extract all years the voter participated in
    participated_years = [event['year'] for event in history if 'year' in event]
    if not participated_years:
        return 0.0
    
    # Find first election year
    first_year = min(participated_years)
    
    # If local_years not provided, derive them based on the pattern
    # Local elections are in odd years after midterms
    if local_years is None:
        # Find midterm election years (even years not divisible by 4)
        midterm_years = [year for year in range(2002, 2028, 4)]
        # Local elections are the year after midterms
        local_years = [year + 1 for year in midterm_years]
    
    # Find eligible local elections (those after first vote)
    eligible_local_years = [year for year in local_years if year >= first_year]
    
    if not eligible_local_years:
        return 0.0
    
    # Count local elections participated in
    local_votes = 0
    for year in eligible_local_years:
        # Check if voter participated in a general election in this local year
        for event in history:
            if ('year' in event and event['year'] == year and 
                event.get('is_general', False)):
                local_votes += 1
                break
    
    # Calculate participation rate
    participation_rate = local_votes / len(eligible_local_years)
    return participation_rate
def extract_primary_election_features(voter_history):
    """
    Extract detailed features about primary election participation patterns
    
    Args:
        voter_history: List of parsed voting history events
    
    Returns:
        Dictionary with primary election features
    """
    if not voter_history:
        return {
            'primary_participation_rate': 0.0,
            'primary_to_general_conversion': 0.0,
            'contested_primary_participation': 0.0,
            'presidential_primary_rate': 0.0,
            'recent_primary_participation': 0,
            'primary_consistency': 0.0
        }
    
    # Get all events with year information
    valid_events = [e for e in voter_history if 'year' in e]
    if not valid_events:
        return {
            'primary_participation_rate': 0.0,
            'primary_to_general_conversion': 0.0,
            'contested_primary_participation': 0.0,
            'presidential_primary_rate': 0.0,
            'recent_primary_participation': 0,
            'primary_consistency': 0.0
        }
    
    # Sort by year
    sorted_events = sorted(valid_events, key=lambda x: x['year'])
    first_year = sorted_events[0]['year']
    current_year = 2025
    
    # Identify primary elections
    primary_events = [e for e in sorted_events if e.get('is_primary', False)]
    
    # Count elections by type and year
    election_years = {}
    for event in sorted_events:
        year = event['year']
        if year not in election_years:
            election_years[year] = {'primary': False, 'general': False}
        
        if event.get('is_primary', False):
            election_years[year]['primary'] = True
        elif event.get('is_general', False):
            election_years[year]['general'] = True
    
    # Calculate primary participation rate
    # Count years with primary elections since first vote
    eligible_primary_years = []
    for year in range(first_year, current_year + 1, 2):  # Primaries in even years typically
        if year >= first_year:
            eligible_primary_years.append(year)
    
    primary_years_participated = len([e for e in primary_events if e['year'] in eligible_primary_years])
    primary_participation_rate = primary_years_participated / max(1, len(eligible_primary_years))
    
    # Calculate primary-to-general conversion rate
    primary_to_general_pairs = 0
    primary_with_general = 0
    
    for year, elections in election_years.items():
        if elections['primary']:
            primary_with_general += 1
            if elections['general']:
                primary_to_general_pairs += 1
    
    primary_to_general_conversion = primary_to_general_pairs / max(1, primary_with_general)
    
    # Identify presidential primary years (divisible by 4)
    presidential_primary_years = [year for year in eligible_primary_years if year % 4 == 0]
    presidential_primaries_voted = len([e for e in primary_events if e['year'] in presidential_primary_years])
    presidential_primary_rate = presidential_primaries_voted / max(1, len(presidential_primary_years))
    
    # Calculate consistency in primary participation
    primary_streaks = []
    current_streak = 0
    
    for year in sorted(eligible_primary_years):
        participated = False
        for event in primary_events:
            if event['year'] == year:
                participated = True
                break
        
        if participated:
            current_streak += 1
        else:
            if current_streak > 0:
                primary_streaks.append(current_streak)
                current_streak = 0
    
    if current_streak > 0:
        primary_streaks.append(current_streak)
    
    primary_consistency = max(primary_streaks) if primary_streaks else 0
    primary_consistency = primary_consistency / max(1, len(eligible_primary_years) / 2)  # Normalize
    
    # Check for recent primary participation (last two cycles)
    recent_primary_years = [year for year in eligible_primary_years if year >= current_year - 4]
    recent_primary_participation = int(any(event['year'] in recent_primary_years for event in primary_events))
    
    # Calculate "contested primary" participation (when both primary and general exist in same year)
    contested_primaries = sum(1 for year, elections in election_years.items() 
                             if elections['primary'] and elections['general'])
    contested_primary_participation = contested_primaries / max(1, len(eligible_primary_years))
    
    return {
        'primary_participation_rate': primary_participation_rate,
        'primary_to_general_conversion': primary_to_general_conversion,
        'contested_primary_participation': contested_primary_participation,
        'presidential_primary_rate': presidential_primary_rate,
        'recent_primary_participation': recent_primary_participation,
        'primary_consistency': primary_consistency
    }
    
def extract_clean_features(df, parse_history_function, mayoral_dates=None):
    """
    Extract basic features from voter history data.
    
    Args:
        df (DataFrame): Voter data with 'voterhistory' column
        parse_history_function (function): Function to parse histories
        mayoral_dates (list, optional): Mayoral election dates
        
    Returns:
        tuple: (DataFrame with features, list of feature columns)
    """
    
    print(f"Extracting clean features from {len(df)} voter records")
    
    # Create a copy to work with
    result_df = df.copy()
    
    # Ensure no NaN values in voterhistory column
    result_df = result_df.dropna(subset=['voterhistory'])
    result_df = result_df[result_df['voterhistory'] != '']
    
    # Parse histories
    print("Parsing voter histories...")
    parsed_histories = []
    for hist in result_df['voterhistory']:
        try:
            parsed = parse_history_function(hist)
            parsed_histories.append(parsed)
        except:
            parsed_histories.append([])
    
    result_df['parsed_history'] = parsed_histories
    
    # Only keep voters with successfully parsed histories
    result_df = result_df[result_df['parsed_history'].apply(lambda x: len(x) > 0)]
    print(f"Kept {len(result_df)} voters with valid parsed histories")
    
    # Derive election dates
    print("Deriving all election dates from voter histories...")
    all_dates = set()
    for history in result_df['parsed_history']:
        for event in history:
            if 'year' in event:
                year = event['year']
                if event.get('is_general', False):
                    all_dates.add(f"{year}1108")
                elif event.get('is_primary', False):
                    all_dates.add(f"{year}0622")
                elif event.get('is_local_only', False):
                    all_dates.add(f"{year}1107")
    
    all_election_dates = sorted(list(all_dates), reverse=True)
    print(f"Derived {len(all_election_dates)} election dates")
    
    # Convert mayoral_dates to mayoral_years
    mayoral_years = None
    if mayoral_dates:
        mayoral_years = [int(date[:4]) for date in mayoral_dates]
    else:
        presidential_years = [year for year in range(2000, 2028, 4)]
        mayoral_years = [year + 1 for year in presidential_years]
    
    # Calculate local election years
    midterm_years = [year for year in range(2002, 2028, 4)]
    local_years = [year + 1 for year in midterm_years]
    
    print(f"Mayoral election years: {mayoral_years}")
    print(f"Local election years: {local_years}")
    
    # Extract basic features
    print("Extracting features...")
    
    result_df['has_voting_history'] = 1
    result_df['total_votes'] = result_df['parsed_history'].apply(len)
    result_df['voted_general'] = result_df['parsed_history'].apply(
        lambda x: int(any(e.get('is_general', False) for e in x))
    )
    result_df['voted_primary'] = result_df['parsed_history'].apply(
        lambda x: int(any(e.get('is_primary', False) for e in x))
    )
    result_df['voted_local'] = result_df['parsed_history'].apply(
        lambda x: int(any(e.get('is_local_only', False) for e in x))
    )
    result_df['is_consistent_voter'] = result_df['total_votes'].apply(
        lambda x: int(x >= 3)
    )
    
    # First election info
    print("Calculating first election information...")
    def get_first_election_info(history):
        if not history:
            return {
                'first_election_year': None, 
                'years_registered': 0,
                'elections_since_first': 0
            }
        
        years = [event['year'] for event in history if 'year' in event]
        if not years:
            return {
                'first_election_year': None, 
                'years_registered': 0,
                'elections_since_first': 0
            }
            
        first_year = min(years)
        current_year = 2025
        
        eligible_general_elections = len([y for y in range(first_year, current_year + 1) if y % 2 == 0])
        
        return {
            'first_election_year': first_year,
            'years_registered': current_year - first_year,
            'elections_since_first': eligible_general_elections
        }
    
    first_election_data = result_df['parsed_history'].apply(get_first_election_info)
    result_df['first_election_year'] = first_election_data.apply(lambda x: x['first_election_year'])
    result_df['years_registered'] = first_election_data.apply(lambda x: x['years_registered'])
    result_df['elections_since_first'] = first_election_data.apply(lambda x: x['elections_since_first'])
    
    # Calculate participation rates
    print("Calculating normalized participation rates...")
    result_df['participation_rate'] = result_df.apply(
        lambda row: row['total_votes'] / max(1, row['elections_since_first']), 
        axis=1
    )
    
    # Mayoral and local participation
    print("Calculating mayoral and local participation rates...")
    result_df['mayoral_participation_rate'] = result_df['parsed_history'].apply(
        lambda x: calc_mayoral_participation(x, mayoral_years)
    )
    
    result_df['local_participation_rate'] = result_df['parsed_history'].apply(
        lambda x: calc_local_participation(x, local_years)
    )
    
    # ORIGINAL TARGET EXTRACTION - THIS WAS WORKING
    print("Setting target variable...")
    if mayoral_dates:
        target_date = mayoral_dates[0]
        target_year = int(target_date[:4])
        
        # YOUR ORIGINAL LOGIC - DON'T CHANGE
        result_df['voted_last_mayoral'] = result_df['parsed_history'].apply(
            lambda x: any(e.get('year', 0) == target_year and 
                         e.get('is_general', False) and 
                         not e.get('is_primary', False)
                         for e in x)
        ).astype(int)
    else:
        most_recent_mayoral = max(mayoral_years)
        
        result_df['voted_last_mayoral'] = result_df['parsed_history'].apply(
            lambda x: any(e.get('year', 0) == most_recent_mayoral and 
                         e.get('is_general', False) and
                         not e.get('is_primary', False)
                         for e in x)
        ).astype(int)
    
    # Voting trend
    print("Calculating voting trends...")
    def calc_voting_trend(history, recent_years=6):
        if len(history) < 2:
            return 0
        
        sorted_history = sorted(history, key=lambda x: x.get('year', 0))
        years = [event.get('year', 0) for event in sorted_history]
        
        if len(set(years)) < 2:
            return 0
            
        first_year = min(years)
        last_year = max(years)
        all_years = list(range(first_year, last_year + 1))
        
        if len(all_years) > recent_years:
            all_years = all_years[-recent_years:]
            
        if len(all_years) < 4:
            return 0
            
        midpoint = len(all_years) // 2
        early_years = all_years[:midpoint]
        late_years = all_years[midpoint:]
        
        early_votes = sum(1 for event in history 
                         if 'year' in event and event['year'] in early_years)
        late_votes = sum(1 for event in history 
                        if 'year' in event and event['year'] in late_years)
        
        early_rate = early_votes / len(early_years) if early_years else 0
        late_rate = late_votes / len(late_years) if late_years else 0
        
        if late_rate > early_rate + 0.15:
            return 1
        elif late_rate < early_rate - 0.15:
            return -1
        return 0
    
    result_df['voting_trend'] = result_df['parsed_history'].apply(calc_voting_trend)
    
    # Vote method preference
    def get_preferred_method(history):
        methods = [e.get('vote_method', '') for e in history if 'vote_method' in e]
        if not methods:
            return ''
        
        method_counts = {}
        for m in methods:
            method_counts[m] = method_counts.get(m, 0) + 1
            
        return max(method_counts.items(), key=lambda x: x[1])[0] if method_counts else ''
    
    result_df['preferred_method'] = result_df['parsed_history'].apply(get_preferred_method)
    result_df['prefers_early_voting'] = result_df['preferred_method'].apply(
        lambda x: int(x in ['E', 'D', 'T'])
    )
    result_df['prefers_absentee'] = result_df['preferred_method'].apply(
        lambda x: int(x == 'A')
    )
    
    # Method change
    def has_method_changed(history):
        methods = [e.get('vote_method', '') for e in history if 'vote_method' in e]
        if len(methods) < 2:
            return 0
            
        year_method_pairs = [(e.get('year', 0), e.get('vote_method', '')) 
                            for e in history if 'vote_method' in e and 'year' in e]
        sorted_pairs = sorted(year_method_pairs, key=lambda x: x[0])
        
        recent_methods = [m for _, m in sorted_pairs[-2:]]
        return int(recent_methods[0] != recent_methods[1])
    
    result_df['recent_method_change'] = result_df['parsed_history'].apply(has_method_changed)
    
    # Recency score
    def calc_recency_score(history, current_year=2025):
        if not history:
            return 0
            
        scores = []
        max_years_back = 10
        
        for event in history:
            if 'year' in event:
                years_ago = current_year - event['year']
                if years_ago <= max_years_back:
                    weight = 2.0 ** (-years_ago / 3)
                    scores.append(weight)
                    
        return sum(scores) / len(scores) if scores else 0
        
    result_df['recency_score'] = result_df['parsed_history'].apply(calc_recency_score)
    
    # Final cleanup
    feature_cols = [
        'has_voting_history', 'total_votes', 'voted_general', 'voted_primary',
        'voted_local', 'is_consistent_voter', 'prefers_early_voting', 'prefers_absentee',
        'years_registered', 'elections_since_first', 'participation_rate',
        'mayoral_participation_rate', 'local_participation_rate', 'voting_trend', 
        'recent_method_change', 'recency_score'
    ]
    
    for col in feature_cols:
        nan_count = result_df[col].isna().sum()
        if nan_count > 0:
            result_df[col] = result_df[col].fillna(0)
    
    print("Feature extraction complete")
    return result_df, feature_cols

def extract_temporal_features(voter_history):
    """
    Extract features about changes in voting behavior over time.
    
    Args:
        voter_history (list): Parsed voting history
        
    Returns:
        dict: Temporal pattern features
    """
    
    # Sort history chronologically
    sorted_history = sorted(voter_history, key=lambda x: x.get('year', 0))
    
    # Get total history length in years
    if len(sorted_history) < 2:  # SAME AS BEFORE
        return {"behavior_change_features": {}}
    
    first_year = min(e.get('year', 9999) for e in sorted_history)
    last_year = max(e.get('year', 0) for e in sorted_history)
    history_span = last_year - first_year + 1
    
    # ONLY CHANGE: Split history into periods - LOWERED FROM 6 TO 3
    if history_span >= 3:  # CHANGED FROM >= 6
        early_cutoff = first_year + history_span//3
        recent_cutoff = last_year - history_span//3
        
        early_history = [e for e in sorted_history if e.get('year', 0) <= early_cutoff]
        middle_history = [e for e in sorted_history if early_cutoff < e.get('year', 0) < recent_cutoff]
        recent_history = [e for e in sorted_history if e.get('year', 0) >= recent_cutoff]
        
        # Calculate period participation rates
        early_rate = len(early_history) / max(1, early_cutoff - first_year + 1)
        middle_rate = len(middle_history) / max(1, recent_cutoff - early_cutoff)
        recent_rate = len(recent_history) / max(1, last_year - recent_cutoff + 1)
        
        # Calculate acceleration/deceleration
        early_to_middle_change = middle_rate - early_rate
        middle_to_recent_change = recent_rate - middle_rate
        overall_acceleration = middle_to_recent_change - early_to_middle_change
        
        return {
            "behavior_change_features": {
                "early_participation_rate": early_rate,
                "middle_participation_rate": middle_rate,
                "recent_participation_rate": recent_rate,
                "early_to_middle_change": early_to_middle_change,
                "middle_to_recent_change": middle_to_recent_change,
                "participation_acceleration": overall_acceleration
            }
        }
    else:
        # For shorter histories, use simpler first half vs second half
        midpoint = first_year + history_span//2
        first_half = [e for e in sorted_history if e.get('year', 0) <= midpoint]
        second_half = [e for e in sorted_history if e.get('year', 0) > midpoint]
        
        first_half_rate = len(first_half) / max(1, midpoint - first_year + 1)
        second_half_rate = len(second_half) / max(1, last_year - midpoint)
        participation_change = second_half_rate - first_half_rate
        
        return {
            "behavior_change_features": {
                "first_half_rate": first_half_rate,
                "second_half_rate": second_half_rate,
                "participation_change": participation_change,
                # ADDED: Return the expected features with 0 values
                "early_to_middle_change": 0,
                "middle_to_recent_change": 0,
                "participation_acceleration": participation_change
            }
        }

def extract_sequence_patterns(voter_history):
    """
    Extract patterns in voting sequence (gaps, streaks).
    
    Args:
        voter_history (list): Parsed voting history
        
    Returns:
        dict: Sequence pattern features
    """
    
    if len(voter_history) < 2:  # CHANGED FROM < 3
        return {"sequence_features": {
            "avg_gap": 0,
            "gap_variability": 0,
            "longest_participation_streak": len(voter_history),
            "recent_pattern_change": 0
        }}
    
    # Sort by year
    sorted_history = sorted(voter_history, key=lambda x: x.get('year', 0))
    
    # Find consecutive participation and gaps
    years = [e.get('year', 0) for e in sorted_history]
    
    # Calculate gaps between votes
    gaps = []
    for i in range(1, len(years)):
        gaps.append(years[i] - years[i-1])
    
    # Identify streaks and interruptions
    longest_streak = 1
    current_streak = 1
    
    for i in range(1, len(years)):
        if years[i] == years[i-1] + 1:  # Consecutive years
            current_streak += 1
        else:
            current_streak = 1
        
        longest_streak = max(longest_streak, current_streak)
    
    # Calculate recency of participation pattern changes
    recent_pattern_change = 0
    if len(gaps) >= 2:  # CHANGED FROM >= 3
        # Check if recent gaps are different from earlier gaps
        avg_early_gap = sum(gaps[:-1]) / max(1, len(gaps)-1)  # FIXED DIVISION
        recent_gap = gaps[-1]  # Most recent gap
        
        # Significant change in voting frequency
        if abs(recent_gap - avg_early_gap) > 1:
            recent_pattern_change = 1
    
    import numpy as np
    return {
        "sequence_features": {
            "avg_gap": sum(gaps) / max(1, len(gaps)),
            "gap_variability": np.std(gaps) if len(gaps) > 1 else 0,
            "longest_participation_streak": longest_streak,
            "recent_pattern_change": recent_pattern_change
        }
    }

def detect_behavior_change_points(voter_history):
    """
    Detect significant changes in voting behavior.
    
    Args:
        voter_history (list): Parsed voting history
        
    Returns:
        dict: Behavior change features
    """
    
    if len(voter_history) < 3:  # CHANGED FROM < 5
        return {"change_point_features": {
            "has_behavior_change": 0,
            "change_count": 0,
            "years_since_last_change": 0,
            "change_direction": 0,
            "recent_change_magnitude": 0
        }}
    
    # Sort by year
    sorted_history = sorted(voter_history, key=lambda x: x.get('year', 0))
    years = [e.get('year', 0) for e in sorted_history]
    
    # Create a binary participation array by year
    all_years = list(range(min(years), max(years) + 1))
    participation = [1 if year in years else 0 for year in all_years]
    
    # Use rolling windows to detect changes
    window_size = 2  # CHANGED FROM 3
    if len(participation) < 2 * window_size:
        return {"change_point_features": {
            "has_behavior_change": 0,
            "change_count": 0,
            "years_since_last_change": 0,
            "change_direction": 0,
            "recent_change_magnitude": 0
        }}
    
    change_points = []
    for i in range(window_size, len(participation) - window_size + 1):
        before_rate = sum(participation[i-window_size:i]) / window_size
        after_rate = sum(participation[i:i+window_size]) / window_size
        
        # Significant change in participation rate
        if abs(after_rate - before_rate) > 0.3:  # SAME THRESHOLD
            change_points.append(all_years[i])
    
    # Calculate features based on change points
    has_behavior_change = int(len(change_points) > 0)
    most_recent_change = max(change_points) if change_points else 0
    years_since_change = 2025 - most_recent_change if most_recent_change else 0
    
    # Determine direction of most recent change
    change_direction = 0
    change_magnitude = 0  # ADDED THIS VARIABLE
    if most_recent_change:
        idx = all_years.index(most_recent_change)
        before_rate = sum(participation[idx-window_size:idx]) / window_size
        after_rate = sum(participation[idx:idx+window_size]) / window_size
        change_direction = 1 if after_rate > before_rate else -1
        change_magnitude = abs(after_rate - before_rate)  # CALCULATE MAGNITUDE
    
    return {
        "change_point_features": {
            "has_behavior_change": has_behavior_change,
            "change_count": len(change_points),
            "years_since_last_change": years_since_change,
            "change_direction": change_direction,
            "recent_change_magnitude": change_magnitude
        }
    }

def extract_clean_features_enhanced(df, parse_history_function, mayoral_dates=None):
    """
    Extract enhanced behavioral features from voter history.
    
    Args:
        df (DataFrame): Voter data with 'voterhistory' column
        parse_history_function (function): Function to parse histories
        mayoral_dates (list, optional): Mayoral election dates
        
    Returns:
        tuple: (DataFrame with enhanced features, list of feature columns)
    """
    
    # Call YOUR ORIGINAL extract_clean_features - NO CHANGES
    result_df, basic_features = extract_clean_features(df, parse_history_function, mayoral_dates)
    
    # Add enhanced features for behavior change detection
    print("Extracting behavior change features...")
    
    # Temporal features with lowered thresholds
    temporal_features = result_df['parsed_history'].apply(extract_temporal_features)
    for feature in ['early_to_middle_change', 'middle_to_recent_change', 'participation_acceleration']:
        result_df[feature] = temporal_features.apply(
            lambda x: x.get('behavior_change_features', {}).get(feature, 0)
        )
    
    # Sequence pattern features
    sequence_features = result_df['parsed_history'].apply(extract_sequence_patterns)
    for feature in ['avg_gap', 'gap_variability', 'longest_participation_streak', 'recent_pattern_change']:
        result_df[feature] = sequence_features.apply(
            lambda x: x.get('sequence_features', {}).get(feature, 0)
        )
    
    # Transition features
    transition_features = result_df['parsed_history'].apply(extract_transition_features)
    for feature in ['participation_expansion', 'diversification_score']:
        result_df[feature] = transition_features.apply(
            lambda x: x.get('transition_features', {}).get(feature, 0)
        )
    
    # Change point detection
    change_point_features = result_df['parsed_history'].apply(detect_behavior_change_points)
    for feature in ['has_behavior_change', 'years_since_last_change', 'change_direction']:
        result_df[feature] = change_point_features.apply(
            lambda x: x.get('change_point_features', {}).get(feature, 0)
        )
    
    # Combine all features - basic + enhanced
    enhanced_features = basic_features + [
        'early_to_middle_change', 'middle_to_recent_change', 'participation_acceleration',
        'avg_gap', 'gap_variability', 'longest_participation_streak', 'recent_pattern_change',
        'participation_expansion', 'diversification_score',
        'has_behavior_change', 'years_since_last_change', 'change_direction'
    ]
    
    return result_df, enhanced_features
def extract_transition_features(voter_history):
    """
    Extract features about transitions between election types.
    
    Args:
        voter_history (list): Parsed voting history
        
    Returns:
        dict: Transition pattern features
    """
    
    if len(voter_history) < 2:
        return {"transition_features": {}}
    
    # Sort by year
    sorted_history = sorted(voter_history, key=lambda x: x.get('year', 0))
    
    # Define event types
    def get_event_type(event):
        if event.get('is_primary', False):
            return 'primary'
        elif event.get('is_local_only', False):
            if event.get('year', 0) % 4 == 1:  # Year after presidential
                return 'mayoral'
            else:
                return 'local'
        elif event.get('year', 0) % 4 == 0:  # Presidential
            return 'presidential'
        elif event.get('is_general', False):
            if event.get('year', 0) % 4 == 2:  # Midterm
                return 'midterm'
            else:
                return 'general'
        else:
            return 'other'
    
    # Get event types in sequence
    event_types = [get_event_type(e) for e in sorted_history]
    
    # Calculate transitions
    primary_to_general = 0
    local_to_mayoral = 0
    midterm_to_presidential = 0
    
    for i in range(1, len(event_types)):
        if event_types[i-1] == 'primary' and event_types[i] in ['mayoral', 'presidential', 'midterm', 'general']:
            primary_to_general += 1
        
        if event_types[i-1] == 'local' and event_types[i] == 'mayoral':
            local_to_mayoral += 1
            
        if event_types[i-1] == 'midterm' and event_types[i] == 'presidential':
            midterm_to_presidential += 1
    
    # Calculate expansion of participation (moving from one type to multiple types)
    unique_types_first_half = len(set(event_types[:len(event_types)//2]))
    unique_types_second_half = len(set(event_types[len(event_types)//2:]))
    participation_expansion = unique_types_second_half - unique_types_first_half
    
    # Ensure non-zero denominator for ratios
    primary_count = event_types.count('primary')
    local_count = event_types.count('local')
    midterm_count = event_types.count('midterm')
    
    return {
        "transition_features": {
            "primary_to_general_ratio": primary_to_general / max(1, primary_count),
            "local_to_mayoral_ratio": local_to_mayoral / max(1, local_count),
            "midterm_to_presidential_ratio": midterm_to_presidential / max(1, midterm_count),
            "participation_expansion": participation_expansion,
            "diversification_score": unique_types_second_half / max(1, unique_types_first_half)
        }
    }

def extract_temporal_features(voter_history):
    """
    Extract patterns in the sequence of participation.
    
    Analyzes gaps between votes, participation streaks, and changes in
    voting frequency.
    
    Args:
        voter_history (list): List of parsed voting history events
        
    Returns:
        dict: Dictionary with sequence pattern features
    """
    
    # Sort history chronologically
    sorted_history = sorted(voter_history, key=lambda x: x.get('year', 0))
    
    # Get total history length in years
    if len(sorted_history) < 2:  # SAME AS BEFORE
        return {"behavior_change_features": {}}
    
    first_year = min(e.get('year', 9999) for e in sorted_history)
    last_year = max(e.get('year', 0) for e in sorted_history)
    history_span = last_year - first_year + 1
    
    # ONLY CHANGE: Split history into periods - LOWERED FROM 6 TO 3
    if history_span >= 3:  # CHANGED FROM >= 6
        early_cutoff = first_year + history_span//3
        recent_cutoff = last_year - history_span//3
        
        early_history = [e for e in sorted_history if e.get('year', 0) <= early_cutoff]
        middle_history = [e for e in sorted_history if early_cutoff < e.get('year', 0) < recent_cutoff]
        recent_history = [e for e in sorted_history if e.get('year', 0) >= recent_cutoff]
        
        # Calculate period participation rates
        early_rate = len(early_history) / max(1, early_cutoff - first_year + 1)
        middle_rate = len(middle_history) / max(1, recent_cutoff - early_cutoff)
        recent_rate = len(recent_history) / max(1, last_year - recent_cutoff + 1)
        
        # Calculate acceleration/deceleration
        early_to_middle_change = middle_rate - early_rate
        middle_to_recent_change = recent_rate - middle_rate
        overall_acceleration = middle_to_recent_change - early_to_middle_change
        
        return {
            "behavior_change_features": {
                "early_participation_rate": early_rate,
                "middle_participation_rate": middle_rate,
                "recent_participation_rate": recent_rate,
                "early_to_middle_change": early_to_middle_change,
                "middle_to_recent_change": middle_to_recent_change,
                "participation_acceleration": overall_acceleration
            }
        }
    else:
        # For shorter histories, use simpler first half vs second half
        midpoint = first_year + history_span//2
        first_half = [e for e in sorted_history if e.get('year', 0) <= midpoint]
        second_half = [e for e in sorted_history if e.get('year', 0) > midpoint]
        
        first_half_rate = len(first_half) / max(1, midpoint - first_year + 1)
        second_half_rate = len(second_half) / max(1, last_year - midpoint)
        participation_change = second_half_rate - first_half_rate
        
        return {
            "behavior_change_features": {
                "first_half_rate": first_half_rate,
                "second_half_rate": second_half_rate,
                "participation_change": participation_change,
                # ADDED: Return the expected features with 0 values
                "early_to_middle_change": 0,
                "middle_to_recent_change": 0,
                "participation_acceleration": participation_change
            }
        }

def extract_sequence_patterns(voter_history):
    """
    Detect significant changes in voting behavior.
    
    Uses rolling window analysis to identify points where participation
    patterns significantly changed.
    
    Args:
        voter_history (list): List of parsed voting history events
        
    Returns:
        dict: Dictionary with behavior change features
    """
    
    if len(voter_history) < 2:  # CHANGED FROM < 3
        return {"sequence_features": {
            "avg_gap": 0,
            "gap_variability": 0,
            "longest_participation_streak": len(voter_history),
            "recent_pattern_change": 0
        }}
    
    # Sort by year
    sorted_history = sorted(voter_history, key=lambda x: x.get('year', 0))
    
    # Find consecutive participation and gaps
    years = [e.get('year', 0) for e in sorted_history]
    
    # Calculate gaps between votes
    gaps = []
    for i in range(1, len(years)):
        gaps.append(years[i] - years[i-1])
    
    # Identify streaks and interruptions
    longest_streak = 1
    current_streak = 1
    
    for i in range(1, len(years)):
        if years[i] == years[i-1] + 1:  # Consecutive years
            current_streak += 1
        else:
            current_streak = 1
        
        longest_streak = max(longest_streak, current_streak)
    
    # Calculate recency of participation pattern changes
    recent_pattern_change = 0
    if len(gaps) >= 2:  # CHANGED FROM >= 3
        # Check if recent gaps are different from earlier gaps
        avg_early_gap = sum(gaps[:-1]) / max(1, len(gaps)-1)  # FIXED DIVISION
        recent_gap = gaps[-1]  # Most recent gap
        
        # Significant change in voting frequency
        if abs(recent_gap - avg_early_gap) > 1:
            recent_pattern_change = 1
    
    import numpy as np
    return {
        "sequence_features": {
            "avg_gap": sum(gaps) / max(1, len(gaps)),
            "gap_variability": np.std(gaps) if len(gaps) > 1 else 0,
            "longest_participation_streak": longest_streak,
            "recent_pattern_change": recent_pattern_change
        }
    }

def detect_behavior_change_points(voter_history):
    """
    Enhanced feature extraction - adds behavioral pattern analysis features.
    
    Extends the basic feature extraction with advanced behavioral patterns,
    temporal trends, and transition analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with voter data including 'voterhistory' column
        parse_history_function (function): Function to parse voter history strings
        mayoral_dates (list, optional): List of mayoral election dates
        
    Returns:
        tuple: (DataFrame with enhanced features, list of feature column names)
    """
    
    if len(voter_history) < 3:  # CHANGED FROM < 5
        return {"change_point_features": {
            "has_behavior_change": 0,
            "change_count": 0,
            "years_since_last_change": 0,
            "change_direction": 0,
            "recent_change_magnitude": 0
        }}
    
    # Sort by year
    sorted_history = sorted(voter_history, key=lambda x: x.get('year', 0))
    years = [e.get('year', 0) for e in sorted_history]
    
    # Create a binary participation array by year
    all_years = list(range(min(years), max(years) + 1))
    participation = [1 if year in years else 0 for year in all_years]
    
    # Use rolling windows to detect changes
    window_size = 2  # CHANGED FROM 3
    if len(participation) < 2 * window_size:
        return {"change_point_features": {
            "has_behavior_change": 0,
            "change_count": 0,
            "years_since_last_change": 0,
            "change_direction": 0,
            "recent_change_magnitude": 0
        }}
    
    change_points = []
    for i in range(window_size, len(participation) - window_size + 1):
        before_rate = sum(participation[i-window_size:i]) / window_size
        after_rate = sum(participation[i:i+window_size]) / window_size
        
        # Significant change in participation rate
        if abs(after_rate - before_rate) > 0.3:  # SAME THRESHOLD
            change_points.append(all_years[i])
    
    # Calculate features based on change points
    has_behavior_change = int(len(change_points) > 0)
    most_recent_change = max(change_points) if change_points else 0
    years_since_change = 2025 - most_recent_change if most_recent_change else 0
    
    # Determine direction of most recent change
    change_direction = 0
    change_magnitude = 0  # ADDED THIS VARIABLE
    if most_recent_change:
        idx = all_years.index(most_recent_change)
        before_rate = sum(participation[idx-window_size:idx]) / window_size
        after_rate = sum(participation[idx:idx+window_size]) / window_size
        change_direction = 1 if after_rate > before_rate else -1
        change_magnitude = abs(after_rate - before_rate)  # CALCULATE MAGNITUDE
    
    return {
        "change_point_features": {
            "has_behavior_change": has_behavior_change,
            "change_count": len(change_points),
            "years_since_last_change": years_since_change,
            "change_direction": change_direction,
            "recent_change_magnitude": change_magnitude
        }
    }