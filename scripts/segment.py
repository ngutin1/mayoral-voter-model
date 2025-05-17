"""
segment.py - Voter Segmentation Module

This module provides functionality to segment voters into actionable campaign targeting 
groups based on their predicted turnout probabilities for mayoral elections. It helps
campaign teams prioritize resources and tailor outreach strategies for maximum impact.

Author: Nicholas Gutin
Date: May 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def segment_voters_for_campaign(voter_df):
    """
    Segment voters into actionable campaign targeting groups based on turnout probabilities.
    
    Args:
        voter_df (pandas.DataFrame): Voter data with turnout_probability column (0-1 values)
    
    Returns:
        pandas.DataFrame: Original DataFrame with added campaign segments and targeting priorities
        
    Notes:
        - HIGH_PROPENSITY (85%+): Reliable voters needing voter protection
        - LIKELY (65-85%): Strong voters targeted for GOTV efforts
        - PERSUADABLE (35-65%): Key targets for persuasion campaigns
        - LOW_PROPENSITY (<35%): Low-engagement voters receiving minimal outreach
    """
    # Create a copy to avoid modifying the original
    df = voter_df.copy()
    
    # 1. Primary segmentation based on turnout probability
    conditions = [
        (df['turnout_probability'] >= 0.85),                               # High certainty voters
        (df['turnout_probability'] >= 0.65) & (df['turnout_probability'] < 0.85),  # Likely voters
        (df['turnout_probability'] >= 0.35) & (df['turnout_probability'] < 0.65),  # Persuadable voters
        (df['turnout_probability'] < 0.35)                                # Low propensity voters
    ]
    
    segments = ['HIGH_PROPENSITY', 'LIKELY', 'PERSUADABLE', 'LOW_PROPENSITY']
    df['segment'] = np.select(conditions, segments, default='UNKNOWN')
    
    # 2. Tactical assignment based on segments
    tactical_assignments = {
        'HIGH_PROPENSITY': 'VOTER_PROTECTION',     # Ensure these reliable voters can vote
        'LIKELY': 'CANVASS',                          # Get Out The Vote calls/texts
        'PERSUADABLE': 'CANVASS',                  # Door-to-door canvassing priority
        'LOW_PROPENSITY': 'MAIL_ONLY'              # Mail programs only
    }
    df['primary_tactic'] = df['segment'].map(tactical_assignments)
    
    
    # 3. Calculate targeting score for prioritization within segments
    # Different logic for each segment:
    
    # For high propensity: prioritize highest probability voters for protection
    high_mask = df['segment'] == 'HIGH_PROPENSITY'
    df.loc[high_mask, 'targeting_score'] = df.loc[high_mask, 'turnout_probability']
    
    # For likely voters: prioritize slightly less certain for GOTV effect
    likely_mask = df['segment'] == 'LIKELY'
    df.loc[likely_mask, 'targeting_score'] = 2 - df.loc[likely_mask, 'turnout_probability']
    
    # For persuadable: middle probabilities get highest priority (most movable)
    persuadable_mask = df['segment'] == 'PERSUADABLE'
    df.loc[persuadable_mask, 'targeting_score'] = 1 - abs(0.5 - df.loc[persuadable_mask, 'turnout_probability']) * 2
    
    # For low propensity: higher scores among this group get priority
    low_mask = df['segment'] == 'LOW_PROPENSITY'
    df.loc[low_mask, 'targeting_score'] = df.loc[low_mask, 'turnout_probability'] * 2
    
    # 4. Additional targeting flags for specialized outreach
    
    # Early Vote targeting (subset of likely voters who should be pushed to vote early)
    early_vote_conditions = (
        (df['segment'].isin(['LIKELY', 'PERSUADABLE'])) &
        ((df['prefers_early_voting'] == 1) if 'prefers_early_voting' in df.columns else True)
    )
    df['target_early_vote'] = early_vote_conditions.astype(int)
    
    # Inconsistent voter flag (for specialized messaging)
    if 'has_behavior_change' in df.columns and 'voting_trend' in df.columns:
        df['inconsistent_voter'] = ((df['has_behavior_change'] == 1) | 
                                    (df['voting_trend'] != 0)).astype(int)
    else:
        # Proxy based on probability if behavioral features not available
        df['inconsistent_voter'] = ((df['turnout_probability'] > 0.3) & 
                                    (df['turnout_probability'] < 0.7)).astype(int)
    
    # 5. Sort within each segment by targeting score
    df = df.sort_values(['segment', 'targeting_score'], ascending=[True, False])
    
    return df