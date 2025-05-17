from feature_eng import *

def working_backtest(voter_df, model, feature_cols, test_years=[2017, 2013], mayoral_election_dates=None):
    """
    Fixed backtest that properly handles all voters and features
    """
    if mayoral_election_dates is None:
        mayoral_election_dates = [
            '20211102',
            '20171107', 
            '20131105',
            '20091103',
            '20051108',
            '20011106'
        ]
    
    results = {}
    
    for year in test_years:
        print(f"\n=== Backtesting {year} ===")
        
        # 1. Create a working copy and preserve original index
        working_df = voter_df.copy()
        working_df['original_index'] = working_df.index
        
        # 2. Determine actual turnout for ALL voters
        actual_votes = np.zeros(len(working_df))

        for i, row in enumerate(working_df.itertuples(index=False)):
            history = parse_voter_history(row.voterhistory)
            for event in history:
                if event['year'] == year:
                    actual_votes[i] = 1
                    break  # Found a match

        working_df['actual_turnout'] = actual_votes
        actual_count = sum(actual_votes)
        print(f"Actual voters: {actual_count}/{len(working_df)} = {actual_count/len(working_df):.1%}")
        
        # 3. Filter histories to before test year
        filtered_histories = []
        voters_with_history = 0
        
        for idx, row in working_df.iterrows():
            history = parse_voter_history(row['voterhistory'])
            past_events = [e for e in history if e['year'] < year]
            
            if past_events:
                voters_with_history += 1
                history_str = ';'.join([
                    f"{e['year']} {e['election_name']}({e['vote_method']})"
                    for e in past_events
                ])
            else:
                history_str = ""
            
            filtered_histories.append(history_str)
        
        working_df['voterhistory'] = filtered_histories
        print(f"Voters with history before {year}: {voters_with_history}")
        
        # 4. Get mayoral dates before test year
        past_mayoral_dates = [d for d in mayoral_election_dates if int(d[:4]) < year]
        print(f"Using mayoral dates: {[d[:4] for d in past_mayoral_dates]}")
        
        # 5. Extract features - this will filter to valid histories
        features_df, returned_feature_names = extract_clean_features_enhanced(
            working_df,
            parse_voter_history,
            past_mayoral_dates
        )
        
        print(f"Features extracted for: {len(features_df)} voters")
        print(f"Returned features: {len(returned_feature_names)}")
        
        # 6. Create feature matrix using the features we want to use for prediction
        if not set(feature_cols).issubset(set(features_df.columns)):
            missing = set(feature_cols) - set(features_df.columns)
            print(f"WARNING: Missing features: {missing}")
        
        X = features_df[feature_cols].fillna(0)
        y_true = working_df.loc[features_df.index, 'actual_turnout']
        
        print(f"Final prediction matrix shape: {X.shape}")
        print(f"Eligible voters: {len(X)}")
        print(f"Eligible turnout: {y_true.mean():.1%}")
        
        # 7. Make predictions
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        # 8. Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        
        
        # 9. Segment analysis
        segment_df = pd.DataFrame({
            'actual': y_true,
            'predicted_proba': y_proba
        })
        
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        segment_df['segment'] = pd.cut(y_proba, bins=bins, labels=['Low', 'Med-Low', 'Med-High', 'High'])
        
        segment_stats = segment_df.groupby('segment').agg({
            'actual': ['mean', 'count'],
            'predicted_proba': 'mean'
        })
        
        # Calculate projected turnout for each segment
        segment_stats['projected_voters'] = segment_stats[('predicted_proba', 'mean')] * segment_stats[('actual', 'count')]

        # Calculate total projected turnout
        total_voters = segment_stats[('actual', 'count')].sum()
        total_projected_turnout = segment_stats['projected_voters'].sum()
        projected_turnout_rate = total_projected_turnout / total_voters
        
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC: {auc:.3f}")
        print(f"Actual turnout: {y_true.mean():.1%}")
        print(f"Predicted turnout: {projected_turnout_rate:.1%}")
        print(f"Error: {abs(y_true.mean() - projected_turnout_rate):.1%}")
        
        print("\nSegment Analysis:")
        print(segment_stats)
        
        # Store results
        results[year] = {
            'accuracy': accuracy,
            'auc': auc,
            'actual_turnout': y_true.mean(),
            'predicted_turnout': y_proba.mean(),
            'segment_stats': segment_stats,
            'predictions': y_proba,
            'actuals': y_true
        }
    
    return results