#!/usr/bin/env python3
"""
Script to analyze the relationship between session and ID in a nested data structure.
This script processes one entry at a time and shows how many sessions each ID appears in.
"""

import json
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


def find_keys(obj, target_key, path=None, results=None):
    """Recursively search for keys in a nested dictionary/list structure"""
    if path is None:
        path = []
    if results is None:
        results = []
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                results.append((path + [k], v))
            find_keys(v, target_key, path + [k], results)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_keys(item, target_key, path + [f"[{i}]"], results)
            
    return results


def process_item(item, item_index=None):
    """Process a single item from the info structure to find ID-session relationships"""
    print(f"\nProcessing item {item_index if item_index is not None else ''}")
    
    # Create mapping for IDs to sessions
    id_to_sessions = defaultdict(set)
    
    # Find all session and id keys
    session_results = find_keys(item, 'session')
    id_results = find_keys(item, 'id')
    
    print(f"Found {len(session_results)} 'session' keys and {len(id_results)} 'id' keys")
    
    # Try to find relationships based on path proximity
    for id_path, id_value in id_results:
        id_path_str = '.'.join(map(str, id_path[:-1]))
        id_value_str = str(id_value)
        
        for session_path, session_value in session_results:
            session_path_str = '.'.join(map(str, session_path[:-1]))
            session_value_str = str(session_value)
            
            # Check if they might be related (in same object or parent/child relationship)
            common_prefix = os.path.commonprefix([id_path_str, session_path_str])
            if common_prefix and len(common_prefix) > 1:  # They share a common parent
                id_to_sessions[id_value_str].add(session_value_str)
    
    # Also look for direct relationships in common patterns
    if isinstance(item, dict):
        # Pattern: {'cells': [{'id': 1, 'session': 'A'}, {'id': 2, 'session': 'B'}]}
        for key, value in item.items():
            if key in ['cells', 'components', 'items'] and isinstance(value, list):
                for component in value:
                    if isinstance(component, dict) and 'id' in component and 'session' in component:
                        id_to_sessions[str(component['id'])].add(str(component['session']))
    
    return id_to_sessions


def analyze_id_session_counts(mapping):
    """Analyze and display counts of sessions per ID"""
    if not mapping:
        print("No ID-session relationships found in this item.")
        return {}
    
    print(f"\nFound {len(mapping)} unique IDs across multiple sessions")
    
    # Count sessions per ID and show IDs with multiple sessions
    multi_session_ids = {}
    for id_val, sessions in mapping.items():
        if len(sessions) > 1:
            multi_session_ids[id_val] = sessions
    
    if multi_session_ids:
        print(f"\n{len(multi_session_ids)} IDs appear in multiple sessions:")
        for id_val, sessions in sorted(multi_session_ids.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"ID {id_val}: {len(sessions)} sessions - {sorted(sessions)}")
    else:
        print("No IDs appear in multiple sessions in this item.")
    
    return mapping


def visualize_results(all_mappings):
    """Create visualizations for the aggregated results"""
    if not all_mappings:
        print("No data to visualize.")
        return
    
    # Combine all mappings
    combined_mapping = defaultdict(set)
    for mapping in all_mappings:
        for id_val, sessions in mapping.items():
            combined_mapping[id_val].update(sessions)
    
    # Calculate session counts per ID
    session_counts = [len(sessions) for sessions in combined_mapping.values()]
    
    # Create a histogram
    plt.figure(figsize=(10, 6))
    bins = range(1, max(session_counts) + 2) if session_counts else [0, 1]
    plt.hist(session_counts, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Sessions')
    plt.ylabel('Number of IDs')
    plt.title('Distribution of Number of Sessions per ID')
    plt.grid(axis='y', alpha=0.75)
    if session_counts:
        plt.xticks(range(1, max(session_counts) + 1))
    plt.savefig('session_counts_histogram.png')
    print("Saved histogram to session_counts_histogram.png")
    
    # Count distribution
    count_distribution = defaultdict(int)
    for count in session_counts:
        count_distribution[count] += 1
    
    # Print statistics
    print("\nDistribution of session counts:")
    for count, num_ids in sorted(count_distribution.items()):
        print(f"{count} session(s): {num_ids} IDs")
    
    if session_counts:
        avg_count = sum(session_counts) / len(session_counts)
        max_count = max(session_counts)
        min_count = min(session_counts)
        
        print("\nStatistics:")
        print(f"Average sessions per ID: {avg_count:.2f}")
        print(f"Maximum sessions per ID: {max_count}")
        print(f"Minimum sessions per ID: {min_count}")
    
    # Unique sessions
    all_sessions = set()
    for sessions in combined_mapping.values():
        all_sessions.update(sessions)
    
    print(f"Total unique IDs: {len(combined_mapping)}")
    print(f"Total unique sessions: {len(all_sessions)}")


def main():
    """Main function to process the info variable"""
    print("Session-ID Analysis Script")
    print("=========================")
    
    # Get the path to the zarr file from arguments or use default
    zarr_path = sys.argv[1] if len(sys.argv) > 1 else '/nrs/spruston/Gaby_imaging/raw/M54/multi_day_demix/vr2p.zarr'
    
    print(f"Opening zarr file: {zarr_path}")
    import zarr
    
    try:
        # Open the zarr file
        z = zarr.open(zarr_path, mode='r')
        
        # Try to get the info variable
        # The actual path will depend on your zarr structure
        # You may need to adjust this based on the exploration results
        print("Looking for info in the zarr file...")
        
        # Option 1: Try directly accessing info if it's at the root
        info = None
        try:
            if 'info' in z:
                info = z['info'][:]
                print("Found info at root level")
        except:
            pass
        
        # Option 2: Look for info in other common locations
        if info is None:
            try:
                for key in z.keys():
                    if 'info' in z[key]:
                        info = z[key]['info'][:]
                        print(f"Found info in {key}/info")
                        break
            except:
                pass
        
        # If info still not found, let user provide interactive path
        if info is None:
            print("Could not automatically find info in the zarr file.")
            print("Please provide the path to info (e.g., 'group/subgroup/info'):")
            info_path = input("> ")
            
            # Split the path and navigate through zarr
            path_parts = info_path.strip('/').split('/')
            current = z
            for part in path_parts:
                current = current[part]
            
            info = current[:]
            print(f"Found info at {info_path}")
        
        # Process each item individually
        if isinstance(info, list):
            print(f"Info contains {len(info)} items")
            
            # Ask if user wants to process all items or specific ones
            print("\nOptions:")
            print("1. Process all items at once")
            print("2. Process items one by one")
            print("3. Process a specific item by index")
            choice = input("Enter your choice (1-3): ")
            
            all_mappings = []
            
            if choice == '1':
                # Process all items at once
                print(f"Processing all {len(info)} items...")
                for i, item in enumerate(info):
                    mapping = process_item(item, i)
                    id_session_map = analyze_id_session_counts(mapping)
                    if id_session_map:
                        all_mappings.append(id_session_map)
                
                # Visualize combined results
                visualize_results(all_mappings)
                
            elif choice == '2':
                # Process items one by one
                print("Processing items one by one...")
                for i, item in enumerate(info):
                    print(f"\n===== Item {i} =====")
                    mapping = process_item(item, i)
                    id_session_map = analyze_id_session_counts(mapping)
                    
                    if id_session_map:
                        all_mappings.append(id_session_map)
                    
                    if i < len(info) - 1:
                        cont = input("\nPress Enter to continue to next item, or 'q' to quit: ")
                        if cont.lower() == 'q':
                            break
                
                # After processing, visualize combined results
                visualize_results(all_mappings)
                
            elif choice == '3':
                # Process a specific item
                item_index = int(input(f"Enter item index (0-{len(info)-1}): "))
                if 0 <= item_index < len(info):
                    item = info[item_index]
                    mapping = process_item(item, item_index)
                    analyze_id_session_counts(mapping)
                    if mapping:
                        all_mappings.append(mapping)
                else:
                    print("Invalid index")
            
            else:
                print("Invalid choice")
                
        elif isinstance(info, dict):
            print("Info is a dictionary (single item)")
            mapping = process_item(info)
            analyze_id_session_counts(mapping)
            
        else:
            print(f"Info has unexpected type: {type(info)}")
        
    except Exception as e:
        print(f"Error processing zarr file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()