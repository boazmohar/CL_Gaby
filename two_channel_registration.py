#!/usr/bin/env python
"""
Two-channel registration script for aligning red and green channel images from Suite2p.
Based on the separate_twochan.ipynb notebook.
"""

import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import contextlib
from suite2p import default_ops
from suite2p.io import tiff
from suite2p.registration import register
from suite2p import io
from suite2p.detection import chan2detect
from natsort import natsorted

def normalize99(img):
    """Normalize image values between 0 and 1 based on 1st and 99th percentiles"""
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    X = np.clip(X, 0, 1)
    return X

def convert_to_binary(data_path, save_path=None, mesoscan=False):
    """
    Convert tiff files to binary format for suite2p processing.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the directory containing the tiff files
    save_path : str or Path, optional
        Path to save the binary files, defaults to data_path
    mesoscan : bool, optional
        Whether the data is from a mesoscope
        
    Returns:
    --------
    dict
        Updated operations dictionary
    """
    print(f"Converting tiffs to binary files from: {data_path}")
    
    ops = default_ops()
    ops["data_path"] = [data_path]
    ops["nchannels"] = 2
    ops["save_path0"] = str(save_path) if save_path else str(data_path)
    
    # For mesoscan, need to find the required parameters in the JSON files
    if mesoscan:
        # Try to find JSON file for metadata
        import glob
        import json
        fpath = os.path.join(data_path, "*json")
        fs = glob.glob(fpath)
        
        if len(fs) == 0:
            raise ValueError(f"No JSON files found in {data_path}. Mesoscan requires JSON metadata files.")
            
        print(f"Found JSON file: {fs[0]}")
        with open(fs[0], "r") as f:
            opsj = json.load(f)
        
        # Set required mesoscan parameters from JSON
        ops["mesoscan"] = 1
        
        # Check for required parameters
        required_params = ["lines", "dy", "dx"]
        missing_params = [param for param in required_params if param not in opsj]
        
        if missing_params:
            raise ValueError(f"Missing required parameters in JSON file: {missing_params}")
        
        # Copy required parameters from JSON
        ops["lines"] = opsj["lines"]
        ops["dy"] = opsj["dy"]
        ops["dx"] = opsj["dx"]
        
        # Other important parameters
        if "nrois" in opsj:
            ops["nrois"] = opsj["nrois"]
        if "nplanes" in opsj:
            ops["nplanes"] = opsj["nplanes"]
        if "fs" in opsj:
            ops["fs"] = opsj["fs"]
            
        # Now run mesoscan_to_binary
        ops = tiff.mesoscan_to_binary(ops)
    else:
        ops = tiff.tiff_to_binary(ops)
    
    return ops

def register_channels(data_path, ref_path, nplanes=1, align_by_chan2=False, suite2p_folder=None):
    """
    Register channels in a two-channel recording.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the directory containing the binary files
    ref_path : str or Path
        Path to the directory containing the reference images
    nplanes : int, optional
        Number of planes to process
    align_by_chan2 : bool, optional
        Whether to align by channel 2 (red) instead of channel 1 (green)
    suite2p_folder : str, optional
        Name of the suite2p output folder. If None, will try "suite2p_done" then "suite2p"
        
    Returns:
    --------
    list
        List of updated operations dictionaries for each plane
    """
    data_path = Path(data_path)
    ref_path = Path(ref_path)
    
    print(f"Registering data using data path: {data_path}")
    print(f"Using reference path: {ref_path}")
    
    # Try to find the suite2p folder
    if suite2p_folder is None:
        # Try suite2p_done first, then suite2p
        potential_folders = ["suite2p_done", "suite2p"]
        found_folder = None
        
        for folder in potential_folders:
            if (data_path / folder).exists():
                found_folder = folder
                break
                
        if found_folder is None:
            raise FileNotFoundError(f"Could not find suite2p or suite2p_done folder in {data_path}")
            
        suite2p_folder = found_folder
    else:
        # Verify the specified folder exists
        if not (data_path / suite2p_folder).exists():
            raise FileNotFoundError(f"Specified suite2p folder '{suite2p_folder}' not found in {data_path}")
    
    print(f"Using suite2p folder: {suite2p_folder}")
    
    # Get plane folders
    save_folder = suite2p_folder
    save_path = data_path / save_folder
    plane_folders = natsorted([
        f.path for f in os.scandir(save_path) 
        if f.is_dir() and f.name[:5] == "plane"
    ])
    ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
    
    # Find the suite2p folder in the reference path
    potential_ref_folders = ["suite2p", "suite2p_done"]
    ref_suite2p_folder = None
    
    for folder in potential_ref_folders:
        if (ref_path / folder).exists():
            ref_suite2p_folder = folder
            break
            
    if ref_suite2p_folder is None:
        raise FileNotFoundError(f"Could not find suite2p or suite2p_done folder in reference path {ref_path}")
        
    print(f"Using reference suite2p folder: {ref_suite2p_folder}")
    
    # Get reference images
    ref_ops_paths = [ref_path / ref_suite2p_folder / f"plane{ipl}" / "ops.npy" for ipl in range(nplanes)]
    print(f"Reference ops paths: {ref_ops_paths}")
    
    refImgs = []
    for ops_path in ref_ops_paths:
        if Path(ops_path).exists():
            try:
                ref_ops = np.load(str(ops_path), allow_pickle=True).item()
                refImgs.append(ref_ops["meanImg"])
            except Exception as e:
                print(f"Error loading reference image from {ops_path}: {e}")
        else:
            print(f"Reference path does not exist: {ops_path}")
    
    if len(refImgs) < nplanes:
        raise ValueError(f"Found only {len(refImgs)} reference images, expected {nplanes}")
    
    # Register each plane
    updated_ops_list = []
    for ipl, ops_path in enumerate(ops_paths[:nplanes]):
        print(f"\n>>>> Registering PLANE {ipl}")
        ops = np.load(ops_path, allow_pickle=True).item()
        
        # Get binary file paths
        raw = ops.get("keep_movie_raw") and "raw_file" in ops and os.path.isfile(ops["raw_file"])
        
        # Check if reg_file path exists, if not try to fix it
        reg_file = ops["reg_file"]
        if not os.path.isfile(reg_file):
            # Extract just the filename from the path
            reg_filename = os.path.basename(reg_file)
            # Try to find the file in the current plane folder
            potential_reg_file = os.path.join(os.path.dirname(ops_path), reg_filename)
            if os.path.isfile(potential_reg_file):
                print(f"Found binary file at: {potential_reg_file}")
                reg_file = potential_reg_file
            else:
                # Try looking for data.bin in suite2p_done/suite2p/plane structure
                plane_name = os.path.basename(os.path.dirname(ops_path))
                potential_path = os.path.join(data_path, suite2p_folder, "suite2p", plane_name, reg_filename)
                if os.path.isfile(potential_path):
                    print(f"Found binary file at: {potential_path}")
                    reg_file = potential_path
                else:
                    print(f"WARNING: Could not find binary file {reg_file}")
                    print(f"Searched locations: {potential_reg_file}, {potential_path}")
                    raise FileNotFoundError(f"Could not find binary file for plane {ipl}")
                    
        raw_file = ops.get("raw_file", 0) if raw else reg_file
        
        # Setup channel 2 files
        if ops["nchannels"] > 1:
            reg_file_chan2 = ops["reg_file_chan2"]
            
            # Check if reg_file_chan2 path exists, if not try to fix it
            if not os.path.isfile(reg_file_chan2):
                # Extract just the filename from the path
                reg_chan2_filename = os.path.basename(reg_file_chan2)
                # Try to find the file in the current plane folder
                potential_reg_chan2_file = os.path.join(os.path.dirname(ops_path), reg_chan2_filename)
                if os.path.isfile(potential_reg_chan2_file):
                    print(f"Found binary chan2 file at: {potential_reg_chan2_file}")
                    reg_file_chan2 = potential_reg_chan2_file
                else:
                    # Try looking for data_chan2.bin in suite2p_done/suite2p/plane structure
                    plane_name = os.path.basename(os.path.dirname(ops_path))
                    potential_path = os.path.join(data_path, suite2p_folder, "suite2p", plane_name, reg_chan2_filename)
                    if os.path.isfile(potential_path):
                        print(f"Found binary chan2 file at: {potential_path}")
                        reg_file_chan2 = potential_path
                    else:
                        print(f"WARNING: Could not find binary chan2 file {reg_file_chan2}")
                        print(f"Searched locations: {potential_reg_chan2_file}, {potential_path}")
                        raise FileNotFoundError(f"Could not find binary chan2 file for plane {ipl}")
            
            raw_file_chan2 = ops.get("raw_file_chan2", 0) if raw else reg_file_chan2
        else:
            reg_file_chan2 = reg_file
            raw_file_chan2 = reg_file
        
        # Shape of binary files
        n_frames, Ly, Lx = ops["nframes"], ops["Ly"], ops["Lx"]
        
        # Perform registration
        twoc = ops["nchannels"] > 1
        null = contextlib.nullcontext()
        
        with io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file, n_frames=n_frames) \
            if raw else null as f_raw, \
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file, n_frames=n_frames) as f_reg, \
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file_chan2, n_frames=n_frames) \
            if raw and twoc else null as f_raw_chan2, \
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file_chan2, n_frames=n_frames) \
            if twoc else null as f_reg_chan2:
                print(f"Registration files: {ops['ops_path']}, {f_reg.filename}")
                registration_outputs = register.registration_wrapper(
                    f_reg, f_raw=f_raw, f_reg_chan2=f_reg_chan2, f_raw_chan2=f_raw_chan2,
                    refImg=refImgs[ipl], align_by_chan2=align_by_chan2, ops=ops)
                
                ops = register.save_registration_outputs_to_ops(registration_outputs, ops)
                
                meanImgE = register.compute_enhanced_mean_image(
                    ops["meanImg"].astype(np.float32), ops)
                ops["meanImgE"] = meanImgE
        
        # Save updated ops file
        np.save(ops["ops_path"], ops)
        updated_ops_list.append(ops)
    
    return updated_ops_list

def detect_red_cells(data_path, ref_path, nplanes=1, suite2p_folder=None):
    """
    Detect red cells based on channel 2 intensity.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the directory containing the registered files
    ref_path : str or Path
        Path to the directory containing the reference files with cell stats
    nplanes : int, optional
        Number of planes to process
    suite2p_folder : str, optional
        Name of the suite2p output folder. If None, will try "suite2p_done" then "suite2p"
        
    Returns:
    --------
    list
        List of redcell arrays for each plane
    """
    data_path = Path(data_path)
    ref_path = Path(ref_path)
    
    print(f"Detecting red cells using data path: {data_path}")
    print(f"Using reference path: {ref_path}")
    
    # Try to find the suite2p folder in data path
    if suite2p_folder is None:
        # Try suite2p_done first, then suite2p
        potential_folders = ["suite2p_done", "suite2p"]
        found_folder = None
        
        for folder in potential_folders:
            if (data_path / folder).exists():
                found_folder = folder
                break
                
        if found_folder is None:
            raise FileNotFoundError(f"Could not find suite2p or suite2p_done folder in {data_path}")
            
        suite2p_folder = found_folder
    else:
        # Verify the specified folder exists
        if not (data_path / suite2p_folder).exists():
            raise FileNotFoundError(f"Specified suite2p folder '{suite2p_folder}' not found in {data_path}")
    
    print(f"Using suite2p folder: {suite2p_folder}")
    
    # Get plane folders and paths
    save_folder = suite2p_folder
    save_path = data_path / save_folder
    plane_folders = natsorted([
        f.path for f in os.scandir(save_path) 
        if f.is_dir() and f.name[:5] == "plane"
    ])
    ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
    
    # Find the suite2p folder in the reference path
    potential_ref_folders = ["suite2p", "suite2p_done"]
    ref_suite2p_folder = None
    
    for folder in potential_ref_folders:
        if (ref_path / folder).exists():
            ref_suite2p_folder = folder
            break
            
    if ref_suite2p_folder is None:
        raise FileNotFoundError(f"Could not find suite2p or suite2p_done folder in reference path {ref_path}")
        
    print(f"Using reference suite2p folder: {ref_suite2p_folder}")
    
    # Get reference stat paths
    stat_paths = [ref_path / ref_suite2p_folder / f"plane{ipl}" / "stat.npy" for ipl in range(nplanes)]
    redcell_paths = [ref_path / ref_suite2p_folder / f"plane{ipl}" / "redcell.npy" for ipl in range(nplanes)]
    
    # Process each plane
    redcell_list = []
    for ipl, ops_path in enumerate(ops_paths[:nplanes]):
        if ipl >= len(stat_paths) or not os.path.exists(stat_paths[ipl]):
            print(f"Skipping plane {ipl}: stat file not found")
            continue
            
        print(f"\n>>>> Processing red cells for PLANE {ipl}")
        ops = np.load(ops_path, allow_pickle=True).item()
        stat = np.load(stat_paths[ipl], allow_pickle=True)
        
        # Detect red cells
        ops, redstats = chan2detect.detect(ops, stat)
        
        # Save updated ops file and redcell stats
        np.save(ops_path, ops)
        
        # Update reference ops file with channel 2 information
        ref_ops_path = ref_path / f"plane{ipl}" / "ops.npy"
        if os.path.exists(ref_ops_path):
            ref_ops = np.load(ref_ops_path, allow_pickle=True).item()
            ref_ops["meanImg_chan2"] = ops["meanImg_chan2"]
            ref_ops["meanImg_chan2_corrected"] = ops["meanImg_chan2_corrected"]
            ref_ops["nchannels"] = 2
            np.save(ref_ops_path, ref_ops)
        
        # Save redcell stats
        np.save(redcell_paths[ipl], redstats)
        redcell_list.append(redstats)
    
    return redcell_list

def visualize_alignment(ops_paths, ops_paths_ref, output_dir=None):
    """
    Visualize the alignment between reference and registered images.
    
    Parameters:
    -----------
    ops_paths : list
        List of paths to the registered ops.npy files
    ops_paths_ref : list
        List of paths to the reference ops.npy files
    output_dir : str or Path, optional
        Directory to save visualization figures
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    nplanes = min(len(ops_paths), len(ops_paths_ref))
    plt.figure(figsize=(12, 4 * nplanes))
    
    for ipl in range(nplanes):
        # Load ops files
        ops = np.load(ops_paths[ipl], allow_pickle=True).item()
        ops_ref = np.load(ops_paths_ref[ipl], allow_pickle=True).item()
        
        # Create subplots for this plane
        plt.subplot(nplanes, 4, 1 + ipl*4)
        plt.imshow(normalize99(ops_ref["meanImg"]), vmin=0, vmax=1, cmap="gray")
        plt.title(f"Plane {ipl}, Reference")
        plt.axis("off")
        
        plt.subplot(nplanes, 4, 2 + ipl*4)
        plt.imshow(normalize99(ops["meanImg"]), vmin=0, vmax=1, cmap="gray")
        plt.title(f"Plane {ipl}, Registered")
        plt.axis("off")
        
        # Create RGB overlay
        rgb = np.zeros((*ops["meanImg"].shape, 3))
        rgb[:,:,1] = np.clip(normalize99(ops["meanImg"]), 0, 1)  # Green
        rgb[:,:,2] = np.clip(normalize99(ops_ref["meanImg"]), 0, 1)  # Blue
        plt.subplot(nplanes, 4, 3 + ipl*4)
        plt.imshow(rgb)
        plt.title(f"Plane {ipl}, Overlay")
        plt.axis("off")
        
        # Display red channel mean image
        plt.subplot(nplanes, 4, 4 + ipl*4)
        if "meanImg_chan2" in ops:
            plt.imshow(normalize99(ops["meanImg_chan2"]), vmin=0, vmax=1, cmap="gray")
            plt.title(f"Plane {ipl}, Red Channel")
        else:
            plt.title(f"Plane {ipl}, Red Channel (Not Available)")
        plt.axis("off")
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / "channel_alignment.png", dpi=150)
    
    plt.show()

def combine_planes(ref_path):
    """
    Combine data from all planes.
    
    Parameters:
    -----------
    ref_path : str or Path
        Path to the directory containing the plane folders
    """
    from suite2p.io.save import combined
    ref_path = Path(ref_path)
    
    print(f"Combining data from all planes in: {ref_path}")
    combined(str(ref_path))
    print("Planes combined successfully")

def main():
    parser = argparse.ArgumentParser(description="Two-channel registration for Suite2p data")
    parser.add_argument("--green_path", required=True, help="Path to the green channel suite2p directory")
    parser.add_argument("--data_path", required=True, help="Path to the two-channel data directory")
    parser.add_argument("--output_dir", help="Directory to save visualizations")
    parser.add_argument("--num_planes", type=int, default=1, help="Number of planes to process")
    parser.add_argument("--align_by_chan2", action="store_true", help="Align by channel 2 (red) instead of channel 1 (green)")
    parser.add_argument("--mesoscan", action="store_true", help="Data is from a mesoscope")
    parser.add_argument("--skip_binary", action="store_true", help="Skip binary conversion step")
    parser.add_argument("--skip_registration", action="store_true", help="Skip registration step")
    parser.add_argument("--skip_red_detection", action="store_true", help="Skip red cell detection step")
    parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization step")
    parser.add_argument("--skip_combine", action="store_true", help="Skip combining planes step")
    
    args = parser.parse_args()
    
    green_path = Path(args.green_path)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Create todo list
    todos = []
    if not args.skip_binary:
        todos.append("Convert to binary")
    if not args.skip_registration:
        todos.append("Register channels")
    if not args.skip_red_detection:
        todos.append("Detect red cells")
    if not args.skip_visualization:
        todos.append("Visualize alignment")
    if not args.skip_combine:
        todos.append("Combine planes")
    
    print(f"Processing {args.num_planes} planes")
    print(f"Green channel path: {green_path}")
    print(f"Data path: {data_path}")
    print(f"Tasks to perform: {', '.join(todos)}")
    
    # Step 1: Convert to binary if needed
    if not args.skip_binary:
        print("\n===== Converting to Binary =====")
        ops = convert_to_binary(data_path, mesoscan=args.mesoscan)
    
    # Step 2: Register channels
    if not args.skip_registration:
        print("\n===== Registering Channels =====")
        register_channels(
            data_path, 
            green_path, 
            nplanes=args.num_planes,
            align_by_chan2=args.align_by_chan2
        )
    
    # Step 3: Detect red cells
    if not args.skip_red_detection:
        print("\n===== Detecting Red Cells =====")
        detect_red_cells(
            data_path,
            green_path,
            nplanes=args.num_planes
        )
    
    # Step 4: Visualize alignment
    if not args.skip_visualization:
        print("\n===== Visualizing Alignment =====")
        
        # Try to find the suite2p folder
        potential_folders = ["suite2p_done", "suite2p"]
        found_folder = None
        
        for folder in potential_folders:
            if (data_path / folder).exists():
                found_folder = folder
                break
                
        if found_folder is None:
            raise FileNotFoundError(f"Could not find suite2p or suite2p_done folder in {data_path}")
            
        save_folder = found_folder
        print(f"Using suite2p folder for visualization: {save_folder}")
        
        # Get ops paths
        save_path = data_path / save_folder
        plane_folders = natsorted([
            f.path for f in os.scandir(save_path) 
            if f.is_dir() and f.name[:5] == "plane"
        ])
        ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
        
        # Find the suite2p folder in the reference path
        potential_ref_folders = ["suite2p", "suite2p_done"]
        ref_suite2p_folder = None
        
        for folder in potential_ref_folders:
            if (green_path / folder).exists():
                ref_suite2p_folder = folder
                break
                
        if ref_suite2p_folder is None:
            raise FileNotFoundError(f"Could not find suite2p or suite2p_done folder in reference path {green_path}")
            
        print(f"Using reference suite2p folder for visualization: {ref_suite2p_folder}")
        
        # Get reference ops paths
        ops_paths_ref = [green_path / ref_suite2p_folder / f"plane{ipl}" / "ops.npy" for ipl in range(args.num_planes)]
        
        visualize_alignment(
            ops_paths[:args.num_planes],
            ops_paths_ref,
            output_dir
        )
    
    # Step 5: Combine planes
    if not args.skip_combine:
        print("\n===== Combining Planes =====")
        combine_planes(green_path)
    
    print("\n===== Processing Complete =====")

if __name__ == "__main__":
    main()