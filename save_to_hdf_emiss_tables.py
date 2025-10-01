#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:44:34 2025

@author: edne8319
"""
"""
IDL Save File to HDF5 Converter for IPT Emission Tables
========================================================

This script converts IDL save files containing CHIANTI emission tables
to HDF5 format for use with the Python IPT UV emission model.

Requirements:
- scipy (for reading IDL save files)
- h5py (for writing HDF5 files)
- numpy
"""

import numpy as np
from scipy.io import readsav
import h5py
import argparse
from typing import Dict, Any


def read_idl_save_file(filename: str) -> Dict[str, Any]:
    """
    Read an IDL save file and return its contents.
    
    Parameters
    ----------
    filename : str
        Path to IDL save file (.sav)
        
    Returns
    -------
    dict
        Dictionary containing all variables from the save file
    """
    print(f"Reading IDL save file: {filename}")
    data = readsav(filename, verbose=True)
    
    print("\nContents of IDL save file:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    return data


def convert_single_maxwellian_to_hdf5(idl_filename: str, hdf5_filename: str):
    """
    Convert single Maxwellian IDL save file to HDF5 format.
    
    Expected IDL file structure:
    - TEMP_ARR: array of temperatures [eV]
    - DENS_ARR: array of densities [cm^-3]
    - XWAVI: structure with wavelength arrays for each species
    - YPTSI: structure with emissivity arrays [n_temp, n_dens, n_lines]
    """
    # Read IDL file
    data = read_idl_save_file(idl_filename)
    
    print(f"\nConverting to HDF5: {hdf5_filename}")
    
    with h5py.File(hdf5_filename, 'w') as f:
        # Create groups
        param_group = f.create_group('parameters')
        wav_group = f.create_group('wavelengths')
        emiss_group = f.create_group('emissivities')
        
        # Store parameters
        param_group.create_dataset('temperature', data=data['TEMP_ARR'])
        param_group.create_dataset('density', data=data['DENS_ARR'])
        
        # Add metadata
        param_group.attrs['temperature_units'] = 'eV'
        param_group.attrs['density_units'] = 'cm^-3'
        
        # Process wavelengths and emissivities
        xwavi = data['XWAVI']
        yptsi = data['YPTSI']
        
        # Map species names (IDL structure fields to standardized names)
        species_map = {
            'S': 'S',      # S I
            'SP': 'SP',    # S II
            'S2P': 'S2P',  # S III
            'S3P': 'S3P',  # S IV
            'S4P': 'S4P',  # S V
            'O': 'O',      # O I
            'OP': 'OP',    # O II
            'O2P': 'O2P',  # O III
            'NAP': 'NAP'   # Na II
        }
        
        # Extract data from IDL structures
        for field_name in xwavi.dtype.names:
            if field_name in species_map:
                hdf_key = species_map[field_name]
                
                # Get wavelengths
                wavelengths = xwavi[field_name][0]
                wav_group.create_dataset(hdf_key, data=wavelengths)
                wav_group[hdf_key].attrs['units'] = 'Angstroms'
                
                # Get emissivities
                emissivities = yptsi[field_name][0]
                emiss_group.create_dataset(hdf_key, data=emissivities)
                emiss_group[hdf_key].attrs['units'] = 'photons/s/cm^3'
                
                print(f"  Converted {field_name} -> {hdf_key}: {len(wavelengths)} lines")
        
        # Add global metadata
        f.attrs['description'] = 'CHIANTI emission tables - Single Maxwellian'
        f.attrs['source'] = f'Converted from {idl_filename}'
        f.attrs['chianti_version'] = '11.0.2'
    
    print(f"Successfully created: {hdf5_filename}")


def convert_double_maxwellian_to_hdf5(idl_filename: str, hdf5_filename: str):
    """
    Convert double Maxwellian IDL save file to HDF5 format.
    
    Expected IDL file structure:
    - TEC_ARR: array of core temperatures [eV]
    - TEH_ARR: array of hot temperatures [eV]
    - NE_ARR: array of densities [cm^-3]
    - FEH_ARR: array of hot electron fractions
    - XWAVI: structure with wavelength arrays
    - YPTSI: structure with emissivity arrays [n_tec, n_teh, n_ne, n_feh, n_lines]
    """
    # Read IDL file
    data = read_idl_save_file(idl_filename)
    
    print(f"\nConverting to HDF5: {hdf5_filename}")
    
    with h5py.File(hdf5_filename, 'w') as f:
        # Create groups
        param_group = f.create_group('parameters')
        wav_group = f.create_group('wavelengths')
        emiss_group = f.create_group('emissivities')
        
        # Store parameters
        param_group.create_dataset('core_temperature', data=data['TEC_ARR'])
        param_group.create_dataset('hot_temperature', data=data['TEH_ARR'])
        param_group.create_dataset('density', data=data['NE_ARR'])
        param_group.create_dataset('hot_fraction', data=data['FEH_ARR'])
        
        # Add metadata
        param_group.attrs['core_temperature_units'] = 'eV'
        param_group.attrs['hot_temperature_units'] = 'eV'
        param_group.attrs['density_units'] = 'cm^-3'
        param_group.attrs['hot_fraction_units'] = 'dimensionless'
        
        # Process wavelengths and emissivities
        xwavi = data['XWAVI']
        yptsi = data['YPTSI']
        
        # Map species names
        species_map = {
            'S': 'S',
            'SP': 'SP',
            'S2P': 'S2P',
            'S3P': 'S3P',
            'S4P': 'S4P',
            'O': 'O',
            'OP': 'OP',
            'O2P': 'O2P',
            'NAP': 'NAP'
        }
        
        # Extract data from IDL structures
        for field_name in xwavi.dtype.names:
            if field_name in species_map:
                hdf_key = species_map[field_name]
                
                # Get wavelengths
                wavelengths = xwavi[field_name][0]
                wav_group.create_dataset(hdf_key, data=wavelengths)
                wav_group[hdf_key].attrs['units'] = 'Angstroms'
                
                # Get emissivities
                emissivities = yptsi[field_name][0]
                emiss_group.create_dataset(hdf_key, data=emissivities)
                emiss_group[hdf_key].attrs['units'] = 'photons/s/cm^3'
                
                print(f"  Converted {field_name} -> {hdf_key}: {len(wavelengths)} lines")
        
        # Add global metadata
        f.attrs['description'] = 'CHIANTI emission tables - Double Maxwellian'
        f.attrs['source'] = f'Converted from {idl_filename}'
        f.attrs['chianti_version'] = '11.0.2'
    
    print(f"Successfully created: {hdf5_filename}")


def verify_hdf5_file(filename: str):
    """
    Verify and display contents of created HDF5 file.
    """
    print(f"\nVerifying HDF5 file: {filename}")
    
    with h5py.File(filename, 'r') as f:
        print("\nFile attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        print("\nParameter arrays:")
        for key in f['parameters'].keys():
            dataset = f['parameters'][key]
            print(f"  {key}: shape {dataset.shape}, range [{dataset[()].min():.2f}, {dataset[()].max():.2f}]")
        
        print("\nSpecies data:")
        species_list = list(f['wavelengths'].keys())
        for species in species_list:
            wav = f['wavelengths'][species]
            emiss = f['emissivities'][species]
            print(f"  {species}: {len(wav)} lines, emissivity shape {emiss.shape}")


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Convert IDL save files to HDF5 format for IPT emission model'
    )
    parser.add_argument('input_file', help='Input IDL save file (.sav)')
    parser.add_argument('output_file', help='Output HDF5 file (.h5)')
    parser.add_argument('--type', choices=['single', 'double'], required=True,
                       help='Type of Maxwellian table')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the created HDF5 file')
    
    args = parser.parse_args()
    
    # Convert based on type
    if args.type == 'single':
        convert_single_maxwellian_to_hdf5(args.input_file, args.output_file)
    else:
        convert_double_maxwellian_to_hdf5(args.input_file, args.output_file)
    
    # Verify if requested
    if args.verify:
        verify_hdf5_file(args.output_file)


if __name__ == "__main__":
    # Example usage (can be run directly or from command line)
    
    # For command-line usage:
    # python convert_idl_to_hdf5.py input.sav output.h5 --type single --verify
    
    # For direct Python usage:
    example_mode = False  # Set to True to run examples
    
    if example_mode:
        # Example 1: Convert single Maxwellian table
        print("Example 1: Converting single Maxwellian table")
        convert_single_maxwellian_to_hdf5(
            'CHIANTI_11.0.2_emiss_arrays_all_species_all_wavelengths_50x50_logspaced.sav',
            'CHIANTI_11.0.2_emiss_single_maxwellian.h5'
        )
        verify_hdf5_file('CHIANTI_11.0.2_emiss_single_maxwellian.h5')
        
        print("\n" + "="*70 + "\n")
        
        # Example 2: Convert double Maxwellian table
        print("Example 2: Converting double Maxwellian table")
        convert_double_maxwellian_to_hdf5(
            'CHIANTI_11.0.2_emiss_arrays_all_species_all_wavelengths_15x10x20x10_hote_logspaced.sav',
            'CHIANTI_11.0.2_emiss_double_maxwellian.h5'
        )
        verify_hdf5_file('CHIANTI_11.0.2_emiss_double_maxwellian.h5')
    else:
        # Run command-line interface
        main()