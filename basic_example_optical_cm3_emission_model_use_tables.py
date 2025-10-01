#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 18:58:46 2025

@author: Edward G Nerney

OPTICAL EMISSION MODEL FOR IO PLASMA TORUS - TABLE-BASED VERSION
=================================================================

This code simulates optical emission spectra from the Io Plasma Torus (IPT) in
Jupiter's magnetosphere using pre-calculated CHIANTI emission tables.

TARGET INSTRUMENT:
- DIS (Dual Imaging Spectrograph) at Apache Point Observatory (APO)
- High resolution mode: 0.62 Å/pixel
- Spectral resolution (FWHM): 1.86 Å (3x bin width)
- Wavelength coverage: 3200-7500 Å (optical range)

PHYSICAL MODEL:
- Uses pre-calculated emissivities from CHIANTI 11.0.2
- Supports both single and double Maxwellian electron distributions
- Double Maxwellian uses proper 4D interpolation (not linear superposition)
- Converts atomic emissivities to observable Rayleigh intensities
- Applies realistic instrument response function for DIS/APO

KEY ASSUMPTIONS:
- Optically thin plasma (no absorption/scattering)
- Electron impact excitation dominates (photoexcitation negligible)
- Local thermodynamic equilibrium NOT assumed (uses CHIANTI non-LTE calculations)
- Line-of-sight integration approximated by column densities

OPTICAL EMISSION LINES:
In the optical range (3200-7500 Å), the IPT shows:
- Forbidden lines: [S II] 6716,6731 Å, [S III] 6312,9069,9531 Å, [O II] 3726,3729 Å
- Permitted lines: Various S and O multiplets
- Auroral lines: [O I] 5577,6300,6364 Å (if neutral O present)
These lines are diagnostic of plasma conditions and composition.

REQUIRED FILES:
- Single Maxwellian: CHIANTI_11.0.2_emiss_single_maxwellian.h5
- Double Maxwellian: CHIANTI_11.0.2_emiss_double_maxwellian.h5
  (Generated from CHIANTI using make_emission_tables_chianti11.pro)

REFERENCES:
- Nerney et al. 2017, 2020, 2022, & 2025
- Steffl et al. 2004b, 2006 (optical observations)
- Brown et al. 1983 (S II and S III optical diagnostics)
- CHIANTI atomic database (Dere et al. 1997, Del Zanna et al. 2020, & Dufresne et al. 2024)
- CHIANTI is a collaborative project involving George Mason University,
  the University of Michigan (USA), University of Cambridge (UK) 
  and NASA Goddard Space Flight Center (USA).

Author: Python implementation for open-access community use
License: Open source for academic and research use
"""

import numpy as np
import h5py
from scipy import interpolate
from scipy.special import erf
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings


class EmissionTables:
    """
    Class to handle pre-calculated emission tables from CHIANTI.
    
    These tables contain emissivities (photons/s/cm^3) calculated over
    grids of plasma parameters using CHIANTI 11.0.2 atomic data.
    Includes both UV and optical transitions.
    """
    
    def __init__(self):
        self.single_maxwellian_loaded = False
        self.double_maxwellian_loaded = False
        
        # Ion species names mapping following spectroscopic notation
        self.ion_names = {
            'S': 'S I',      # Neutral sulfur (weak in IPT)
            'SP': 'S II',    # S+ (strong forbidden lines at 6716,6731 Å)
            'S2P': 'S III',  # S++ (forbidden lines at 6312,9069,9531 Å)
            'S3P': 'S IV',   # S+++ (weaker optical lines)
            'S4P': 'S V',    # S++++ (very weak in optical)
            'O': 'O I',      # Neutral oxygen (auroral lines if present)
            'OP': 'O II',    # O+ (forbidden doublet at 3726,3729 Å)
            'O2P': 'O III',  # O++ (forbidden lines at 4959,5007 Å)
            'NAP': 'Na II'   # Na+ (D lines if present)
        }
    
    def load_single_maxwellian_tables(self, filename: str):
        """
        Load single Maxwellian emission tables from HDF5 file.
        
        Parameters
        ----------
        filename : str
            Path to HDF5 file containing emission tables
            
        Table Structure
        ---------------
        The HDF5 file contains:
        - /parameters/temperature : array of temperatures [eV]
        - /parameters/density : array of densities [cm^-3]
        - /wavelengths/[species] : wavelengths for each species [Angstroms]
        - /emissivities/[species] : emissivity grid [n_lines, n_temp, n_dens]
        
        Note: Tables include both UV and optical wavelengths
        """
        print("Loading single Maxwellian emission tables...")
        
        with h5py.File(filename, 'r') as f:
            # Load parameter grids
            self.temp_arr = f['parameters/temperature'][:]  # [eV]
            self.dens_arr = f['parameters/density'][:]      # [cm^-3]
            
            # Load wavelengths and emissivities for each species
            self.wavelengths_single = {}
            self.emissivities_single = {}
            
            for species_key in f['wavelengths'].keys():
                self.wavelengths_single[species_key] = f[f'wavelengths/{species_key}'][:]
                # Arrays from IDL are transposed due to column-major vs row-major ordering
                # Original IDL: [n_temp, n_dens, n_lines] -> Python: [n_lines, n_dens, n_temp]
                # We transpose to [n_lines, n_temp, n_dens] for consistent indexing
                emiss_raw = f[f'emissivities/{species_key}'][:]
                self.emissivities_single[species_key] = np.transpose(emiss_raw, (0, 2, 1))
        
        # Create log-space grids for interpolation
        # Emissivities vary as power laws, so log interpolation is more accurate
        self.log_temp = np.log10(self.temp_arr)
        self.log_dens = np.log10(self.dens_arr)
        
        self.single_maxwellian_loaded = True
        
        print("Tables loaded successfully:")
        print(f"  Temperature range: {self.temp_arr.min():.8f} - {self.temp_arr.max():.5f} eV")
        print(f"  Density range: {self.dens_arr.min():.8f} - {self.dens_arr.max():.3f} cm^-3")
        print(f"  Grid size: {len(self.temp_arr)} x {len(self.dens_arr)}")
    
    def load_double_maxwellian_tables(self, filename: str):
        """
        Load double Maxwellian emission tables from HDF5 file.
        
        Parameters
        ----------
        filename : str
            Path to HDF5 file containing emission tables
            
        Table Structure
        ---------------
        The HDF5 file contains:
        - /parameters/core_temperature : array of core temperatures [eV]
        - /parameters/hot_temperature : array of hot temperatures [eV]
        - /parameters/density : array of densities [cm^-3]
        - /parameters/hot_fraction : array of hot electron fractions
        - /wavelengths/[species] : wavelengths for each species [Angstroms]
        - /emissivities/[species] : emissivity grid [n_lines, n_feh, n_dens, n_teh, n_tec]
        """
        print("Loading double Maxwellian emission tables...")
        
        with h5py.File(filename, 'r') as f:
            # Load parameter grids
            self.tec_arr = f['parameters/core_temperature'][:]  # [eV]
            self.teh_arr = f['parameters/hot_temperature'][:]   # [eV]
            self.ne_arr = f['parameters/density'][:]            # [cm^-3]
            self.feh_arr = f['parameters/hot_fraction'][:]      # [fraction]
            
            # Load wavelengths and emissivities for each species
            self.wavelengths_double = {}
            self.emissivities_double = {}
            
            for species_key in f['wavelengths'].keys():
                self.wavelengths_double[species_key] = f[f'wavelengths/{species_key}'][:]
                # Store raw arrays - will handle transposition during interpolation
                self.emissivities_double[species_key] = f[f'emissivities/{species_key}'][:]
        
        # Create log-space grids for interpolation
        self.log_tec = np.log10(self.tec_arr)
        self.log_teh = np.log10(self.teh_arr)
        self.log_ne = np.log10(self.ne_arr)
        
        self.double_maxwellian_loaded = True
        
        print("Tables loaded successfully:")
        print(f"  Core temperature range: {self.tec_arr.min():.8f} - {self.tec_arr.max():.5f} eV")
        print(f"  Hot temperature range: {self.teh_arr.min():.5f} - {self.teh_arr.max():.5f} eV")
        print(f"  Density range: {self.ne_arr.min():.7f} - {self.ne_arr.max():.4f} cm^-3")
        print(f"  Hot fraction range: {self.feh_arr.min():.10f} - {self.feh_arr.max():.8f}")
        print(f"  Grid size: {len(self.tec_arr)} x {len(self.teh_arr)} x {len(self.ne_arr)} x {len(self.feh_arr)}")


def interpolate_emissivity_2d(tables: EmissionTables, 
                              temperature: float, 
                              density: float,
                              species_list: list = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate emissivities from 2D single Maxwellian tables.
    
    Uses bilinear interpolation in log10(T) - log10(n) space for smooth
    interpolation of emissivities which vary as power laws.
    
    Parameters
    ----------
    tables : EmissionTables
        Object containing loaded emission tables
    temperature : float
        Electron temperature [eV]
    density : float
        Electron density [cm^-3]
    species_list : list, optional
        List of species to include (default: ionized S and O species)
        
    Returns
    -------
    wavelengths : np.ndarray
        Combined wavelengths from all species [Angstroms]
    emissivities : np.ndarray
        Interpolated emissivities [photons/s/cm^3]
        
    Physics Notes
    -------------
    For forbidden optical transitions:
    - Critical density effects become important at higher densities
    - Collisional de-excitation competes with radiative decay
    - Tables include these density-dependent effects
    """
    if not tables.single_maxwellian_loaded:
        raise ValueError("Single Maxwellian tables not loaded")
    
    # Default species list for optical emission
    # Include all ionized species that have optical lines
    if species_list is None:
        species_list = ['SP', 'S2P', 'S3P', 'S4P', 'OP', 'O2P']
    
    # Convert to log space for interpolation
    log_t = np.log10(temperature)
    log_n = np.log10(density)
    
    # Check bounds and warn if extrapolating
    if log_t < tables.log_temp.min() or log_t > tables.log_temp.max():
        warnings.warn(f"Temperature {temperature:.2f} eV outside table range")
    if log_n < tables.log_dens.min() or log_n > tables.log_dens.max():
        warnings.warn(f"Density {density:.2e} cm^-3 outside table range")
    
    # Find nearest indices for interpolation
    temp_idx = np.searchsorted(tables.log_temp, log_t)
    dens_idx = np.searchsorted(tables.log_dens, log_n)
    
    # Ensure indices are within bounds for interpolation
    temp_idx = np.clip(temp_idx, 1, len(tables.log_temp) - 1)
    dens_idx = np.clip(dens_idx, 1, len(tables.log_dens) - 1)
    
    # Collect all wavelengths and emissivities
    all_wavelengths = []
    all_emissivities = []
    
    for species_key in species_list:
        if species_key not in tables.wavelengths_single:
            continue
            
        wavelengths = tables.wavelengths_single[species_key]
        emiss_table = tables.emissivities_single[species_key]
        
        # Bilinear interpolation
        # Get surrounding points
        t0, t1 = temp_idx - 1, temp_idx
        d0, d1 = dens_idx - 1, dens_idx
        
        # Calculate interpolation weights
        wt = (log_t - tables.log_temp[t0]) / (tables.log_temp[t1] - tables.log_temp[t0])
        wd = (log_n - tables.log_dens[d0]) / (tables.log_dens[d1] - tables.log_dens[d0])
        
        # Bilinear interpolation formula
        # emissivities array is [n_lines, n_temp, n_dens]
        interp_emiss = (
            (1 - wt) * (1 - wd) * emiss_table[:, t0, d0] +
            wt * (1 - wd) * emiss_table[:, t1, d0] +
            (1 - wt) * wd * emiss_table[:, t0, d1] +
            wt * wd * emiss_table[:, t1, d1]
        )
        
        all_wavelengths.extend(wavelengths)
        all_emissivities.extend(interp_emiss)
    
    # Convert to arrays and sort by wavelength
    wavelengths = np.array(all_wavelengths)
    emissivities = np.array(all_emissivities)
    
    sort_idx = np.argsort(wavelengths)
    return wavelengths[sort_idx], emissivities[sort_idx]


def interpolate_emissivity_4d(tables: EmissionTables,
                              core_temp: float,
                              hot_temp: float,
                              density: float,
                              hot_fraction: float,
                              species_list: list = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate emissivities from 4D double Maxwellian tables.
    
    Uses proper 4D interpolation since emission is nonlinear in the parameters.
    The double Maxwellian represents:
    f(v) = (1-feh) * f_Maxwell(v,Tec) + feh * f_Maxwell(v,Teh)
    
    Parameters
    ----------
    tables : EmissionTables
        Object containing loaded emission tables
    core_temp : float
        Core electron temperature [eV]
    hot_temp : float
        Hot electron temperature [eV]
    density : float
        Total electron density [cm^-3]
    hot_fraction : float
        Fraction of hot electrons (0 to 1)
    species_list : list, optional
        List of species to include
        
    Returns
    -------
    wavelengths : np.ndarray
        Combined wavelengths from all species [Angstroms]
    emissivities : np.ndarray
        Interpolated emissivities [photons/s/cm^3]
        
    Physics Notes
    -------------
    For optical forbidden lines:
    - Hot electrons have minimal effect on low-lying metastable levels
    - Main enhancement is for higher excitation transitions
    - Critical density effects are included in the tables
    """
    if not tables.double_maxwellian_loaded:
        raise ValueError("Double Maxwellian tables not loaded")
    
    # Default species list for optical
    if species_list is None:
        species_list = ['SP', 'S2P', 'S3P', 'S4P', 'OP', 'O2P']
    
    # Convert to log space for temperature and density
    log_tec = np.log10(core_temp)
    log_teh = np.log10(hot_temp)
    log_n = np.log10(density)
    
    # Check bounds
    if log_tec < tables.log_tec.min() or log_tec > tables.log_tec.max():
        warnings.warn(f"Core temperature {core_temp:.2f} eV outside table range")
    if log_teh < tables.log_teh.min() or log_teh > tables.log_teh.max():
        warnings.warn(f"Hot temperature {hot_temp:.2f} eV outside table range")
    if log_n < tables.log_ne.min() or log_n > tables.log_ne.max():
        warnings.warn(f"Density {density:.2e} cm^-3 outside table range")
    if hot_fraction < tables.feh_arr.min() or hot_fraction > tables.feh_arr.max():
        warnings.warn(f"Hot fraction {hot_fraction:.4f} outside table range")
    
    # Collect all wavelengths and emissivities
    all_wavelengths = []
    all_emissivities = []
    
    for species_key in species_list:
        if species_key not in tables.wavelengths_double:
            continue
            
        wavelengths = tables.wavelengths_double[species_key]
        emiss_table = tables.emissivities_double[species_key]
        
        # Transpose array from [n_lines, n_feh, n_dens, n_teh, n_tec] 
        # to [n_tec, n_teh, n_dens, n_feh, n_lines] for interpolation
        emiss_reordered = np.transpose(emiss_table, (4, 3, 2, 1, 0))
        
        n_lines = len(wavelengths)
        interp_emiss = np.zeros(n_lines)
        
        # Interpolate each emission line
        for i in range(n_lines):
            interp_func = interpolate.RegularGridInterpolator(
                (tables.log_tec, tables.log_teh, tables.log_ne, tables.feh_arr),
                emiss_reordered[:, :, :, :, i],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            interp_emiss[i] = interp_func((log_tec, log_teh, log_n, hot_fraction))
        
        all_wavelengths.extend(wavelengths)
        all_emissivities.extend(interp_emiss)
    
    # Convert to arrays and sort by wavelength
    wavelengths = np.array(all_wavelengths)
    emissivities = np.array(all_emissivities)
    
    sort_idx = np.argsort(wavelengths)
    return wavelengths[sort_idx], emissivities[sort_idx]


def calculate_ipt_emiss_tables_single(tables: EmissionTables,
                                     temperature: float,
                                     density: float,
                                     column_densities: Dict[str, float],
                                     min_wav: float = 3200.0,
                                     max_wav: float = 7500.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate IPT optical emission line intensities using single Maxwellian tables.
    
    Parameters
    ----------
    tables : EmissionTables
        Object containing loaded emission tables
    temperature : float
        Electron temperature [eV]
    density : float
        Electron density [cm^-3]
    column_densities : dict
        Ion column densities [cm^-2] with keys:
        'S+', 'S++', 'S+++', 'S++++', 'O+', 'O++'
    min_wav : float
        Minimum wavelength [Angstroms], default=3200
    max_wav : float
        Maximum wavelength [Angstroms], default=7500
        
    Returns
    -------
    wavelengths : np.ndarray
        Emission line wavelengths [Angstroms]
    intensities : np.ndarray
        Line intensities [Rayleighs]
        
    Optical Lines
    -------------
    Key diagnostic lines in this range:
    - [S II] 4068,4076 Å (auroral), 6716,6731 Å (nebular)
    - [S III] 6312 Å (auroral), 9069,9531 Å (nebular)
    - [O II] 3726,3729 Å (nebular doublet), 7319,7330 Å (auroral)
    - [O III] 4363 Å (auroral), 4959,5007 Å (nebular)
    """
    print("Calculating single Maxwellian optical emission using tables...")
    
    # Get interpolated emissivities for all species
    wavelengths, emissivities = interpolate_emissivity_2d(tables, temperature, density)
    
    # Map column densities to species notation
    species_columns = {
        'SP': column_densities.get('S+', 0.0),
        'S2P': column_densities.get('S++', 0.0),
        'S3P': column_densities.get('S+++', 0.0),
        'S4P': column_densities.get('S++++', 0.0),
        'OP': column_densities.get('O+', 0.0),
        'O2P': column_densities.get('O++', 0.0)
    }
    
    # Calculate intensities by multiplying emissivities by column densities
    intensities = np.zeros_like(emissivities)
    
    # Build wavelength-to-species mapping
    wav_species_map = {}
    for species_key in species_columns.keys():
        if species_key in tables.wavelengths_single:
            for wav in tables.wavelengths_single[species_key]:
                wav_species_map[wav] = species_key
    
    # Apply column densities to corresponding lines
    for i, wav in enumerate(wavelengths):
        # Find which species this wavelength belongs to
        for wav_key, species in wav_species_map.items():
            if abs(wav - wav_key) < 0.01:  # Within 0.01 Angstrom tolerance
                # Convert to Rayleighs: 1 R = 10^6 photons/s/cm^2/4π sr
                intensities[i] = emissivities[i] * species_columns[species] * 1e-6
                break
    
    # Filter to optical wavelength range
    mask = (wavelengths >= min_wav) & (wavelengths <= max_wav)
    
    return wavelengths[mask], intensities[mask]


def calculate_ipt_emiss_tables_double(tables: EmissionTables,
                                     core_temp: float,
                                     hot_temp: float,
                                     density: float,
                                     hot_fraction: float,
                                     column_densities: Dict[str, float],
                                     min_wav: float = 3200.0,
                                     max_wav: float = 7500.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate IPT optical emission line intensities using double Maxwellian tables.
    
    Parameters
    ----------
    tables : EmissionTables
        Object containing loaded emission tables
    core_temp : float
        Core electron temperature [eV]
    hot_temp : float
        Hot electron temperature [eV]
    density : float
        Total electron density [cm^-3]
    hot_fraction : float
        Fraction of hot electrons (0 to 1)
    column_densities : dict
        Ion column densities [cm^-2]
    min_wav : float
        Minimum wavelength [Angstroms]
    max_wav : float
        Maximum wavelength [Angstroms]
        
    Returns
    -------
    wavelengths : np.ndarray
        Emission line wavelengths [Angstroms]
    intensities : np.ndarray
        Line intensities [Rayleighs]
        
    Physics
    -------
    Hot electrons have limited impact on optical forbidden lines
    from low-lying metastable levels, but can enhance auroral lines
    from higher energy levels.
    """
    print("Calculating double Maxwellian optical emission using 4D tables...")
    
    # Get interpolated emissivities
    wavelengths, emissivities = interpolate_emissivity_4d(
        tables, core_temp, hot_temp, density, hot_fraction
    )
    
    # Map column densities to species notation
    species_columns = {
        'SP': column_densities.get('S+', 0.0),
        'S2P': column_densities.get('S++', 0.0),
        'S3P': column_densities.get('S+++', 0.0),
        'S4P': column_densities.get('S++++', 0.0),
        'OP': column_densities.get('O+', 0.0),
        'O2P': column_densities.get('O++', 0.0)
    }
    
    # Calculate intensities
    intensities = np.zeros_like(emissivities)
    
    # Build wavelength-to-species mapping
    wav_species_map = {}
    for species_key in species_columns.keys():
        if species_key in tables.wavelengths_double:
            for wav in tables.wavelengths_double[species_key]:
                wav_species_map[wav] = species_key
    
    # Apply column densities
    for i, wav in enumerate(wavelengths):
        for wav_key, species in wav_species_map.items():
            if abs(wav - wav_key) < 0.01:
                intensities[i] = emissivities[i] * species_columns[species] * 1e-6
                break
    
    # Filter to optical wavelength range
    mask = (wavelengths >= min_wav) & (wavelengths <= max_wav)
    
    return wavelengths[mask], intensities[mask]


def simulate_ipt_spectrum_rayleighs_erf_form(wavelength_grid: np.ndarray,
                                            bin_width: float,
                                            line_wavelengths: np.ndarray,
                                            line_intensities: np.ndarray,
                                            fwhm: float = 1.86) -> np.ndarray:
    """
    Convolve discrete emission lines with DIS/APO instrument response function.
    
    Uses Error Function (ERF) formulation for exact integration of Gaussian
    line profile over finite wavelength bins. Optimized for DIS high-resolution mode.
    
    Parameters
    ----------
    wavelength_grid : np.ndarray
        Output wavelength grid [Angstroms]
    bin_width : float
        Width of wavelength bins [Angstroms] - 0.62 Å for DIS high-res
    line_wavelengths : np.ndarray
        Wavelengths of emission lines [Angstroms]
    line_intensities : np.ndarray
        Intensities of emission lines [Rayleighs]
    fwhm : float
        Full Width Half Maximum of Gaussian response [Angstroms]
        1.86 Å for DIS high-resolution mode (3x bin width)
        
    Returns
    -------
    spectrum : np.ndarray
        Convolved spectrum [Rayleighs/Angstrom]
        
    DIS Characteristics
    -------------------
    The Dual Imaging Spectrograph at Apache Point Observatory:
    - Simultaneous red and blue channel coverage
    - High resolution mode: ~0.62 Å/pixel
    - Spectral resolution: R ~ 2000-4000 depending on wavelength
    - Excellent for resolving IPT forbidden line doublets
    """
    # Gaussian profile parameters
    # FWHM = 2σ√(2ln2), so σ = FWHM/(2√(2ln2))
    # For ERF formulation, we need 1/(σ√2)
    rootc = 2.0 * np.sqrt(np.log(2.0)) / fwhm
    
    # Initialize output spectrum
    spectrum = np.zeros_like(wavelength_grid)
    
    # Convolve each line with instrument response
    for wav_line, intensity in zip(line_wavelengths, line_intensities):
        # Calculate contribution using exact ERF integral
        spectrum += intensity * 0.5 * (
            erf((wavelength_grid - wav_line + bin_width/2.0) * rootc) -
            erf((wavelength_grid - wav_line - bin_width/2.0) * rootc)
        )
    
    # Convert from integrated intensity to intensity per unit wavelength
    spectrum /= bin_width
    
    return spectrum


def basic_example_optical_cm3_emission_model_use_tables():
    """
    Main example demonstrating optical emission calculations for the Io Plasma
    Torus using pre-calculated CHIANTI emission tables, optimized for DIS/APO
    observations.
    
    This procedure:
    1. Loads pre-calculated emission tables from HDF5 files
    2. Sets up typical IPT plasma parameters at ~6 Jupiter radii
    3. Calculates optical emission spectra for single Maxwellian case
    4. Calculates optical emission spectra for double Maxwellian case
    5. Plots the resulting spectra with DIS/APO resolution
    
    Returns
    -------
    dict
        Dictionary containing:
        - xwav: wavelength grid [Angstroms]
        - ypts_single_maxwellian: single Maxwellian spectrum [R/Å]
        - ypts_double_maxwellian: double Maxwellian spectrum [R/Å]
        - xwavi_single: single Max emission line wavelengths [Angstroms]
        - yptsi_single: single Max emission line intensities [R]
        - xwavi_double: double Max emission line wavelengths [Angstroms]
        - yptsi_double: double Max emission line intensities [R]
    
    Observational Context
    ---------------------
    Optical observations complement UV by providing:
    - Density diagnostics from forbidden line ratios
    - Temperature diagnostics from auroral/nebular line ratios
    - Abundance measurements from multiple ionization states
    - Ground-based accessibility (unlike UV which requires space telescopes)
    """
    
    # ============================================================================
    # WAVELENGTH GRID SETUP FOR DIS/APO
    # ============================================================================
    # Define the wavelength range and resolution for DIS high-resolution mode
    # Optical range covers major forbidden and permitted S and O emission lines
    
    min_xwav = 3200.0       # Minimum wavelength [Angstroms] - blue channel limit
    max_xwav = 7500.0       # Maximum wavelength [Angstroms] - red channel limit
    xwav_bin_width = 0.62   # Spectral bin width [Angstroms] - DIS high-res mode
    
    # Create wavelength grid (bin centers)
    xwav = np.arange(min_xwav, max_xwav + xwav_bin_width, xwav_bin_width)
    
    # ============================================================================
    # DIS/APO INSTRUMENT PARAMETERS
    # ============================================================================
    # Full Width Half Maximum of instrument response function
    # For DIS high-resolution mode, FWHM ~ 3x pixel size
    fwhm = 1.86  # [Angstroms] - 3 × 0.62 Å for DIS high-res
    
    # ============================================================================
    # SINGLE MAXWELLIAN CALCULATION
    # ============================================================================
    print("="*64)
    print("SINGLE MAXWELLIAN CALCULATION - OPTICAL")
    print("="*64)
    
    # Initialize tables
    tables = EmissionTables()
    
    # Load single Maxwellian tables (same tables include optical lines)
    tables.load_single_maxwellian_tables('CHIANTI_11.0.2_emiss_single_maxwellian.h5')
    
    # Plasma parameters - single Maxwellian
    # Core/cold electron population parameters
    Te = 5.0          # Electron temperature [eV] - typical IPT value
    ne = 2200.0       # Electron density [cm^-3] - peak torus density
    
    # Column densities [cm^-2]
    # These correspond to ~6 R_J path length through the Io torus
    # Based on Nerney et al. 2017 and Steffl et al. 2004b
    column_densities = {
        'S+': 1.2e13,    # S II - strong optical forbidden lines
        'S++': 4.2e13,   # S III - dominant S ion, bright optical lines
        'S+++': 5.92e12, # S IV - weaker optical lines
        'S++++': 6.0e11, # S V - very weak in optical
        'O+': 5.2e13,    # O II - strong forbidden doublet
        'O++': 5.92e12   # O III - bright nebular lines
    }
    
    # Calculate emission lines
    xwavi_single, yptsi_single = calculate_ipt_emiss_tables_single(
        tables, Te, ne, column_densities, min_wav=min_xwav, max_wav=max_xwav
    )
    
    # ============================================================================
    # DOUBLE MAXWELLIAN CALCULATION
    # ============================================================================
    print()
    print("="*64)
    print("DOUBLE MAXWELLIAN CALCULATION - OPTICAL")
    print("="*64)
    
    # Load double Maxwellian tables
    tables.load_double_maxwellian_tables('CHIANTI_11.0.2_emiss_double_maxwellian.h5')
    
    # Plasma parameters - double Maxwellian
    # Magnetospheric plasma case with core + hot populations
    Tec = 5.0          # Core temperature [eV]
    Teh = 270.0        # Hot temperature [eV] - from wave heating
    feh = 0.0025       # Hot electron fraction (0.25% typical)
    fec = 1.0 - feh    # Core electron fraction
    nec = 2200.0       # Core density
    neh = nec * (1.0/fec - 1.0)  # Hot density
    ne_total = nec + neh          # Total density
    
    # Calculate emission lines
    xwavi_double, yptsi_double = calculate_ipt_emiss_tables_double(
        tables, Tec, Teh, ne_total, feh, column_densities, 
        min_wav=min_xwav, max_wav=max_xwav
    )
    
    # ============================================================================
    # CONVOLVE WITH DIS INSTRUMENT RESPONSE
    # ============================================================================
    # Create realistic spectra by convolving discrete lines with DIS PSF
    
    ypts_single_maxwellian = simulate_ipt_spectrum_rayleighs_erf_form(
        xwav, xwav_bin_width, xwavi_single, yptsi_single, fwhm=fwhm
    )
    
    ypts_double_maxwellian = simulate_ipt_spectrum_rayleighs_erf_form(
        xwav, xwav_bin_width, xwavi_double, yptsi_double, fwhm=fwhm
    )
    
    # ============================================================================
    # SUMMARY OUTPUT
    # ============================================================================
    print()
    print("="*64)
    print("SIMULATION COMPLETE - OPTICAL TABLE-BASED VERSION")
    print("="*64)
    print("Instrument Configuration:")
    print(f"  Telescope: Apache Point Observatory 3.5m")
    print(f"  Instrument: DIS (Dual Imaging Spectrograph)")
    print(f"  Mode: High resolution")
    print(f"  Pixel scale: {xwav_bin_width:.2f} Å/pixel")
    print(f"  Resolution (FWHM): {fwhm:.2f} Å")
    print(f"  Wavelength range: {min_xwav:.0f} - {max_xwav:.0f} Å")
    print()
    print("Plasma parameters used:")
    print(f"  Single Maxwellian: Te = {Te:12.7f} eV, ne = {ne:12.4f} cm^-3")
    print(f"  Double Maxwellian: Tec = {Tec:12.7f} eV, Teh = {Teh:12.5f} eV")
    print(f"                     feh = {feh:15.10f}, ne_total = {ne_total:12.4f} cm^-3")
    print("Column densities [cm^-2]:")
    for ion, value in column_densities.items():
        print(f"  {ion:<5}: {value:14.7e}")
    print("Number of emission lines in optical range:")
    print(f"  Single Maxwellian: {len(xwavi_single):11d}")
    print(f"  Double Maxwellian: {len(xwavi_double):11d}")
    print("Variables available for inspection:")
    print("  xwav                    - wavelength grid")
    print("  ypts_single_maxwellian  - single Maxwellian spectrum")
    print("  ypts_double_maxwellian  - double Maxwellian spectrum")
    print("  xwavi_single/double     - emission line wavelengths")
    print("  yptsi_single/double     - emission line intensities")
    
    # ============================================================================
    # PLOT RESULTS WITH KEY LINES MARKED
    # ============================================================================
    # Create publication-quality plots showing the simulated optical spectra
    
    # Define key diagnostic lines to mark
    key_lines = {
        '[O II] 3726,3729': 3727.5,
        '[S III] 6312': 6312,
        '[S II] 6716': 6716,
        '[S II] 6731': 6731,
        '[O III] 4959': 4959,
        '[O III] 5007': 5007
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Single Maxwellian plot
    axes[0].plot(xwav, ypts_single_maxwellian, 'b-', linewidth=1.0, alpha=0.9)
    axes[0].set_xlabel('Wavelength [Å]', fontsize=12)
    axes[0].set_ylabel('Intensity [R/Å]', fontsize=12)
    axes[0].set_title(f'IPT Optical Emission (DIS/APO): Single Maxwellian (Te={Te:.1f} eV, ne={ne:.0f} cm⁻³)', 
                     fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(min_xwav, max_xwav)
    axes[0].set_ylim(bottom=0)
    
    # Mark key lines
    ylim = axes[0].get_ylim()
    for label, wav in key_lines.items():
        axes[0].axvline(wav, color='gray', linestyle='--', alpha=0.3)
        axes[0].text(wav, ylim[1]*0.95, label, rotation=90, 
                    fontsize=8, ha='right', va='top')
    
    # Double Maxwellian plot
    axes[1].plot(xwav, ypts_double_maxwellian, 'r-', linewidth=1.0, alpha=0.9)
    axes[1].set_xlabel('Wavelength [Å]', fontsize=12)
    axes[1].set_ylabel('Intensity [R/Å]', fontsize=12)
    axes[1].set_title(f'IPT Optical Emission (DIS/APO): Double Maxwellian (Tec={Tec:.1f} eV, Teh={Teh:.0f} eV, feh={feh:.4f})', 
                     fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(min_xwav, max_xwav)
    axes[1].set_ylim(bottom=0)
    
    # Mark key lines
    ylim = axes[1].get_ylim()
    for label, wav in key_lines.items():
        axes[1].axvline(wav, color='gray', linestyle='--', alpha=0.3)
        axes[1].text(wav, ylim[1]*0.95, label, rotation=90, 
                    fontsize=8, ha='right', va='top')
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # COMPARISON PLOT
    # ============================================================================
    # Direct comparison of single vs double Maxwellian in optical
    
    fig2, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(xwav, ypts_single_maxwellian, 'b-', linewidth=1.0, 
            label='Single Maxwellian', alpha=0.8)
    ax.plot(xwav, ypts_double_maxwellian, 'r-', linewidth=1.0, 
            label='Double Maxwellian', alpha=0.8)
    ax.set_xlabel('Wavelength [Å]', fontsize=12)
    ax.set_ylabel('Intensity [R/Å]', fontsize=12)
    ax.set_title('IPT Optical Emission Comparison (DIS/APO): Single vs Double Maxwellian', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_xwav, max_xwav)
    ax.set_ylim(bottom=0)
    
    # Mark key diagnostic lines
    ylim = ax.get_ylim()
    for label, wav in key_lines.items():
        ax.axvline(wav, color='gray', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # ZOOM PLOTS FOR KEY DIAGNOSTIC REGIONS
    # ============================================================================
    # Create zoomed plots for important diagnostic line regions
    
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Region 1: [O II] 3726,3729 doublet
    ax = axes[0, 0]
    mask = (xwav >= 3700) & (xwav <= 3750)
    ax.plot(xwav[mask], ypts_single_maxwellian[mask], 'b-', label='Single', linewidth=1.5)
    ax.plot(xwav[mask], ypts_double_maxwellian[mask], 'r-', label='Double', linewidth=1.5)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Intensity [R/Å]')
    ax.set_title('[O II] 3726,3729 Doublet Region')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Region 2: [O III] 4959,5007 nebular lines
    ax = axes[0, 1]
    mask = (xwav >= 4900) & (xwav <= 5050)
    ax.plot(xwav[mask], ypts_single_maxwellian[mask], 'b-', label='Single', linewidth=1.5)
    ax.plot(xwav[mask], ypts_double_maxwellian[mask], 'r-', label='Double', linewidth=1.5)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Intensity [R/Å]')
    ax.set_title('[O III] 4959,5007 Nebular Lines')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Region 3: [S III] 6312 auroral line
    ax = axes[1, 0]
    mask = (xwav >= 6290) & (xwav <= 6330)
    ax.plot(xwav[mask], ypts_single_maxwellian[mask], 'b-', label='Single', linewidth=1.5)
    ax.plot(xwav[mask], ypts_double_maxwellian[mask], 'r-', label='Double', linewidth=1.5)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Intensity [R/Å]')
    ax.set_title('[S III] 6312 Auroral Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Region 4: [S II] 6716,6731 doublet
    ax = axes[1, 1]
    mask = (xwav >= 6700) & (xwav <= 6750)
    ax.plot(xwav[mask], ypts_single_maxwellian[mask], 'b-', label='Single', linewidth=1.5)
    ax.plot(xwav[mask], ypts_double_maxwellian[mask], 'r-', label='Double', linewidth=1.5)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Intensity [R/Å]')
    ax.set_title('[S II] 6716,6731 Doublet (Density Diagnostic)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Key Diagnostic Line Regions - DIS/APO Resolution', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Return results for further analysis
    return {
        'xwav': xwav,
        'ypts_single_maxwellian': ypts_single_maxwellian,
        'ypts_double_maxwellian': ypts_double_maxwellian,
        'xwavi_single': xwavi_single,
        'yptsi_single': yptsi_single,
        'xwavi_double': xwavi_double,
        'yptsi_double': yptsi_double
    }


if __name__ == "__main__":
    # Run the main example
    results = basic_example_optical_cm3_emission_model_use_tables()
    
    # Results are now available in the 'results' dictionary for further analysis
    print("\nResults stored in 'results' dictionary for further analysis.")
    print("\nKey diagnostic line ratios can be calculated from the results:")
    print("  - [S II] 6731/6716 for electron density")
    print("  - [S III] 6312/9069 for electron temperature")
    print("  - [O II] 3729/3726 for density in low-density limit")
    print("  - [O III] 4363/(4959+5007) for temperature")