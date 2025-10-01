;+
; AUTHOR: Edward G. Nerney
; 
; UV EMISSION MODEL FOR IO PLASMA TORUS
; ======================================
; This code simulates UV emission spectra from the Io Plasma Torus (IPT) in
; Jupiter's magnetosphere using the CHIANTI atomic database.
;
; PHYSICAL MODEL:
; - Uses the "cubic-cm" emission model for optically thin plasma
; - Calculates emission from electron impact excitation of S and O ions
; - Supports both single and double Maxwellian electron distributions
; - Converts atomic emissivities to observable Rayleigh intensities
;
; KEY ASSUMPTIONS:
; - Optically thin plasma (no absorption/scattering)
; - Electron impact excitation dominates (photoexcitation negligible)
; - Local thermodynamic equilibrium NOT assumed (uses CHIANTI non-LTE calculations)
; - Line-of-sight integration approximated by column densities
;-

;==============================================================================
function calculate_IPT_emiss_te_same_size_ne, Tel, nel, Nsp, Ns2p, Ns3p,Ns4p, Nop, No2p, $
  min = min, max = max, xwavi=xwavi
  ;+
  ; NAME:
  ;   calculate_IPT_emiss_te_same_size_ne
  ;
  ; PURPOSE:
  ;   Calculate UV emission line intensities for multiple temperature/density pairs
  ;   simultaneously. This is the vectorized version for efficiency when computing
  ;   multiple plasma conditions.
  ;
  ; INPUTS:
  ;   Tel    - Electron temperature array [eV]
  ;   nel    - Electron number density array [#/cm^3]
  ;   Nsp    - S+ (S II) column density array [#/cm^2]
  ;   Ns2p   - S++ (S III) column density array [#/cm^2]
  ;   Ns3p   - S+++ (S IV) column density array [#/cm^2]
  ;   Ns4p   - S++++ (S V) column density array [#/cm^2]
  ;   Nop    - O+ (O II) column density array [#/cm^2]
  ;   No2p   - O++ (O III) column density array [#/cm^2]
  ;
  ; KEYWORDS:
  ;   min    - Minimum wavelength for spectral range [Angstroms], default=550
  ;   max    - Maximum wavelength for spectral range [Angstroms], default=1800
  ;   xwavi  - OUTPUT: Array of emission line wavelengths [Angstroms]
  ;
  ; OUTPUTS:
  ;   Returns 2D array [n_conditions, n_lines] of line intensities in Rayleighs
  ;
  ; PHYSICS:
  ;   Uses CHIANTI emissivities (photons/s/cm^3) multiplied by column densities
  ;   to get line-integrated intensities in Rayleighs (10^6 photons/s/cm^2/4π sr)
  ;-

  ; Set wavelength range defaults if not specified
  if keyword_set(max) then maxwav = max else maxwav = 1800d
  if keyword_set(min) then minwav = min else minwav = 550d

  ; Get number of temperature/density conditions to calculate
  num_tel=n_elements(tel)

  ; UNIT CONVERSION: eV to Kelvin
  ; From NIST: 1 eV = 11604.51812 K (https://physics.nist.gov/cuu/Constants)
  conversion=11604.51812d

  ; Convert to log10 scale as required by CHIANTI routines
  log10tel=alog10(Tel*conversion)  ; Log10 of temperature in Kelvin
  log10nel=alog10(nel)              ; Log10 of electron density in cm^-3

  ; ============================================================================
  ; CALCULATE EMISSIVITIES USING CHIANTI
  ; ============================================================================
  ; emiss_calc returns emissivities in units of photons/s/cm^3
  ; These are volume emissivities that must be integrated along line of sight
  ;
  ; Key parameters used:
  ;   /no_de     - Drops the hc/lambda factor in the computation of the 
  ;   emissivities. Useful for emission measure analyses involving 
  ;   photon fluxes
  ;   radt=1.d   - Specify background radiation temperature (default: 6000 K) set to 1 to neglect this
  ;   /quiet     - Suppress informational messages
  ;   /NOPROT    - Exclude proton collision rates
  ;   /NOIONREC  - Exclude ionization/recombination (use fixed ion fractions)
  ;   /NO_RREC   - Exclude radiative recombination

  ; Calculate emissivities for Sulfur ions (S, S+, S++, S+++, S++++)
  ; Atomic number for Sulfur = 16
  
  ;s1em = emiss_calc(16, 1, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s2em = emiss_calc(16, 2, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s3em = emiss_calc(16, 3, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s4em = emiss_calc(16, 4, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s5em = emiss_calc(16, 5, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)

  ; Calculate emissivities for Oxygen ions (O, O+, O++)
  ; Atomic number for Oxygen = 8
  ;o1em = emiss_calc(8, 1, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  o2em = emiss_calc(8, 2, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  o3em = emiss_calc(8, 3, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)

  ; Na+
  ;na2em = emiss_calc(11, 2, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)

  ; ============================================================================
  ; ORGANIZE WAVELENGTHS AND ION NAMES
  ; ============================================================================
  ; Combine all wavelengths from different ions
  ;xwavi = [s1em.lambda,s2em.lambda, s3em.lambda, s4em.lambda,s5em.lambda,o1em.lambda, o2em.lambda, o3em.lambda,na2em.lambda]
  xwavi = [s2em.lambda, s3em.lambda, s4em.lambda,s5em.lambda, o2em.lambda, o3em.lambda]

  ; Sort by wavelength for organized output
  wsort = sort(xwavi)
  xwavi = xwavi[wsort]

  ; Keep track of ion names for line identification
  ;yname = [s1em.ion_name ,s2em.ion_name, s3em.ion_name, s4em.ion_name, s5em.ion_name, o1em.ion_name, o2em.ion_name, o3em.ion_name, na2em.ion_name]
  yname = [s2em.ion_name, s3em.ion_name, s4em.ion_name, s5em.ion_name, o2em.ion_name, o3em.ion_name]

  yname = yname[wsort]

  ; ============================================================================
  ; FILTER TO REQUESTED WAVELENGTH RANGE
  ; ============================================================================
  avgwav = (minwav + maxwav)/2.d
  wrange = where(abs(xwavi - avgwav) le avgwav - minwav)
  xwavi = xwavi[wrange]
  yname = yname[wrange]

  ; ============================================================================
  ; CALCULATE LINE INTENSITIES
  ; ============================================================================
  ; Initialize output array [n_conditions, n_lines]
  yptsi=dblarr(num_tel,n_elements(xwavi))

  ; Loop through each temperature/density condition
  for i=0, num_tel - 1 do begin
    ; Extract emissivities for this condition and multiply by column densities
    ; The factor of 1d-6 converts to Rayleighs (10^6 photons/s/cm^2/4π sr)
    ; reform() extracts the [i,i] diagonal element (same Te and ne index)
    ;yptsi_temp = (1d-6)*[reform(s1em.em[i,i,*])*Ns[i] ,reform(s2em.em[i,i,*])*Nsp[i], $
    ;  reform(s3em.em[i,i,*])*Ns2p[i], $
     ; reform(s4em.em[i,i,*])*Ns3p[i], $
     ; reform(s5em.em[i,i,*])*Ns4p[i], $
     ; reform(o1em.em[i,i,*])*No[i], $
     ; reform(o2em.em[i,i,*])*Nop[i], $
     ; reform(o3em.em[i,i,*])*No2p[i],$
     ; reform(na2em.em[i,i,*])*Nnap[i]]
     
     yptsi_temp = (1d-6)*[reform(s2em.em[i,i,*])*Nsp[i], $
       reform(s3em.em[i,i,*])*Ns2p[i], $
       reform(s4em.em[i,i,*])*Ns3p[i], $
       reform(s5em.em[i,i,*])*Ns4p[i], $
     reform(o2em.em[i,i,*])*Nop[i], $
       reform(o3em.em[i,i,*])*No2p[i]]

    ; Apply wavelength sorting and range filtering
    yptsi_temp= yptsi_temp[wsort]
    yptsi[i,*] =yptsi_temp[wrange]
  endfor

  return, yptsi
  ; Note: xwavi is returned as a keyword output
end

;==============================================================================
function calculate_IPT_emiss_citep2, Tel, nel, Nsp, Ns2p, Ns3p,Ns4p, Nop, No2p, $
  min = min, max = max, xwavi=xwavi
  ;+
  ; NAME:
  ;   calculate_IPT_emiss_citep2
  ;
  ; PURPOSE:
  ;   Calculate UV emission line intensities for a SINGLE Maxwellian electron
  ;   distribution. This is the standard case for thermal plasma.
  ;
  ; INPUTS:
  ;   Tel    - Electron temperature [eV] - single value
  ;   nel    - Electron number density [#/cm^3] - single value
  ;   Nsp    - S+ (S II) column density [#/cm^2]
  ;   Ns2p   - S++ (S III) column density [#/cm^2]
  ;   Ns3p   - S+++ (S IV) column density [#/cm^2]
  ;   Ns4p   - S++++ (S V) column density [#/cm^2]
  ;   Nop    - O+ (O II) column density [#/cm^2]
  ;   No2p   - O++ (O III) column density [#/cm^2]
  ;
  ; KEYWORDS:
  ;   min    - Minimum wavelength [Angstroms], default=550
  ;   max    - Maximum wavelength [Angstroms], default=1800
  ;   xwavi  - OUTPUT: Array of emission line wavelengths [Angstroms]
  ;
  ; OUTPUTS:
  ;   Returns 1D array of line intensities in Rayleighs
  ;
  ; PHYSICS NOTES:
  ;   - Assumes electron velocity distribution is Maxwellian at temperature Tel
  ;   - Emissivities from CHIANTI include:
  ;     * Electron impact excitation rates (dominant process)
  ;     * Cascades from higher levels
  ;     * Fine structure transitions
  ;   - The Einstein A coefficients (A_ji) give spontaneous emission rates
  ;   - Level populations (N_j) calculated assuming statistical equilibrium
  ;-

  ; Set wavelength range
  if keyword_set(max) then maxwav = max else maxwav = 1800d
  if keyword_set(min) then minwav = min else minwav = 550d

  ; Convert units for CHIANTI
  conversion=11604.51812d  ; eV to Kelvin conversion factor (NIST)
  log10tel=alog10(Tel*conversion)
  log10nel=alog10(nel)

  ; ============================================================================
  ; DETAILED PHYSICS OF CHIANTI EMISSIVITY CALCULATIONS
  ; ============================================================================
  ; emiss_calc solves the statistical equilibrium equations:
  ;   dN_j/dt = 0 = Sum_i(N_i * R_ij) - N_j * Sum_k(R_jk)
  ;
  ; Where R_ij includes:
  ;   - Electron impact excitation/de-excitation
  ;   - Radiative decay (A_ji Einstein coefficients)
  ;   - NOT included here: photoexcitation, proton collisions, recombination
  ;
  ; The output emissivity is: N_j * A_ji (photons/s/cm^3)
  ; Where N_j is the fractional population of upper level j

  ; Calculate Sulfur ion emissivities
  s2em = emiss_calc(16, 2, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s3em = emiss_calc(16, 3, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s4em = emiss_calc(16, 4, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s5em = emiss_calc(16, 5, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)

  ; Calculate Oxygen ion emissivities
  o2em = emiss_calc(8, 2, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  o3em = emiss_calc(8, 3, temp = log10tel, dens = log10nel, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)

  ; ============================================================================
  ; CONVERT TO OBSERVABLE INTENSITIES
  ; ============================================================================
  ; Multiply volume emissivity by column density and convert to Rayleighs
  ; 1 Rayleigh = 10^6 photons/s/cm^2/4π sr (hence the 1d-6 factor)
  s2emiss = (1d-6)*reform(s2em.em)*Nsp    ; S II lines
  s3emiss = (1d-6)*reform(s3em.em)*Ns2p   ; S III lines
  s4emiss = (1d-6)*reform(s4em.em)*Ns3p   ; S IV lines
  s5emiss=  (1d-6)*reform(s5em.em)*Ns4p   ; S V lines
  o2emiss = (1d-6)*reform(o2em.em)*Nop    ; O II lines
  o3emiss = (1d-6)*reform(o3em.em)*No2p   ; O III lines

  ; Organize wavelengths and intensities
  xwavi = [s2em.lambda, s3em.lambda, s4em.lambda,s5em.lambda, o2em.lambda, o3em.lambda]
  wsort = sort(xwavi)
  xwavi = xwavi[wsort]

  yptsi = [s2emiss, s3emiss, s4emiss, s5emiss, o2emiss, o3emiss]
  yptsi = yptsi[wsort]

  yname = [s2em.ion_name, s3em.ion_name, s4em.ion_name, s5em.ion_name, o2em.ion_name, o3em.ion_name]
  yname = yname[wsort]

  ; Filter to requested wavelength range
  avgwav = (minwav + maxwav)/2.d
  wrange = where(abs(xwavi - avgwav) le avgwav - minwav)
  xwavi = xwavi[wrange]
  yptsi = yptsi[wrange]
  yname = yname[wrange]

  return, yptsi
end

;==============================================================================
function calculate_IPT_emiss_citep2_double_max, Tec,Teh, ne_total,feh, Nsp, Ns2p, Ns3p,Ns4p, Nop, No2p, $
  min = min, max = max, xwavi=xwavi
  ;+
  ; NAME:
  ;   calculate_IPT_emiss_citep2_double_max
  ;
  ; PURPOSE:
  ;   Calculate UV emission for a DOUBLE Maxwellian electron distribution.
  ;   This represents plasma with both cold/core and hot electron populations,
  ;   common in magnetospheric plasmas.
  ;
  ; INPUTS:
  ;   Tec      - Core (cold) electron temperature [eV]
  ;   Teh      - Hot electron temperature [eV]
  ;   ne_total - Total electron density [#/cm^3]
  ;   feh      - Fraction of hot electrons (0 to 1)
  ;   Nsp      - S+ column density [#/cm^2]
  ;   Ns2p     - S++ column density [#/cm^2]
  ;   Ns3p     - S+++ column density [#/cm^2]
  ;   Ns4p     - S++++ column density [#/cm^2]
  ;   Nop      - O+ column density [#/cm^2]
  ;   No2p     - O++ column density [#/cm^2]
  ;
  ; KEYWORDS:
  ;   min      - Minimum wavelength [Angstroms]
  ;   max      - Maximum wavelength [Angstroms]
  ;   xwavi    - OUTPUT: Emission line wavelengths
  ;
  ; PHYSICS:
  ;   The double Maxwellian represents:
  ;   f(v) = fec * f_Maxwell(v,Tec) + feh * f_Maxwell(v,Teh)
  ;   where fec + feh = 1
  ;
  ;   This is important for:
  ;   - Magnetospheric plasmas with wave-particle heating
  ;   - Pickup ion populations
  ;   - Reconnection regions
  ;-

  if keyword_set(max) then maxwav = max else maxwav = 1800d
  if keyword_set(min) then minwav = min else minwav = 550d

  ; Convert temperatures to Kelvin (log scale for CHIANTI)
  conversion=11604.51812d
  log10tec=alog10(Tec*conversion)
  log10teh=alog10(Teh*conversion)
  log10ne_total=alog10(ne_total)

  ; Calculate fraction of cold electrons
  fec = 1.d - feh

  ; ============================================================================
  ; SETUP FOR CHIANTI'S MULTI-TEMPERATURE CAPABILITY
  ; ============================================================================
  ; CHIANTI can handle multiple Maxwellian components using sum_mwl_coeff
  ; This properly weights the contribution from each temperature component
  log10_temps = [log10tec, log10teh]  ; Array of temperatures
  fracs = [fec, feh]                   ; Fractional weights (must sum to 1)

  ; ============================================================================
  ; CALCULATE WEIGHTED EMISSIVITIES
  ; ============================================================================
  ; The sum_mwl_coeff keyword tells CHIANTI to calculate:
  ; emiss_total = fec * emiss(Tec) + feh * emiss(Teh)
  ; This accounts for different excitation rates at different temperatures

  ; Sulfur ions with double Maxwellian
  s2em = emiss_calc(16, 2, temp = log10_temps, sum_mwl_coeff=fracs, dens = log10ne_total, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s3em = emiss_calc(16, 3, temp = log10_temps, sum_mwl_coeff=fracs, dens = log10ne_total, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s4em = emiss_calc(16, 4, temp = log10_temps, sum_mwl_coeff=fracs, dens = log10ne_total, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  s5em = emiss_calc(16, 5, temp = log10_temps, sum_mwl_coeff=fracs, dens = log10ne_total, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)

  ; Oxygen ions with double Maxwellian
  o2em = emiss_calc(8, 2, temp = log10_temps, sum_mwl_coeff=fracs, dens = log10ne_total, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)
  o3em = emiss_calc(8, 3, temp = log10_temps, sum_mwl_coeff=fracs, dens = log10ne_total, /no_de, radt = 1.d, /quiet,/NOPROT,/NOIONREC,/NO_RREC)

  ; Convert to Rayleighs
  s2emiss = (1d-6)*reform(s2em.em)*Nsp
  s3emiss = (1d-6)*reform(s3em.em)*Ns2p
  s4emiss = (1d-6)*reform(s4em.em)*Ns3p
  s5emiss=  (1d-6)*reform(s5em.em)*Ns4p
  o2emiss = (1d-6)*reform(o2em.em)*Nop
  o3emiss = (1d-6)*reform(o3em.em)*No2p

  ; Organize and sort results
  xwavi = [s2em.lambda, s3em.lambda, s4em.lambda,s5em.lambda, o2em.lambda, o3em.lambda]
  wsort = sort(xwavi)
  xwavi = xwavi[wsort]

  yptsi = [s2emiss, s3emiss, s4emiss, s5emiss, o2emiss, o3emiss]
  yptsi = yptsi[wsort]

  yname = [s2em.ion_name, s3em.ion_name, s4em.ion_name, s5em.ion_name, o2em.ion_name, o3em.ion_name]
  yname = yname[wsort]

  ; Filter to wavelength range
  avgwav = (minwav + maxwav)/2.d
  wrange = where(abs(xwavi - avgwav) le avgwav - minwav)
  xwavi = xwavi[wrange]
  yptsi = yptsi[wrange]
  yname = yname[wrange]

  return, yptsi
end

;==============================================================================
function simulate_IPT_spectrum_Rayleighs_citep2_ERF_form, x, spec_binsize, xwavi, yptsi, $
  fwhm = fwhm
  ;+
  ; NAME:
  ;   simulate_IPT_spectrum_Rayleighs_citep2_ERF_form
  ;
  ; PURPOSE:
  ;   Convolve discrete emission lines with instrument response function to
  ;   create a realistic spectrum as would be observed by a spectrograph.
  ;
  ; INPUTS:
  ;   x            - Wavelength grid for output spectrum [Angstroms]
  ;   spec_binsize - Width of wavelength bins [Angstroms]
  ;   xwavi        - Wavelengths of emission lines [Angstroms]
  ;   yptsi        - Intensities of emission lines [Rayleighs]
  ;
  ; KEYWORDS:
  ;   fwhm         - Full Width Half Maximum of Gaussian instrument response [Angstroms]
  ;
  ; METHOD:
  ;   Uses Error Function (ERF) formulation for exact integration of Gaussian
  ;   line profile over finite wavelength bins. This is more accurate than
  ;   simple Gaussian evaluation at bin centers.
  ;
  ; MATHEMATICS:
  ;   For a Gaussian with FWHM, the integral over a bin [λ-Δλ/2, λ+Δλ/2] is:
  ;   I_bin = I_line * 0.5 * [ERF((λ_bin_upper - λ_line)/σ√2) - ERF((λ_bin_lower - λ_line)/σ√2)]
  ;   where σ = FWHM / (2√(2ln2))
  ;-

  ; ============================================================================
  ; GAUSSIAN PROFILE PARAMETERS
  ; ============================================================================
  ; Relationship between FWHM and Gaussian standard deviation σ:
  ; FWHM = 2σ√(2ln2) ≈ 2.355σ
  ; Therefore: σ = FWHM / (2√(2ln2))
  ;
  ; For the ERF formulation, we need 1/(σ√2), which equals:
  rootc = 2d*sqrt(alog(2d))/fwhm  ; This is 1/(σ√2)

  ; Get wavelength range from input grid
  maxwav = max(x)
  minwav = min(x)
  nj = n_elements(x)    ; Number of wavelength bins
  ni = n_elements(yptsi) ; Number of emission lines

  ; Initialize output spectrum
  ypts = dblarr(nj)

  ; ============================================================================
  ; CONVOLVE EACH LINE WITH INSTRUMENT RESPONSE
  ; ============================================================================
  ; Loop through each discrete emission line
  for i=0, ni-1 do begin
    ; For each line, calculate its contribution to each wavelength bin
    ; using the exact integral of the Gaussian over the bin
    ;
    ; The ERF formula gives the integral of a normalized Gaussian
    ; We multiply by the line intensity (yptsi) and divide by bin width
    ; to get intensity per unit wavelength

    ypts += yptsi(i) * 0.5d * $
      (Erf((x - xwavi(i) + spec_binsize/2d)*rootc) - $
      Erf((x - xwavi(i) - spec_binsize/2d)*rootc))
  endfor

  ; Convert from integrated intensity to intensity per unit wavelength
  ypts /= spec_binsize

  return, ypts
end

;==============================================================================
pro basic_example_UV_cm3_emission_model_use
  ;+
  ; NAME:
  ;   basic_example_UV_cm3_emission_model_use
  ;
  ; PURPOSE:
  ;   Main procedure demonstrating UV emission calculations for the Io Plasma
  ;   Torus using CHIANTI atomic data. Shows examples with single and double
  ;   Maxwellian electron distributions.
  ;
  ; DESCRIPTION:
  ;   This procedure:
  ;   1. Sets up typical IPT plasma parameters at ~6 Jupiter radii
  ;   2. Calculates UV emission spectra for single Maxwellian case
  ;   3. Calculates UV emission spectra for double Maxwellian case
  ;   4. Demonstrates vectorized calculations for multiple conditions
  ;   5. Plots the resulting spectra
  ;
  ; OUTPUTS:
  ;   Creates plots showing simulated UV spectra in Rayleighs/Angstrom
  ;
  ; REFERENCES:
  ;   - Nerney et al. 2017, 2020, 2022, & 2025
  ;   - Steffl et al. 2004b
  ;   - CHIANTI atomic database (Dere et al. 1997, Del Zanna et al. 2020, & Dufresne et al. 2024 ) 
  ;   - CHIANTI is a collaborative project involving George Mason University,
  ;     the University of Michigan (USA), University of Cambridge (UK) 
  ;     and NASA Goddard Space Flight Center (USA).


  ;-

  ; ============================================================================
  ; WAVELENGTH GRID SETUP
  ; ============================================================================
  ; Define the wavelength range and resolution for the simulation
  ; Typical UV (EUV and FUV) range covers major S and O emission lines

  min_xwav = 550d        ; Minimum wavelength [Angstroms] - 
  max_xwav = 1750d       ; Maximum wavelength [Angstroms] - 
  xwav_bin_width = 1d    ; Spectral bin width [Angstroms] - typical for UV spectrographs

  ; Calculate number of wavelength bins
  num_xwav_points = floor((max_xwav - min_xwav)/xwav_bin_width) + 1

  ; Create wavelength grid (bin centers)
  xwav = xwav_bin_width*dindgen(num_xwav_points) + min_xwav  ; [Angstroms]

  ; ============================================================================
  ; INSTRUMENT PARAMETERS
  ; ============================================================================
  ; Full Width Half Maximum of instrument response function
  ; This represents the spectral resolution of the spectrograph
  fwhm = 4d  ; [Angstroms] - typical for space-based UV spectrographs

  ; ============================================================================
  ; PLASMA PARAMETERS - SINGLE MAXWELLIAN
  ; ============================================================================
  ; Core/cold electron population parameters
  Tec = 5d        ; Core electron temperature [eV] - typical IPT value
  nec = 2200d     ; Core electron density [#/cm^3] - peak torus density

  ; ============================================================================
  ; PLASMA PARAMETERS - DOUBLE MAXWELLIAN
  ; ============================================================================
  ; Hot electron component (suprathermal population)
  Teh = 270d      ; Hot electron temperature [eV] - from wave heating
  feh = 0.0025d   ; Fraction of hot electrons (0.25% typical)
  fec = 1d - feh  ; Fraction of cold electrons

  ; Calculate densities for double Maxwellian
  neh = nec*(1d/fec - 1d)  ; Hot electron density
  ne_total = nec + neh      ; Total electron density

  ; ============================================================================
  ; COLUMN DENSITIES AND ION COMPOSITION
  ; ============================================================================
  ; Total electron column density for line-of-sight through torus
  ; This corresponds to ~6 R_J path length through the Io torus
  Ne_column = 2d14  ; [cm^-2] - integrated electron density along LOS

  ; Ion mixing ratios at 6 R_J (from observations)
  ; These are number density ratios relative to total electron density
  ; Based on Nerney et al. 2017 and Steffl et al. 2004b

  ; Sulfur ion mixing ratios
  Nsp_mixr = 0.06d     ; S+ (S II) - singly ionized
  Ns2p_mixr = 0.21d    ; S++ (S III) - dominant S ion
  Ns3p_mixr = 0.0296d  ; S+++ (S IV)
  Ns4p_mixr = 0.003d   ; S++++ (S V) - highly ionized

  ; Oxygen ion mixing ratios
  Nop_mixr = 0.26d     ; O+ (O II) - dominant ion overall
  No2p_mixr = 0.0296d  ; O++ (O III)

  ; Other species for charge neutrality
  Nhp_mixr = 0.1d      ; H+ (protons) and other minor species

  ; ============================================================================
  ; CHARGE NEUTRALITY CHECK
  ; ============================================================================
  ; Verify charge neutrality: Σ(n_i * Z_i) / n_e = 1
  ; where n_i is ion density, Z_i is charge state, n_e is electron density
  ;
  ; neutrality_check = Nsp_mixr + Nop_mixr + Nhp_mixr +
  ;                    2*(Ns2p_mixr + No2p_mixr) +
  ;                    3*Ns3p_mixr + 4*Ns4p_mixr
  ; print, 'Charge neutrality check (should be 1.0):', neutrality_check

  ; Calculate ion column densities from mixing ratios
  Nsp = Ne_column*Nsp_mixr    ; S+ column density [cm^-2]
  Ns2p = Ne_column*Ns2p_mixr  ; S++ column density [cm^-2]
  Ns3p = Ne_column*Ns3p_mixr  ; S+++ column density [cm^-2]
  Ns4p = Ne_column*Ns4p_mixr  ; S++++ column density [cm^-2]
  Nop = Ne_column*Nop_mixr    ; O+ column density [cm^-2]
  No2p = Ne_column*No2p_mixr  ; O++ column density [cm^-2]

  ; ============================================================================
  ; CALCULATE EMISSION - SINGLE MAXWELLIAN
  ; ============================================================================
  ; Standard thermal plasma case with one electron temperature
  ; Key assumptions:
  ; - Single Maxwellian electron distribution at Tec
  ; - Uniform density nec along line of sight (or peak-dominated)
  ; - Electron impact excitation dominates
  ; - Optically thin plasma

  print, 'Calculating single Maxwellian emission...'
  yptsi_single_maxwellian = calculate_IPT_emiss_citep2(Tec, nec, $
    Nsp, Ns2p, Ns3p, Ns4p, Nop, No2p, $
    min = min_xwav, max = max_xwav, xwavi=xwavi)

  ; Convolve with instrument response
  ypts_single_maxwellian = simulate_IPT_spectrum_Rayleighs_citep2_ERF_form( $
    xwav, xwav_bin_width, xwavi, yptsi_single_maxwellian, fwhm = fwhm)

  ; ============================================================================
  ; CALCULATE EMISSION - DOUBLE MAXWELLIAN
  ; ============================================================================
  ; Magnetospheric plasma case with core + hot populations
  ; Represents effects of wave-particle interactions, pickup ions, etc.

  print, 'Calculating double Maxwellian emission...'
  yptsi_double_maxwellian = calculate_IPT_emiss_citep2_double_max( $
    Tec, Teh, ne_total, feh, $
    Nsp, Ns2p, Ns3p, Ns4p, Nop, No2p, $
    min = min_xwav, max = max_xwav, xwavi=xwavi)

  ; Convolve with instrument response
  ypts_double_maxwellian = simulate_IPT_spectrum_Rayleighs_citep2_ERF_form( $
    xwav, xwav_bin_width, xwavi, yptsi_double_maxwellian, fwhm = fwhm)

  ; ============================================================================
  ; PLOT RESULTS
  ; ============================================================================
  ; Create plots showing the simulated spectra

  p1 = plot(xwav, ypts_single_maxwellian, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (Rayleighs/Å)', $
    title='IPT UV Emission: Single Maxwellian (T$_{e}$=' + string(Tec,format='(F4.1)') + ' eV,' + 'n$_{e}$=' + string(nec,format='(F6.1)') +' cm$^{-3}$)')

  p2 = plot(xwav, ypts_double_maxwellian, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (Rayleighs/Å)', $
    title='IPT UV Emission: Double Maxwellian (T$_{ec}$=' + string(Tec,format='(F4.1)') + $
    ' eV, T$_{eh}$=' + string(Teh,format='(F5.1)') + ' eV, f$_{eh}$=' + string(feh,format='(F6.4)') + ')')

  ; ============================================================================
  ; DEMONSTRATE VECTORIZED CALCULATIONS
  ; ============================================================================
  ; Example of calculating multiple plasma conditions simultaneously
  ; Useful for parameter studies or fitting observations

  print, 'Demonstrating vectorized calculations for multiple conditions...'

  ; Define arrays of conditions to calculate
  necs = [2200d, 3000d]  ; Different electron densities [cm^-3]
  Tecs = [5d, 6d]        ; Different electron temperatures [eV]

  ; Replicate column densities for each condition
  Nsp_array = [Nsp, Nsp]
  Ns2p_array = [Ns2p, Ns2p]
  Ns3p_array = [Ns3p, Ns3p]
  Ns4p_array = [Ns4p, Ns4p]
  Nop_array = [Nop, Nop]
  No2p_array = [No2p, No2p]

  ; Calculate all conditions at once
  yptsi_multiple = calculate_IPT_emiss_te_same_size_ne(Tecs, necs, $
    Nsp_array, Ns2p_array, Ns3p_array, Ns4p_array, Nop_array, No2p_array, $
    min = min_xwav, max = max_xwav, xwavi=xwavi)

  ; Convolve each result with instrument response
  ypts3 = simulate_IPT_spectrum_Rayleighs_citep2_ERF_form(xwav, xwav_bin_width, $
    xwavi, reform(yptsi_multiple[0,*]), fwhm = fwhm)

  ypts4 = simulate_IPT_spectrum_Rayleighs_citep2_ERF_form(xwav, xwav_bin_width, $
    xwavi, reform(yptsi_multiple[1,*]), fwhm = fwhm)

  ; Plot the multiple conditions
  p3 = plot(xwav, ypts3, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (Rayleighs/Å)', $
    title='IPT UV Emission: T$_{e}$=' + string(Tecs[0],format='(F4.1)') + $
    ' eV, n$_{e}$=' + string(necs[0],format='(F6.1)') + ' cm$^{-3}$')

  p4 = plot(xwav, ypts4, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (Rayleighs/Å)', $
    title='IPT UV Emission: T$_{e}$=' + string(Tecs[1],format='(F4.1)') + $
    ' eV, n$_{e}$=' + string(necs[1],format='(F6.1)') + ' cm$^{-3}$')

  print, 'Simulation complete. Use "stop" command to examine variables.'
  stop  ; Pause to allow examination of results

end