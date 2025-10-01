;+
; NAME:
;   sim_citep_2.pro
;
; PURPOSE:
;   Calculate UV emission spectra from the Io Plasma Torus (IPT) in Jupiter's
;   magnetosphere with proper line-of-sight integration through the torus.
;   Uses pre-calculated CHIANTI atomic emission tables.
;
; DESCRIPTION:
;   This code simulates UV emission (550-2100 Å) from sulfur and oxygen ions
;   in the Io Plasma Torus by integrating emissivities along the line of sight
;   through the torus. The emission at each wavelength is calculated as:
;
;     B_λ = 10^(-6) × ∫ ε_λ(T_e(s), n_e(s)) × n_ion(s) ds
;
;   where:
;     B_λ = Brightness in Rayleighs
;     ε_λ = Volume emissivity from CHIANTI [photons/s/cm^3]
;     T_e = Electron temperature [eV]
;     n_e = Electron density [cm^-3]
;     n_ion = Ion density [cm^-3]
;     s = Distance along line of sight [cm]
;
; PHYSICAL MODEL:
;   - Torus extends from 4.5 to 10 R_J radially, ±3 R_J vertically
;   - Ion temperatures from polynomial fits (Nerney et al. 2017)
;   - Electron temperature follows power law: T_e = 5×(r/6)^2.48 eV
;   - Ion densities follow power laws with Gaussian vertical profiles
;   - Scale heights calculated including ambipolar electric field (Bagenal 1994)
;   - Quasi-neutrality enforced: n_e = Σ Z_i × n_i
;
; REQUIRED INPUT FILES:
;   CHIANTI_11.0.2_emiss_arrays_all_species_all_wavelengths_50x50_logspaced.sav
;   containing:
;     - temp_arr: Temperature grid [eV]
;     - dens_arr: Density grid [cm^-3]
;     - xwavi: Structure with wavelengths for each ion
;     - yptsi: Structure with emissivity tables for each ion
;
; USAGE:
;   IDL> .compile sim_citep_2
;   IDL> sim_citep_2
;
; CUSTOMIZATION:
;   To use different viewing geometry, modify in main procedure:
;     ray_origin = [x0, y0, z0]  ; Starting position [R_J]
;     ray_dir = [nx, ny, nz]      ; Pointing direction (normalized)
;
;   To modify torus parameters, adjust:
;     an[] - Ion mixing ratios at 6 R_J
;     bn[] - Radial power law indices
;     atel, btel - Electron temperature parameters
;
; COORDINATE SYSTEM:
;   Jovicentric coordinates with X-Y in equatorial plane, Z along spin axis
;   Distances in Jupiter radii (R_J = 71,492 km)
;
; OUTPUT:
;   UV spectrum in Rayleighs/Angstrom
;   2D emission maps
;   Diagnostic plots
;
; REFERENCES:
;   - Nerney et al. (2017, 2020) - Ion composition and temperatures
;   - Bagenal (1994) - Scale height formulation
;   - CHIANTI database (Dere et al. 1997, Del Zanna et al. 2021)
;
; AUTHOR:
;   CITEP Team - Community Io Torus Emission Package
;   For the Magnetospheres of Outer Planets (MOP) community
;
; HISTORY:
;   2024 - Initial release for community use
;-

;==============================================================================
; INTERPOLATION FUNCTIONS
;==============================================================================

function interpolate_emissivity_2D, temp_eV, dens_cm3, temp_arr, dens_arr, emiss_table
  ;+
  ; Bilinear interpolation of emissivity in log(T)-log(n) space
  ; This is appropriate since emissivities vary smoothly as power laws
  ;-

  ; Work in log space where emissivities are smoother
  log_temp = alog10(temp_eV)
  log_dens = alog10(dens_cm3)
  log_temp_arr = alog10(temp_arr)
  log_dens_arr = alog10(dens_arr)

  n_temp = n_elements(temp_arr)
  n_dens = n_elements(dens_arr)
  n_lines = (size(emiss_table))[3]

  ; Find interpolation indices
  it = value_locate(log_temp_arr, log_temp)
  id = value_locate(log_dens_arr, log_dens)

  ; Handle boundaries
  if it lt 0 then it = 0
  if it ge n_temp-1 then it = n_temp-2
  if id lt 0 then id = 0
  if id ge n_dens-1 then id = n_dens-2

  ; Calculate interpolation weights
  wt = (log_temp - log_temp_arr[it]) / (log_temp_arr[it+1] - log_temp_arr[it])
  wd = (log_dens - log_dens_arr[id]) / (log_dens_arr[id+1] - log_dens_arr[id])

  ; Bilinear interpolation for all wavelengths simultaneously
  emiss_interp = (1-wt) * (1-wd) * reform(emiss_table[it,   id,   *]) + $
    (1-wt) * wd     * reform(emiss_table[it,   id+1, *]) + $
    wt     * (1-wd) * reform(emiss_table[it+1, id,   *]) + $
    wt     * wd     * reform(emiss_table[it+1, id+1, *])

  return, reform(emiss_interp)
end

;==============================================================================
function interpolate_emissivity_batch, temp_arr_input, dens_arr_input, $
  temp_grid, dens_grid, emiss_table
  ;+
  ; Batch interpolation for multiple points along line of sight
  ; Processes all points together for efficiency
  ;-

  n_points = n_elements(temp_arr_input)
  n_lines = (size(emiss_table))[3]
  n_temp = n_elements(temp_grid)
  n_dens = n_elements(dens_grid)

  emiss_out = dblarr(n_points, n_lines)

  log_temp_in = alog10(temp_arr_input)
  log_dens_in = alog10(dens_arr_input)
  log_temp_grid = alog10(temp_grid)
  log_dens_grid = alog10(dens_grid)

  ; Find indices for all points
  it_arr = value_locate(log_temp_grid, log_temp_in)
  id_arr = value_locate(log_dens_grid, log_dens_in)

  ; Interpolate each point
  for i = 0L, n_points-1 do begin
    it = it_arr[i]
    id = id_arr[i]

    if it lt 0 then it = 0
    if it ge n_temp-1 then it = n_temp-2
    if id lt 0 then id = 0
    if id ge n_dens-1 then id = n_dens-2

    wt = (log_temp_in[i] - log_temp_grid[it]) / (log_temp_grid[it+1] - log_temp_grid[it])
    wd = (log_dens_in[i] - log_dens_grid[id]) / (log_dens_grid[id+1] - log_dens_grid[id])

    emiss_out[i,*] = (1-wt) * (1-wd) * reform(emiss_table[it,   id,   *]) + $
      (1-wt) * wd     * reform(emiss_table[it,   id+1, *]) + $
      wt     * (1-wd) * reform(emiss_table[it+1, id,   *]) + $
      wt     * wd     * reform(emiss_table[it+1, id+1, *])
  endfor

  return, emiss_out
end

;==============================================================================
; INSTRUMENT RESPONSE FUNCTION
;==============================================================================

function simulate_IPT_spectrum_Rayleighs_ERF_form, x, spec_binsize, xwavi, yptsi, fwhm
  ;+
  ; Convolve discrete emission lines with Gaussian instrument response
  ; Uses error function formulation for exact integration over bins
  ;-

  rootc = 2d*sqrt(alog(2d))/fwhm  ; Related to 1/(σ√2)
  nj = n_elements(x)
  ni = n_elements(yptsi)
  ypts = dblarr(nj)

  ; Convolve each emission line
  for i=0, ni-1 do begin
    if yptsi[i] le 0 then continue

    ; Exact integration of Gaussian over finite bins
    ypts += yptsi[i] * 0.5d * $
      (Erf((x - xwavi[i] + spec_binsize/2d)*rootc) - $
      Erf((x - xwavi[i] - spec_binsize/2d)*rootc))
  endfor

  ; Convert to per-Angstrom units
  ypts /= spec_binsize
  return, ypts
end

;==============================================================================
; RAY TRACING FUNCTIONS
;==============================================================================

function ray_box_intersection, ray_origin, ray_dir, box_min, box_max
  ;+
  ; Calculate intersection of ray with 3D bounding box
  ; Uses slab method: finds intersection with each pair of parallel planes
  ; Returns [t_min, t_max] parametric distances along ray, or [-1, -1] if no intersection
  ;-

  t_min = -1d30
  t_max = 1d30

  ; Check intersection with each dimension's planes
  for i = 0, 2 do begin
    if abs(ray_dir[i]) gt 1d-10 then begin
      ; Ray not parallel to planes
      t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
      t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]

      ; Ensure t1 <= t2
      if t1 gt t2 then begin
        temp = t1
        t1 = t2
        t2 = temp
      endif

      ; Update intersection interval
      t_min = t_min > t1
      t_max = t_max < t2

      if t_min gt t_max then return, [-1d, -1d]  ; No intersection

    endif else begin
      ; Ray parallel to planes - check if between them
      if (ray_origin[i] lt box_min[i]) or (ray_origin[i] gt box_max[i]) then $
        return, [-1d, -1d]
    endelse
  endfor

  ; Only consider forward direction
  if t_max lt 0 then return, [-1d, -1d]
  if t_min lt 0 then t_min = 0d

  return, [t_min, t_max]
end

;==============================================================================
function sample_ray_through_torus, ray_origin, ray_dir, ds, $
  torus_rmin, torus_rmax, torus_zmax
  ;+
  ; Generate sample points along ray passing through torus region
  ;
  ; INPUTS:
  ;   ray_origin - [x, y, z] starting position [R_J]
  ;   ray_dir - [nx, ny, nz] normalized direction vector
  ;   ds - Step size for sampling [R_J]
  ;   torus_rmin, rmax - Radial extent of torus [R_J]
  ;   torus_zmax - Vertical extent [R_J]
  ;
  ; OUTPUTS:
  ;   Structure with arrays of positions and validity flags
  ;-

  ; Default torus boundaries if not specified
  if n_elements(torus_rmin) eq 0 then torus_rmin = 4.5d
  if n_elements(torus_rmax) eq 0 then torus_rmax = 10d
  if n_elements(torus_zmax) eq 0 then torus_zmax = 3d

  ; Define bounding box
  box_min = [-torus_rmax, -torus_rmax, -torus_zmax]
  box_max = [torus_rmax, torus_rmax, torus_zmax]

  ; Find ray-box intersection
  t_range = ray_box_intersection(ray_origin, ray_dir, box_min, box_max)

  if t_range[0] eq -1 then begin
    ; No intersection with torus volume
    return, {s: [0d], x: [ray_origin[0]], y: [ray_origin[1]], $
      z: [ray_origin[2]], valid: [0B], n_valid: 0L}
  endif

  ; Generate sample points along ray
  n_samples = ceil((t_range[1] - t_range[0]) / ds) + 1
  s_array = t_range[0] + ds * dindgen(n_samples)

  ; Include exit point
  if s_array[n_samples-1] lt t_range[1] then begin
    s_array = [s_array, t_range[1]]
    n_samples++
  endif

  ; Calculate 3D positions along ray
  x_array = ray_origin[0] + s_array * ray_dir[0]
  y_array = ray_origin[1] + s_array * ray_dir[1]
  z_array = ray_origin[2] + s_array * ray_dir[2]

  ; Check which points are inside torus
  rho_array = sqrt(x_array^2 + y_array^2)  ; Cylindrical radius
  valid = (rho_array ge torus_rmin) and (rho_array le torus_rmax) and $
    (abs(z_array) le torus_zmax)
  n_valid = total(valid)

  return, {s: s_array, x: x_array, y: y_array, z: z_array, $
    valid: valid, n_valid: n_valid}
end

;==============================================================================
; PLASMA PHYSICS FUNCTIONS
;==============================================================================

function get_scale_heights_RJ, T_ions
  ;+
  ; Calculate vertical scale heights for each ion species
  ; Based on centrifugal confinement with ambipolar electric field
  ; Reference: Bagenal (1994)
  ;
  ; Scale height H = sqrt(2kT(1+qΦ/kT)/(3mΩ²r²))
  ; where Φ is ambipolar potential ≈ kT_e/e
  ;-

  RJ_km = 71492d  ; Jupiter radius in km
  omega = (2d * !dpi) / (9.925d * 3600d)  ; Jupiter rotation rate [rad/s]

  ; Conversion factors for sulfur and oxygen
  factor_s = sqrt((2d*1.60218d-19)/(3d*32d*1.66054d-27*omega*omega))/1d3/RJ_km
  factor_o = sqrt((2d*1.60218d-19)/(3d*16d*1.66054d-27*omega*omega))/1d3/RJ_km

  H = {sp: 0d, s2p: 0d, s3p: 0d, s4p: 0d, op: 0d, o2p: 0d}

  ; Include effect of ambipolar electric field
  H.sp  = factor_s * sqrt(T_ions.sp  * (1d + T_ions.el/T_ions.sp))     ; S+
  H.s2p = factor_s * sqrt(T_ions.s2p * (1d + 2d*T_ions.el/T_ions.s2p)) ; S++
  H.s3p = factor_s * sqrt(T_ions.s3p * (1d + 3d*T_ions.el/T_ions.s3p)) ; S+++
  H.s4p = factor_s * sqrt(T_ions.s4p * (1d + 4d*T_ions.el/T_ions.s4p)) ; S++++
  H.op  = factor_o * sqrt(T_ions.op  * (1d + T_ions.el/T_ions.op))     ; O+
  H.o2p = factor_o * sqrt(T_ions.o2p * (1d + 2d*T_ions.el/T_ions.o2p)) ; O++

  return, H
end

;==============================================================================
function calculate_torus_parameters_along_los, ray_samples, an, bn, atel, btel
  ;+
  ; Calculate plasma parameters at each point along line of sight
  ;
  ; INPUTS:
  ;   ray_samples - Structure from sample_ray_through_torus
  ;   an[] - Ion mixing ratios at 6 R_J
  ;   bn[] - Radial power law indices for ions
  ;   atel - Electron temperature at 6 R_J [eV]
  ;   btel - Electron temperature radial index
  ;
  ; PHYSICS:
  ;   - Ion temperatures from polynomial fits (Nerney et al.)
  ;   - Electron temperature power law
  ;   - Ion densities with radial power laws and Gaussian vertical profiles
  ;   - Electron density from quasi-neutrality
  ;-

  n_points = n_elements(ray_samples.s)

  ; Initialize parameter arrays
  nel = dblarr(n_points)
  nsp = dblarr(n_points)
  ns2p = dblarr(n_points)
  ns3p = dblarr(n_points)
  ns4p = dblarr(n_points)
  nop = dblarr(n_points)
  no2p = dblarr(n_points)
  Tel = dblarr(n_points)

  ; Get valid points in active torus region
  valid_idx = where(ray_samples.valid, n_valid)
  if n_valid eq 0 then begin
    return, {nel: nel, nsp: nsp, ns2p: ns2p, ns3p: ns3p, ns4p: ns4p, $
      nop: nop, no2p: no2p, Tel: Tel}
  endif

  ; Extract positions
  x_valid = ray_samples.x[valid_idx]
  y_valid = ray_samples.y[valid_idx]
  z_valid = ray_samples.z[valid_idx]
  rho_valid = sqrt(x_valid^2 + y_valid^2)

  ; Select points in active region (5.5-10 R_J)
  active_idx = where((rho_valid ge 5.5d) and (rho_valid le 10d), n_active)
  if n_active eq 0 then begin
    return, {nel: nel, nsp: nsp, ns2p: ns2p, ns3p: ns3p, ns4p: ns4p, $
      nop: nop, no2p: no2p, Tel: Tel}
  endif

  final_idx = valid_idx[active_idx]
  rho = rho_valid[active_idx]
  z = z_valid[active_idx]

  ; Ion temperatures from observational fits (Nerney et al. 2017)
  ; Fourth-order polynomial fits valid for 6-10 R_J
  Tsp = -2992.71d + 1401.02d*rho - 269.568d*rho^2 + 27.3165d*rho^3 - 1.12374d*rho^4
  Ts2p = -4650.83d + 2122.14d*rho - 351.473d*rho^2 + 25.9979d*rho^3 - 0.732711d*rho^4
  Ts3p = -2826d + 1232.17d*rho - 188.612d*rho^2 + 12.4472d*rho^3 - 0.299455d*rho^4
  Ts4p = Ts3p * 1.5d  ; S++++ approximately 1.5× S+++ temperature
  Top = -4042.21d + 1885.72d*rho - 321.068d*rho^2 + 24.6705d*rho^3 - 0.728725d*rho^4
  To2p = -1766.47d + 695.67d*rho - 88.4991d*rho^2 + 4.25863d*rho^3 - 0.0512119d*rho^4

  ; Ensure physical temperatures
  Tsp = Tsp > 1d
  Ts2p = Ts2p > 1d
  Ts3p = Ts3p > 1d
  Ts4p = Ts4p > 1d
  Top = Top > 1d
  To2p = To2p > 1d

  ; Electron temperature power law
  Tel_active = atel * (rho/6d)^btel

  ; Calculate scale heights for each point
  H_sp = dblarr(n_active)
  H_s2p = dblarr(n_active)
  H_s3p = dblarr(n_active)
  H_s4p = dblarr(n_active)
  H_op = dblarr(n_active)
  H_o2p = dblarr(n_active)

  for i = 0L, n_active-1 do begin
    T_ions = {sp: Tsp[i], s2p: Ts2p[i], s3p: Ts3p[i], s4p: Ts4p[i], $
      op: Top[i], o2p: To2p[i], el: Tel_active[i]}
    H = get_scale_heights_RJ(T_ions)
    H_sp[i] = H.sp
    H_s2p[i] = H.s2p
    H_s3p[i] = H.s3p
    H_s4p[i] = H.s4p
    H_op[i] = H.op
    H_o2p[i] = H.o2p
  endfor

  ; Electron density radial profile
  nel_active = dblarr(n_active)
  idx_low = where(rho le 7.8d, n_low)
  idx_high = where(rho gt 7.8d, n_high)

  if n_low gt 0 then nel_active[idx_low] = 2200d * (rho[idx_low]/6d)^(-5.4d)
  if n_high gt 0 then nel_active[idx_high] = 2200d * (7.8d/6d)^(-5.4d) * (rho[idx_high]/7.8d)^(-12d)

  ; Ion densities with vertical stratification
  nsp_active = nel_active * an[0] * (rho/6d)^bn[0] * exp(-(z/H_sp)^2)
  ns2p_active = nel_active * an[1] * (rho/6d)^bn[1] * exp(-(z/H_s2p)^2)
  ns3p_active = nel_active * an[2] * (rho/6d)^bn[2] * exp(-(z/H_s3p)^2)
  ns4p_active = nel_active * an[2] * 0.1d * (rho/6d)^bn[2] * exp(-(z/H_s4p)^2)
  nop_active = nel_active * an[3] * (rho/6d)^bn[3] * exp(-(z/H_op)^2)
  no2p_active = nel_active * an[4] * (rho/6d)^bn[4] * exp(-(z/H_o2p)^2)

  ; Enforce quasi-neutrality
  nel_active = (nsp_active + 2d*ns2p_active + 3d*ns3p_active + 4d*ns4p_active + $
    nop_active + 2d*no2p_active) / 0.9d

  ; Store in output arrays
  nel[final_idx] = nel_active
  nsp[final_idx] = nsp_active
  ns2p[final_idx] = ns2p_active
  ns3p[final_idx] = ns3p_active
  ns4p[final_idx] = ns4p_active
  nop[final_idx] = nop_active
  no2p[final_idx] = no2p_active
  Tel[final_idx] = Tel_active

  return, {nel: nel, nsp: nsp, ns2p: ns2p, ns3p: ns3p, ns4p: ns4p, $
    nop: nop, no2p: no2p, Tel: Tel}
end

;==============================================================================
function integrate_emission_along_los, ray_samples, plasma_params, $
  xwav, spec_binsize, fwhm, $
  temp_arr, dens_arr, xwavi_struct, yptsi_struct, $
  wavelength_skip=wavelength_skip
  ;+
  ; Integrate UV emission along line of sight
  ;
  ; PHYSICS:
  ;   B_λ = 10^(-6) × ∫ ε_λ(T_e, n_e) × n_ion × ds
  ;
  ; The factor 10^(-6) converts from photons/s/cm² to Rayleighs
  ; (1 Rayleigh = 10^6 photons/s/cm²/4π sr)
  ;
  ; KEYWORDS:
  ;   wavelength_skip - Process every Nth wavelength for faster preview (default=1)
  ;-

  if n_elements(wavelength_skip) eq 0 then wavelength_skip = 1

  ; Select points with significant plasma
  valid_idx = where(ray_samples.valid and (plasma_params.nel gt 1d), n_valid)
  if n_valid eq 0 then return, dblarr(n_elements(xwav))

  ; Extract plasma parameters at valid points
  Tel_valid = plasma_params.Tel[valid_idx]
  nel_valid = plasma_params.nel[valid_idx]
  nsp_valid = plasma_params.nsp[valid_idx]
  ns2p_valid = plasma_params.ns2p[valid_idx]
  ns3p_valid = plasma_params.ns3p[valid_idx]
  ns4p_valid = plasma_params.ns4p[valid_idx]
  nop_valid = plasma_params.nop[valid_idx]
  no2p_valid = plasma_params.no2p[valid_idx]

  ; Calculate path length elements (trapezoidal rule)
  RJ_cm = 7.1492d9  ; Jupiter radius in cm
  ds_array = dblarr(n_valid)

  for i = 0L, n_valid-1 do begin
    if i eq 0 then begin
      if n_valid gt 1 then begin
        ds_array[i] = ray_samples.s[valid_idx[1]] - ray_samples.s[valid_idx[0]]
      endif else begin
        ds_array[i] = 0.1d
      endelse
    endif else if i eq n_valid-1 then begin
      ds_array[i] = ray_samples.s[valid_idx[i]] - ray_samples.s[valid_idx[i-1]]
    endif else begin
      ds_array[i] = (ray_samples.s[valid_idx[i+1]] - ray_samples.s[valid_idx[i-1]]) / 2d
    endelse
  endfor
  ds_cm = ds_array * RJ_cm

  ; Batch interpolate emissivities for all points
  emiss_sp_batch = interpolate_emissivity_batch(Tel_valid, nel_valid, $
    temp_arr, dens_arr, yptsi_struct.sp)
  emiss_s2p_batch = interpolate_emissivity_batch(Tel_valid, nel_valid, $
    temp_arr, dens_arr, yptsi_struct.s2p)
  emiss_s3p_batch = interpolate_emissivity_batch(Tel_valid, nel_valid, $
    temp_arr, dens_arr, yptsi_struct.s3p)
  emiss_s4p_batch = interpolate_emissivity_batch(Tel_valid, nel_valid, $
    temp_arr, dens_arr, yptsi_struct.s4p)
  emiss_op_batch = interpolate_emissivity_batch(Tel_valid, nel_valid, $
    temp_arr, dens_arr, yptsi_struct.op)
  emiss_o2p_batch = interpolate_emissivity_batch(Tel_valid, nel_valid, $
    temp_arr, dens_arr, yptsi_struct.o2p)

  ; Get number of lines for each ion
  n_lines_sp = n_elements(xwavi_struct.sp)
  n_lines_s2p = n_elements(xwavi_struct.s2p)
  n_lines_s3p = n_elements(xwavi_struct.s3p)
  n_lines_s4p = n_elements(xwavi_struct.s4p)
  n_lines_op = n_elements(xwavi_struct.op)
  n_lines_o2p = n_elements(xwavi_struct.o2p)

  ; Integrate emission × density × path length
  yptsi_sp = dblarr(n_lines_sp)
  yptsi_s2p = dblarr(n_lines_s2p)
  yptsi_s3p = dblarr(n_lines_s3p)
  yptsi_s4p = dblarr(n_lines_s4p)
  yptsi_op = dblarr(n_lines_op)
  yptsi_o2p = dblarr(n_lines_o2p)

  for j = 0L, n_lines_sp-1 do $
    yptsi_sp[j] = 1d-6 * total(emiss_sp_batch[*,j] * nsp_valid * ds_cm)
  for j = 0L, n_lines_s2p-1 do $
    yptsi_s2p[j] = 1d-6 * total(emiss_s2p_batch[*,j] * ns2p_valid * ds_cm)
  for j = 0L, n_lines_s3p-1 do $
    yptsi_s3p[j] = 1d-6 * total(emiss_s3p_batch[*,j] * ns3p_valid * ds_cm)
  for j = 0L, n_lines_s4p-1 do $
    yptsi_s4p[j] = 1d-6 * total(emiss_s4p_batch[*,j] * ns4p_valid * ds_cm)
  for j = 0L, n_lines_op-1 do $
    yptsi_op[j] = 1d-6 * total(emiss_op_batch[*,j] * nop_valid * ds_cm)
  for j = 0L, n_lines_o2p-1 do $
    yptsi_o2p[j] = 1d-6 * total(emiss_o2p_batch[*,j] * no2p_valid * ds_cm)

  ; Combine all ions
  xwavi_all = [xwavi_struct.sp, xwavi_struct.s2p, xwavi_struct.s3p, $
    xwavi_struct.s4p, xwavi_struct.op, xwavi_struct.o2p]
  yptsi_all = [yptsi_sp, yptsi_s2p, yptsi_s3p, yptsi_s4p, yptsi_op, yptsi_o2p]

  ; Apply wavelength skip if requested (for faster previews)
  if wavelength_skip gt 1 then begin
    idx = lindgen(n_elements(xwavi_all)/wavelength_skip) * wavelength_skip
    xwavi_all = xwavi_all[idx]
    yptsi_all = yptsi_all[idx]
  endif

  ; Sort by wavelength
  wsort = sort(xwavi_all)
  xwavi_sorted = xwavi_all[wsort]
  yptsi_sorted = yptsi_all[wsort]

  ; Convolve with instrument response
  spectrum = simulate_IPT_spectrum_Rayleighs_ERF_form(xwav, spec_binsize, $
    xwavi_sorted, yptsi_sorted, fwhm)

  return, spectrum
end

;==============================================================================
; MAIN PROCEDURE
;==============================================================================

pro sim_citep_2
  ;+
  ; Main procedure to calculate IPT UV emission spectra
  ;
  ; WORKFLOW:
  ;   1. Load pre-calculated CHIANTI emission tables
  ;   2. Set up viewing geometry and torus parameters
  ;   3. Trace rays through torus
  ;   4. Calculate plasma parameters along each ray
  ;   5. Integrate emission along line of sight
  ;   6. Generate spectra and emission maps
  ;-

  ; ============================================================================
  ; WAVELENGTH GRID SETUP
  ; ============================================================================
  ; Define spectral range and resolution
  xmin = 550d     ; Minimum wavelength [Å]
  xmax = 2100d    ; Maximum wavelength [Å]
  binsize = 1d    ; Spectral bin width [Å]

  nx = round((xmax - xmin)/binsize) + 1
  xwav = dindgen(nx)*binsize + xmin

  ; Instrument spectral resolution
  fwhm = 4.47d    ; Full width half maximum [Å] - typical for UV spectrographs

  ; ============================================================================
  ; TORUS MODEL PARAMETERS
  ; ============================================================================
  ; Based on Cassini UVIS observations (Nerney et al. 2017, 2020)

  ; Electron temperature model: T_e = atel × (r/6)^btel
  atel = 5d       ; Temperature at 6 R_J [eV]
  btel = 2.484d   ; Radial power law index

  ; Ion mixing ratios at 6 R_J and radial dependencies
  ; Mixing ratio = n_ion/n_e at 6 R_J, varies as (r/6)^bn

  ansp = 0.0726d   ; S+ mixing ratio
  bnsp = -3.38d    ; S+ radial index

  ans2p = 0.231d   ; S++ mixing ratio
  bns2p = -1.13d   ; S++ radial index

  ans3p = 0.0302d  ; S+++ mixing ratio
  bns3p = 1.748d   ; S+++ radial index

  anop = 0.231d    ; O+ mixing ratio
  bnop = -0.963d   ; O+ radial index

  ano2p = 0.0455d  ; O++ mixing ratio
  bno2p = 2.346d   ; O++ radial index

  ; Combine parameters into arrays
  an = [ansp, ans2p, ans3p, anop, ano2p]
  bn = [bnsp, bns2p, bns3p, bnop, bno2p]

  ; ============================================================================
  ; LOAD EMISSION TABLES
  ; ============================================================================
  print, '================================================================'
  print, 'CITEP - Community Io Torus Emission Package'
  print, '================================================================'
  print, 'Loading pre-calculated CHIANTI emission tables...'

  restore, 'CHIANTI_11.0.2_emiss_arrays_all_species_all_wavelengths_50x50_logspaced.sav'
  ; This file contains:
  ;   temp_arr - Temperature grid [eV]
  ;   dens_arr - Density grid [cm^-3]
  ;   xwavi - Structure with wavelengths for each ion
  ;   yptsi - Structure with emissivity tables

  print, 'Tables loaded successfully:'
  print, '  Temperature range: ', min(temp_arr), ' - ', max(temp_arr), ' eV'
  print, '  Density range: ', min(dens_arr), ' - ', max(dens_arr), ' cm^-3'

  ; ============================================================================
  ; EXAMPLE VIEWING GEOMETRIES
  ; ============================================================================
  print, ''
  print, '================================================================'
  print, 'CALCULATING UV EMISSION SPECTRA'
  print, '================================================================'

  ; Integration parameters
  ds = 0.1d  ; Step size along ray [R_J] - smaller values give higher accuracy

  ; ----------------------------------------------------------------------------
  ; Example 1: Equatorial viewing
  ; ----------------------------------------------------------------------------
  print, ''
  print, 'Example 1: Equatorial view (ansa observation)'
  ray_origin = [6d, -20d, 0d]    ; Observer position [R_J]
  ray_dir = [0d, 1d, 0d]          ; Looking in +Y direction

  ; Trace ray through torus
  ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds, 4.5d, 10d, 3d)
  print, '  Ray intersects torus at ', ray_samples.n_valid, ' points'

  ; Calculate plasma parameters along ray
  plasma_params = calculate_torus_parameters_along_los(ray_samples, an, bn, atel, btel)

  ; Integrate emission
  spectrum_eq = integrate_emission_along_los(ray_samples, plasma_params, $
    xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)

  ; ----------------------------------------------------------------------------
  ; Example 2: Oblique viewing
  ; ----------------------------------------------------------------------------
  print, ''
  print, 'Example 2: Oblique view (45° from vertical)'
  ray_origin = [6d, -20d, 10d]
  ray_dir = [0d, 1d/sqrt(2d), -1d/sqrt(2d)]

  ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds, 4.5d, 10d, 3d)
  print, '  Ray intersects torus at ', ray_samples.n_valid, ' points'

  plasma_params = calculate_torus_parameters_along_los(ray_samples, an, bn, atel, btel)
  spectrum_oblique = integrate_emission_along_los(ray_samples, plasma_params, $
    xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)

  ; ----------------------------------------------------------------------------
  ; Example 3: Polar viewing
  ; ----------------------------------------------------------------------------
  print, ''
  print, 'Example 3: Polar view (looking down spin axis)'
  ray_origin = [6d, 0d, 10d]
  ray_dir = [0d, 0d, -1d]

  ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds, 4.5d, 10d, 3d)
  print, '  Ray intersects torus at ', ray_samples.n_valid, ' points'

  plasma_params = calculate_torus_parameters_along_los(ray_samples, an, bn, atel, btel)
  spectrum_polar = integrate_emission_along_los(ray_samples, plasma_params, $
    xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)

  ; ============================================================================
  ; GENERATE PLOTS
  ; ============================================================================

  ; Individual spectra
  p1 = plot(xwav, spectrum_eq, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (R/Å)', $
    title='IPT UV Emission: Equatorial View', $
    thick=2, color='black', $
    xrange=[550, 2100])

  ; Comparison plot
  p2 = plot(xwav, spectrum_eq, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (R/Å)', $
    title='IPT UV Emission: Viewing Geometry Effects', $
    thick=2, color='black', $
    name='Equatorial', $
    xrange=[550, 2100])

  p3 = plot(xwav, spectrum_oblique, $
    /overplot, $
    thick=2, color='blue', $
    name='45° Oblique', $
    transparency=30)

  p4 = plot(xwav, spectrum_polar, $
    /overplot, $
    thick=2, color='red', $
    name='Polar', $
    transparency=30)

  leg = legend(target=[p2, p3, p4], $
    position=[0.7, 0.8], $
    /normal)

  ; ============================================================================
  ; CREATE 2D EMISSION MAP
  ; ============================================================================
  print, ''
  print, '================================================================'
  print, 'GENERATING 2D EMISSION MAP'
  print, '================================================================'

  ; Grid of viewing positions
  n_map = 41
  x_positions = -10d + 20d * dindgen(n_map)/(n_map-1)
  z_positions = -10d + 20d * dindgen(n_map)/(n_map-1)

  total_emission = dblarr(n_map, n_map)

  ; Fixed viewing direction
  ray_dir = [0d, 1d, 0d]

  print, 'Calculating ', n_map, 'x', n_map, ' pixel map...'
  print, 'Using wavelength skip=5 for faster calculation'

  for i = 0, n_map-1 do begin
    for j = 0, n_map-1 do begin
      ray_origin = [x_positions[i], -20d, z_positions[j]]

      ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds, 4.5d, 10d, 3d)

      if ray_samples.n_valid gt 0 then begin
        plasma_params = calculate_torus_parameters_along_los(ray_samples, $
          an, bn, atel, btel)

        ; Use wavelength skip for faster map generation
        spectrum = integrate_emission_along_los(ray_samples, plasma_params, $
          xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi, wavelength_skip=5)

        total_emission[i,j] = total(spectrum)
      endif
    endfor
    if (i mod 10) eq 0 then print, '  Row ', i+1, ' of ', n_map, ' complete'
  endfor

  ; Plot emission map
  c = contour(total_emission, x_positions, z_positions, $
    /fill, $
    n_levels=20, $
    xtitle='X Position (R_J)', $
    ytitle='Z Position (R_J)', $
    title='Total UV Emission (550-2100 Å)', $
    rgb_table=33)

  cb = colorbar(target=c, $
    title='Integrated Intensity (R)', $
    orientation=1)

  ; ============================================================================
  ; SUMMARY
  ; ============================================================================
  print, ''
  print, '================================================================'
  print, 'CALCULATION COMPLETE'
  print, '================================================================'
  print, 'Total integrated emission (550-2100 Å):'
  print, '  Equatorial view: ', total(spectrum_eq), ' Rayleighs'
  print, '  45° Oblique view: ', total(spectrum_oblique), ' Rayleighs'
  print, '  Polar view: ', total(spectrum_polar), ' Rayleighs'
  print, ''
  print, 'Key assumptions:'
  print, '  - Optically thin emission'
  print, '  - Steady-state torus (no temporal variations)'
  print, '  - Axisymmetric density distribution'
  print, '  - Single Maxwellian electron distribution'
  print, '  - CHIANTI atomic data (coronal approximation)'
  print, ''
  print, 'Output variables available:'
  print, '  xwav - Wavelength grid [Å]'
  print, '  spectrum_eq/oblique/polar - Calculated spectra [R/Å]'
  print, '  total_emission - 2D emission map [R]'

  stop
end