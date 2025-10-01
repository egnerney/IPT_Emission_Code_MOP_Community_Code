;+
; NAME:
;   sim_citep_2_diff_eq.pro
;
; PURPOSE:
;   Calculate UV emission spectra from the Io Plasma Torus using a 3D
;   diffusive equilibrium model aligned with Jupiter's magnetic field lines.
;   Integrates emission along arbitrary lines of sight through the torus.
;
; DESCRIPTION:
;   This code simulates UV emission (550-2100 Å) using a physically realistic
;   3D torus model based on diffusive equilibrium along JRM33+CON2020 magnetic
;   field lines. The emission calculation properly integrates through the
;   non-axisymmetric torus structure:
;
;     B_λ = 10^(-6) × ∫ ε_λ(T_e(s), n_e(s)) × n_ion(s) ds
;
;   where the plasma parameters vary along field lines according to
;   diffusive equilibrium with centrifugal and pressure gradient forces.
;
; MODEL STRUCTURE:
;   The torus model is defined on an irregular grid following field lines:
;   - 501 field lines crossing equator at ρ = 5.00 to 10.00 R_J (0.01 R_J steps)
;   - 360 azimuthal positions φ = 0° to 359° (1° steps)
;   - 1001 points along each field line from λ_III = -50° to +50° (0.1° steps)
;   - System III coordinates (not centrifugal equator)
;   - Field-aligned diffusive equilibrium determines density stratification
;
; REQUIRED INPUT FILES:
;   Either the individual model files:
;     - nelec_out_mymodel1_*.txt  - Electron density [cm^-3]
;     - nsp_out_mymodel1_*.txt    - S+ density [cm^-3]
;     - ns2p_out_mymodel1_*.txt   - S++ density [cm^-3]
;     - ns3p_out_mymodel1_*.txt   - S+++ density [cm^-3]
;     - nop_out_mymodel1_*.txt    - O+ density [cm^-3]
;     - no2p_out_mymodel1_*.txt   - O++ density [cm^-3]
;     - Telec_out_mymodel1_*.txt  - Electron temperature [eV]
;     - Tic_out_mymodel1_*.txt    - Ion core temperature [eV]
;     - x_out*.txt, y_out*.txt, z_out*.txt - Positions [R_J]
;
;   Or the pre-processed IDL save file:
;     - variables_rebin3D.sav containing all arrays in 501x360x1001 format
;
;   Plus CHIANTI emission tables:
;     - CHIANTI_11.0.2_emiss_arrays_all_species_all_wavelengths_50x50_logspaced.sav
;
; COORDINATE SYSTEM:
;   System III Jovicentric coordinates:
;   - X, Y in Jovian equatorial plane
;   - Z along rotation axis
;   - ρ = sqrt(X² + Y²) cylindrical radius
;   - φ = atan(Y/X) east longitude
;   - λ_III = asin(Z/r) System III latitude
;
; KEY ASSUMPTIONS:
;   - Diffusive equilibrium along magnetic field lines
;   - JRM33+CON2020 magnetic field model
;   - Optically thin emission
;   - Single Maxwellian electron distribution
;   - Steady-state torus (no temporal variations)
;   - CHIANTI coronal approximation for atomic physics
;
; USAGE:
;   IDL> .compile sim_citep_2_diff_eq
;   IDL> sim_citep_2_diff_eq
;
; OUTPUT:
;   - UV spectra for different viewing geometries [Rayleighs/Å]
;   - 2D emission maps
;   - Diagnostic plots showing torus structure
;
; AUTHOR:
;   CITEP Team - Community Io Torus Emission Package
;   For the Magnetospheres of Outer Planets (MOP) community
;
; HISTORY:
;   2024 - Initial release with field-aligned diffusive equilibrium model
;-

;==============================================================================
; UTILITY FUNCTIONS
;==============================================================================

function load_3d_torus_model
  ;+
  ; Load the 3D field-aligned torus model
  ; Returns structure with all plasma parameters on 501x360x1001 grid
  ;-

  ; Check if pre-processed save file exists
  if file_test('variables_rebin3D.sav') then begin
    print, 'Loading pre-processed 3D model from variables_rebin3D.sav...'
    restore, 'variables_rebin3D.sav'

    model = {x: x_rebin, y: y_rebin, z: z_rebin, $
      r: r_rebin, rho: rho_rebin, phi: phi_rebin, lat: lat_rebin, $
      nel: ne_rebin, nsp: nsp_rebin, ns2p: ns2p_rebin, ns3p: ns3p_rebin, $
      nop: nop_rebin, no2p: no2p_rebin, noph: noph_rebin, $
      nhp: nhp_rebin, nnap: nnap_rebin, $
      Te: Tec_rebin, Ti: Tic_rebin, Thp: Thp_rebin, Toph: Toph_rebin}

    print, 'Model loaded: 501 x 360 x 1001 grid'
    return, model

  endif else begin
    ; Load from individual text files
    print, 'Loading 3D model from text files...'
    print, 'This may take several minutes...'

    ; Load position arrays
    openr,1,'x_outlat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    x=fltarr(180360,1001)
    readf,1,x
    close,1

    openr,1,'y_outlat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    y=fltarr(180360,1001)
    readf,1,y
    close,1

    openr,1,'z_outlat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    z=fltarr(180360,1001)
    readf,1,z
    close,1

    ; Load density arrays
    openr,1,'nelec_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    nel=fltarr(180360,1001)
    readf,1,nel
    close,1

    openr,1,'nsp_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    nsp=fltarr(180360,1001)
    readf,1,nsp
    close,1

    openr,1,'ns2p_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    ns2p=fltarr(180360,1001)
    readf,1,ns2p
    close,1

    openr,1,'ns3p_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    ns3p=fltarr(180360,1001)
    readf,1,ns3p
    close,1

    openr,1,'nop_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    nop=fltarr(180360,1001)
    readf,1,nop
    close,1

    openr,1,'no2p_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    no2p=fltarr(180360,1001)
    readf,1,no2p
    close,1

    ; Load temperature arrays
    openr,1,'Telec_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    tec=fltarr(180360,1001)
    readf,1,tec
    close,1

    openr,1,'Tic_out_mymodel1_diffeq_jrm33+Con2020_0fillslat-50to50_0.1deginterps_501x360_3D_analytic_180360x1001.txt'
    tico=fltarr(180360,1001)
    readf,1,tico
    close,1

    ; Calculate derived quantities
    rho = sqrt(x^2 + y^2)
    r = sqrt(x^2 + y^2 + z^2)
    lat = (180./!pi)*asin(z/r)
    phi = (180./!pi)*atan(y,x)

    ; Reshape from 2D (180360x1001) to 3D (501x360x1001)
    print, 'Reshaping arrays to 3D format...'

    nrfls = 501
    nphifls = 360
    nlat = 1001

    x_rebin = fltarr(nrfls,nphifls,nlat)
    y_rebin = fltarr(nrfls,nphifls,nlat)
    z_rebin = fltarr(nrfls,nphifls,nlat)
    r_rebin = fltarr(nrfls,nphifls,nlat)
    rho_rebin = fltarr(nrfls,nphifls,nlat)
    phi_rebin = fltarr(nrfls,nphifls,nlat)
    lat_rebin = fltarr(nrfls,nphifls,nlat)
    ne_rebin = fltarr(nrfls,nphifls,nlat)
    nsp_rebin = fltarr(nrfls,nphifls,nlat)
    ns2p_rebin = fltarr(nrfls,nphifls,nlat)
    ns3p_rebin = fltarr(nrfls,nphifls,nlat)
    nop_rebin = fltarr(nrfls,nphifls,nlat)
    no2p_rebin = fltarr(nrfls,nphifls,nlat)
    Tec_rebin = fltarr(nrfls,nphifls,nlat)
    Tic_rebin = fltarr(nrfls,nphifls,nlat)

    ; Mapping: 2D index = 360*i + j maps to 3D[i,j,*]
    for i=0,500 do begin
      for j=0,359 do begin
        idx_2d = 360L*i + j
        x_rebin[i,j,*] = x[idx_2d,*]
        y_rebin[i,j,*] = y[idx_2d,*]
        z_rebin[i,j,*] = z[idx_2d,*]
        r_rebin[i,j,*] = r[idx_2d,*]
        rho_rebin[i,j,*] = rho[idx_2d,*]
        phi_rebin[i,j,*] = phi[idx_2d,*]
        lat_rebin[i,j,*] = lat[idx_2d,*]
        ne_rebin[i,j,*] = nel[idx_2d,*]
        nsp_rebin[i,j,*] = nsp[idx_2d,*]
        ns2p_rebin[i,j,*] = ns2p[idx_2d,*]
        ns3p_rebin[i,j,*] = ns3p[idx_2d,*]
        nop_rebin[i,j,*] = nop[idx_2d,*]
        no2p_rebin[i,j,*] = no2p[idx_2d,*]
        Tec_rebin[i,j,*] = tec[idx_2d,*]
        Tic_rebin[i,j,*] = tico[idx_2d,*]
      endfor
    endfor

    ; Assume S4+ is 10% of S3+ if not provided
    ns4p_rebin = 0.1 * ns3p_rebin

    model = {x: x_rebin, y: y_rebin, z: z_rebin, $
      r: r_rebin, rho: rho_rebin, phi: phi_rebin, lat: lat_rebin, $
      nel: ne_rebin, nsp: nsp_rebin, ns2p: ns2p_rebin, ns3p: ns3p_rebin, $
      ns4p: ns4p_rebin, nop: nop_rebin, no2p: no2p_rebin, $
      Te: Tec_rebin, Ti: Tic_rebin}

    print, 'Model loaded and reshaped: 501 x 360 x 1001 grid'
    return, model
  endelse
end

;==============================================================================
function interpolate_field_model, x_point, y_point, z_point, model
  ;+
  ; Interpolate plasma parameters from field-aligned model to arbitrary point
  ;
  ; INPUTS:
  ;   x_point, y_point, z_point - Position to interpolate to [R_J]
  ;   model - Structure with 3D field-aligned model
  ;
  ; OUTPUTS:
  ;   Structure with interpolated plasma parameters
  ;
  ; METHOD:
  ;   1. Find nearest field line in (ρ, φ) space
  ;   2. Interpolate along that field line in latitude
  ;   3. Use bilinear interpolation between adjacent field lines
  ;-

  ; Calculate cylindrical coordinates of query point
  rho_point = sqrt(x_point^2 + y_point^2)
  phi_point = atan(y_point, x_point) * 180./!pi
  if phi_point lt 0 then phi_point += 360.
  r_point = sqrt(x_point^2 + y_point^2 + z_point^2)
  lat_point = asin(z_point/r_point) * 180./!pi

  ; Check if point is in model domain
  if rho_point lt 5.0 or rho_point gt 10.0 then begin
    ; Outside radial range
    return, {nel: 0d, nsp: 0d, ns2p: 0d, ns3p: 0d, ns4p: 0d, $
      nop: 0d, no2p: 0d, Te: 0d, valid: 0B}
  endif

  if abs(lat_point) gt 50. then begin
    ; Outside latitude range
    return, {nel: 0d, nsp: 0d, ns2p: 0d, ns3p: 0d, ns4p: 0d, $
      nop: 0d, no2p: 0d, Te: 0d, valid: 0B}
  endif

  ; Find indices for interpolation
  ; Radial index (0-500 for 5.00-10.00 RJ)
  rho_idx = (rho_point - 5.0) / 0.01
  i_rho = floor(rho_idx)
  if i_rho ge 500 then i_rho = 499
  w_rho = rho_idx - i_rho

  ; Azimuthal index (0-359 for 0-359 degrees)
  phi_idx = phi_point
  i_phi = floor(phi_idx)
  if i_phi ge 360 then i_phi = 359
  w_phi = phi_idx - i_phi

  ; Handle azimuthal wrap-around
  i_phi_next = i_phi + 1
  if i_phi_next ge 360 then i_phi_next = 0

  ; Latitude index (0-1000 for -50 to +50 degrees)
  lat_idx = (lat_point + 50.) / 0.1
  i_lat = floor(lat_idx)
  if i_lat ge 1000 then i_lat = 999
  if i_lat lt 0 then i_lat = 0
  w_lat = lat_idx - i_lat

  ; Bilinear interpolation in (rho, phi) and linear in latitude
  ; Get values at 8 corner points of interpolation cube
  ne_interp = 0d
  nsp_interp = 0d
  ns2p_interp = 0d
  ns3p_interp = 0d
  ns4p_interp = 0d
  nop_interp = 0d
  no2p_interp = 0d
  Te_interp = 0d

  ; Loop over the 8 corners
  for di_rho = 0, 1 do begin
    for di_phi = 0, 1 do begin
      for di_lat = 0, 1 do begin
        ; Calculate indices
        i_r = i_rho + di_rho < 500
        i_p = (di_phi eq 0) ? i_phi : i_phi_next
        i_l = i_lat + di_lat < 1000

        ; Calculate weight
        w_r = (di_rho eq 0) ? (1-w_rho) : w_rho
        w_p = (di_phi eq 0) ? (1-w_phi) : w_phi
        w_l = (di_lat eq 0) ? (1-w_lat) : w_lat
        weight = w_r * w_p * w_l

        ; Extract arrays from structure and get values
        nel_array = model.nel
        nsp_array = model.nsp
        ns2p_array = model.ns2p
        ns3p_array = model.ns3p
        nop_array = model.nop
        no2p_array = model.no2p
        Te_array = model.Te

        ; Accumulate weighted values
        nel_val = nel_array[i_r, i_p, i_l]
        nsp_val = nsp_array[i_r, i_p, i_l]
        ns2p_val = ns2p_array[i_r, i_p, i_l]
        ns3p_val = ns3p_array[i_r, i_p, i_l]
        nop_val = nop_array[i_r, i_p, i_l]
        no2p_val = no2p_array[i_r, i_p, i_l]
        Te_val = Te_array[i_r, i_p, i_l]

        ne_interp += weight * nel_val
        nsp_interp += weight * nsp_val
        ns2p_interp += weight * ns2p_val
        ns3p_interp += weight * ns3p_val
        ns4p_interp += weight * (ns3p_val * 0.1)  ; S4+ = 10% of S3+
        nop_interp += weight * nop_val
        no2p_interp += weight * no2p_val
        Te_interp += weight * Te_val
      endfor
    endfor
  endfor

  return, {nel: ne_interp, nsp: nsp_interp, ns2p: ns2p_interp, $
    ns3p: ns3p_interp, ns4p: ns4p_interp, $
    nop: nop_interp, no2p: no2p_interp, Te: Te_interp, valid: 1B}
end

;==============================================================================
; INTERPOLATION AND CONVOLUTION FUNCTIONS (unchanged from base version)
;==============================================================================

function interpolate_emissivity_2D, temp_eV, dens_cm3, temp_arr, dens_arr, emiss_table
  log_temp = alog10(temp_eV)
  log_dens = alog10(dens_cm3)
  log_temp_arr = alog10(temp_arr)
  log_dens_arr = alog10(dens_arr)

  n_temp = n_elements(temp_arr)
  n_dens = n_elements(dens_arr)
  n_lines = (size(emiss_table))[3]

  it = value_locate(log_temp_arr, log_temp)
  id = value_locate(log_dens_arr, log_dens)

  if it lt 0 then it = 0
  if it ge n_temp-1 then it = n_temp-2
  if id lt 0 then id = 0
  if id ge n_dens-1 then id = n_dens-2

  wt = (log_temp - log_temp_arr[it]) / (log_temp_arr[it+1] - log_temp_arr[it])
  wd = (log_dens - log_dens_arr[id]) / (log_dens_arr[id+1] - log_dens_arr[id])

  emiss_interp = (1-wt) * (1-wd) * reform(emiss_table[it,   id,   *]) + $
    (1-wt) * wd     * reform(emiss_table[it,   id+1, *]) + $
    wt     * (1-wd) * reform(emiss_table[it+1, id,   *]) + $
    wt     * wd     * reform(emiss_table[it+1, id+1, *])

  return, reform(emiss_interp)
end

;==============================================================================
function simulate_IPT_spectrum_Rayleighs_ERF_form, x, spec_binsize, xwavi, yptsi, fwhm
  rootc = 2d*sqrt(alog(2d))/fwhm
  nj = n_elements(x)
  ni = n_elements(yptsi)
  ypts = dblarr(nj)

  for i=0, ni-1 do begin
    if yptsi[i] le 0 then continue
    ypts += yptsi[i] * 0.5d * $
      (Erf((x - xwavi[i] + spec_binsize/2d)*rootc) - $
      Erf((x - xwavi[i] - spec_binsize/2d)*rootc))
  endfor

  ypts /= spec_binsize
  return, ypts
end

;==============================================================================
; RAY TRACING FUNCTIONS
;==============================================================================

function ray_box_intersection, ray_origin, ray_dir, box_min, box_max
  t_min = -1d30
  t_max = 1d30

  for i = 0, 2 do begin
    if abs(ray_dir[i]) gt 1d-10 then begin
      t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
      t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]

      if t1 gt t2 then begin
        temp = t1
        t1 = t2
        t2 = temp
      endif

      t_min = t_min > t1
      t_max = t_max < t2

      if t_min gt t_max then return, [-1d, -1d]

    endif else begin
      if (ray_origin[i] lt box_min[i]) or (ray_origin[i] gt box_max[i]) then $
        return, [-1d, -1d]
    endelse
  endfor

  if t_max lt 0 then return, [-1d, -1d]
  if t_min lt 0 then t_min = 0d

  return, [t_min, t_max]
end

;==============================================================================
function sample_ray_through_torus, ray_origin, ray_dir, ds
  ;+
  ; Generate sample points along ray through torus volume
  ; For field-aligned model, use 5-10 RJ radial range, ±3 RJ vertical
  ;-

  torus_rmin = 5.0d
  torus_rmax = 10.0d
  torus_zmax = 3.0d

  box_min = [-torus_rmax, -torus_rmax, -torus_zmax]
  box_max = [torus_rmax, torus_rmax, torus_zmax]

  t_range = ray_box_intersection(ray_origin, ray_dir, box_min, box_max)

  if t_range[0] eq -1 then begin
    return, {s: [0d], x: [ray_origin[0]], y: [ray_origin[1]], $
      z: [ray_origin[2]], valid: [0B], n_valid: 0L}
  endif

  n_samples = ceil((t_range[1] - t_range[0]) / ds) + 1
  s_array = t_range[0] + ds * dindgen(n_samples)

  if s_array[n_samples-1] lt t_range[1] then begin
    s_array = [s_array, t_range[1]]
    n_samples++
  endif

  x_array = ray_origin[0] + s_array * ray_dir[0]
  y_array = ray_origin[1] + s_array * ray_dir[1]
  z_array = ray_origin[2] + s_array * ray_dir[2]

  ; For field model, valid points are those within model domain
  rho_array = sqrt(x_array^2 + y_array^2)
  r_array = sqrt(x_array^2 + y_array^2 + z_array^2)
  lat_array = asin(z_array/r_array) * 180./!pi

  valid = (rho_array ge torus_rmin) and (rho_array le torus_rmax) and $
    (abs(lat_array) le 50.)  ; Within ±50° latitude coverage
  n_valid = total(valid)

  return, {s: s_array, x: x_array, y: y_array, z: z_array, $
    valid: valid, n_valid: n_valid}
end

;==============================================================================
function integrate_emission_along_los_field_model, ray_samples, field_model, $
  xwav, spec_binsize, fwhm, $
  temp_arr, dens_arr, xwavi_struct, yptsi_struct
  ;+
  ; Integrate emission along LOS using field-aligned model
  ;
  ; Similar to standard version but gets plasma parameters from
  ; interpolated field model rather than analytical functions
  ;-

  valid_idx = where(ray_samples.valid, n_valid)
  if n_valid eq 0 then return, dblarr(n_elements(xwav))

  ; Get plasma parameters at each point by interpolating field model
  nel_array = dblarr(n_valid)
  nsp_array = dblarr(n_valid)
  ns2p_array = dblarr(n_valid)
  ns3p_array = dblarr(n_valid)
  ns4p_array = dblarr(n_valid)
  nop_array = dblarr(n_valid)
  no2p_array = dblarr(n_valid)
  Tel_array = dblarr(n_valid)

  for i = 0L, n_valid-1 do begin
    idx = valid_idx[i]

    ; Interpolate from field model
    params = interpolate_field_model(ray_samples.x[idx], $
      ray_samples.y[idx], $
      ray_samples.z[idx], $
      field_model)

    if params.valid then begin
      nel_array[i] = params.nel
      nsp_array[i] = params.nsp
      ns2p_array[i] = params.ns2p
      ns3p_array[i] = params.ns3p
      ns4p_array[i] = params.ns4p
      nop_array[i] = params.nop
      no2p_array[i] = params.no2p
      Tel_array[i] = params.Te
    endif
  endfor

  ; Filter out points with no plasma
  good_idx = where(nel_array gt 1d and Tel_array gt 0.1d, n_good)
  if n_good eq 0 then return, dblarr(n_elements(xwav))

  ; Calculate path elements
  RJ_cm = 7.1492d9
  ds_array = dblarr(n_good)

  for i = 0L, n_good-1 do begin
    if i eq 0 then begin
      if n_good gt 1 then begin
        ds_array[i] = ray_samples.s[valid_idx[good_idx[1]]] - $
          ray_samples.s[valid_idx[good_idx[0]]]
      endif else begin
        ds_array[i] = 0.1d
      endelse
    endif else if i eq n_good-1 then begin
      ds_array[i] = ray_samples.s[valid_idx[good_idx[i]]] - $
        ray_samples.s[valid_idx[good_idx[i-1]]]
    endif else begin
      ds_array[i] = (ray_samples.s[valid_idx[good_idx[i+1]]] - $
        ray_samples.s[valid_idx[good_idx[i-1]]]) / 2d
    endelse
  endfor
  ds_cm = ds_array * RJ_cm

  ; Combine all wavelengths and intensities
  xwavi_all = [xwavi_struct.sp, xwavi_struct.s2p, xwavi_struct.s3p, $
    xwavi_struct.s4p, xwavi_struct.op, xwavi_struct.o2p]
  n_lines = n_elements(xwavi_all)
  yptsi_all = dblarr(n_lines)

  ; Integrate along LOS
  for i = 0L, n_good-1 do begin
    idx = good_idx[i]

    if nel_array[idx] lt 1d or Tel_array[idx] lt 0.1d then continue

    ; Get emissivities
    emiss_sp = interpolate_emissivity_2D(Tel_array[idx], nel_array[idx], $
      temp_arr, dens_arr, yptsi_struct.sp)
    emiss_s2p = interpolate_emissivity_2D(Tel_array[idx], nel_array[idx], $
      temp_arr, dens_arr, yptsi_struct.s2p)
    emiss_s3p = interpolate_emissivity_2D(Tel_array[idx], nel_array[idx], $
      temp_arr, dens_arr, yptsi_struct.s3p)
    emiss_s4p = interpolate_emissivity_2D(Tel_array[idx], nel_array[idx], $
      temp_arr, dens_arr, yptsi_struct.s4p)
    emiss_op = interpolate_emissivity_2D(Tel_array[idx], nel_array[idx], $
      temp_arr, dens_arr, yptsi_struct.op)
    emiss_o2p = interpolate_emissivity_2D(Tel_array[idx], nel_array[idx], $
      temp_arr, dens_arr, yptsi_struct.o2p)

    ; Accumulate emission
    yptsi_step = [emiss_sp * nsp_array[idx], $
      emiss_s2p * ns2p_array[idx], $
      emiss_s3p * ns3p_array[idx], $
      emiss_s4p * ns4p_array[idx], $
      emiss_op * nop_array[idx], $
      emiss_o2p * no2p_array[idx]]

    yptsi_all += 1d-6 * yptsi_step * ds_cm[i]
  endfor

  ; Sort by wavelength
  wsort = sort(xwavi_all)
  xwavi_sorted = xwavi_all[wsort]
  yptsi_sorted = yptsi_all[wsort]

  ; Apply instrument response
  spectrum = simulate_IPT_spectrum_Rayleighs_ERF_form(xwav, spec_binsize, $
    xwavi_sorted, yptsi_sorted, fwhm)

  return, spectrum
end

;==============================================================================
; MAIN PROCEDURE
;==============================================================================

pro sim_citep_2_diff_eq
  ;+
  ; Main procedure for calculating IPT UV emission using field-aligned
  ; diffusive equilibrium model
  ;-

  ; ============================================================================
  ; SETUP
  ; ============================================================================
  print, '================================================================'
  print, 'CITEP - Community Io Torus Emission Package'
  print, 'Field-Aligned Diffusive Equilibrium Model Version'
  print, '================================================================'

  ; Wavelength grid
  xmin = 550d
  xmax = 2100d
  binsize = 1d
  nx = round((xmax - xmin)/binsize) + 1
  xwav = dindgen(nx)*binsize + xmin

  ; Instrument resolution
  fwhm = 4.47d  ; [Å]

  ; ============================================================================
  ; LOAD MODELS
  ; ============================================================================
  print, ''
  print, 'Loading 3D field-aligned torus model...'
  field_model = load_3d_torus_model()

  print, ''
  print, 'Loading CHIANTI emission tables...'
  restore, 'CHIANTI_11.0.2_emiss_arrays_all_species_all_wavelengths_50x50_logspaced.sav'

  print, 'Tables loaded:'
  print, '  Temperature range: ', min(temp_arr), ' - ', max(temp_arr), ' eV'
  print, '  Density range: ', min(dens_arr), ' - ', max(dens_arr), ' cm^-3'

  ; ============================================================================
  ; DISPLAY MODEL STRUCTURE
  ; ============================================================================
  print, ''
  print, '================================================================'
  print, 'FIELD-ALIGNED MODEL STRUCTURE'
  print, '================================================================'
  print, 'Grid dimensions: 501 x 360 x 1001'
  print, '  501 field lines: ρ_eq = 5.00 to 10.00 R_J (0.01 R_J steps)'
  print, '  360 azimuths: φ = 0° to 359° (1° steps)'
  print, '  1001 latitudes: λ_III = -50° to +50° (0.1° steps)'
  print, ''
  print, 'Magnetic field model: JRM33 + CON2020'
  print, 'Plasma transport: Diffusive equilibrium along field lines'

  ; Show sample densities at equator
  print, ''
  print, 'Sample equatorial densities at φ = 180°:'
  print, '  ρ [R_J]   n_e [cm^-3]   n_S++ [cm^-3]   n_O+ [cm^-3]'
  for i = 0, 500, 50 do begin
    rho_val = 5.0 + i*0.01
    ; Extract arrays from structure
    nel_array = field_model.nel
    ns2p_array = field_model.ns2p
    nop_array = field_model.nop
    ne_eq = nel_array[i, 180, 500]  ; Latitude index 500 = equator
    ns2p_eq = ns2p_array[i, 180, 500]
    nop_eq = nop_array[i, 180, 500]
    if ne_eq gt 0 then $
      print, rho_val, ne_eq, ns2p_eq, nop_eq, format='(F7.2, 3E14.3)'
  endfor

  ; ============================================================================
  ; CALCULATE SPECTRA
  ; ============================================================================
  print, ''
  print, '================================================================'
  print, 'CALCULATING UV EMISSION SPECTRA'
  print, '================================================================'

  ds = 0.05d  ; Integration step size [R_J] - finer for better accuracy

  ; Example 1: Equatorial ansa observation
  print, ''
  print, 'Example 1: Equatorial ansa observation'
  ray_origin = [6d, -20d, 0d]
  ray_dir = [0d, 1d, 0d]

  ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds)
  print, '  Ray samples: ', ray_samples.n_valid, ' points through torus'

  spectrum_eq = integrate_emission_along_los_field_model(ray_samples, field_model, $
    xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)

  ; Example 2: Off-equatorial observation
  print, ''
  print, 'Example 2: Off-equatorial observation (30° latitude)'
  ray_origin = [6d, -20d, 3.46d]  ; 30° latitude at 6 R_J
  ray_dir = [0d, 1d, 0d]

  ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds)
  print, '  Ray samples: ', ray_samples.n_valid, ' points through torus'

  spectrum_off = integrate_emission_along_los_field_model(ray_samples, field_model, $
    xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)

  ; Example 3: Dawn vs Dusk asymmetry
  print, ''
  print, 'Example 3: Dawn observation (φ = 90°)'
  ray_origin = [-20d, 6d, 0d]  ; Dawn side
  ray_dir = [1d, 0d, 0d]

  ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds)
  spectrum_dawn = integrate_emission_along_los_field_model(ray_samples, field_model, $
    xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)

  print, 'Example 4: Dusk observation (φ = 270°)'
  ray_origin = [20d, -6d, 0d]  ; Dusk side
  ray_dir = [-1d, 0d, 0d]

  ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds)
  spectrum_dusk = integrate_emission_along_los_field_model(ray_samples, field_model, $
    xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)

  ; ============================================================================
  ; PLOTS
  ; ============================================================================

  ; Equatorial vs off-equatorial
  p1 = plot(xwav, spectrum_eq, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (R/Å)', $
    title='Field-Aligned Model: Latitude Effects', $
    thick=2, color='black', $
    name='Equatorial', $
    xrange=[550, 2100])

  p2 = plot(xwav, spectrum_off, $
    /overplot, $
    thick=2, color='blue', $
    name='30° Latitude', $
    transparency=30)

  leg1 = legend(target=[p1, p2], $
    position=[0.7, 0.8], $
    /normal)

  ; Dawn-Dusk asymmetry
  p3 = plot(xwav, spectrum_dawn, $
    xtitle='Wavelength (Å)', $
    ytitle='Intensity (R/Å)', $
    title='Field-Aligned Model: Dawn-Dusk Asymmetry', $
    thick=2, color='red', $
    name='Dawn (90°)', $
    xrange=[550, 2100])

  p4 = plot(xwav, spectrum_dusk, $
    /overplot, $
    thick=2, color='orange', $
    name='Dusk (270°)', $
    transparency=30)

  leg2 = legend(target=[p3, p4], $
    position=[0.7, 0.8], $
    /normal)

  ; ============================================================================
  ; 2D EMISSION MAP
  ; ============================================================================
  print, ''
  print, '================================================================'
  print, 'GENERATING 2D EMISSION MAP'
  print, '================================================================'

  n_map = 31  ; Reduced for field model due to computational cost
  x_positions = -10d + 20d * dindgen(n_map)/(n_map-1)
  z_positions = -10d + 20d * dindgen(n_map)/(n_map-1)

  total_emission = dblarr(n_map, n_map)

  ray_dir = [0d, 1d, 0d]

  print, 'Calculating ', n_map, 'x', n_map, ' pixel map...'

  for i = 0, n_map-1 do begin
    for j = 0, n_map-1 do begin
      ray_origin = [x_positions[i], -20d, z_positions[j]]

      ray_samples = sample_ray_through_torus(ray_origin, ray_dir, ds)

      if ray_samples.n_valid gt 0 then begin
        spectrum = integrate_emission_along_los_field_model(ray_samples, $
          field_model, xwav, binsize, fwhm, temp_arr, dens_arr, xwavi, yptsi)
        total_emission[i,j] = total(spectrum)
      endif
    endfor
    if (i mod 5) eq 0 then print, '  Row ', i+1, ' of ', n_map, ' complete'
  endfor

  c = contour(total_emission, x_positions, z_positions, $
    /fill, $
    n_levels=20, $
    xtitle='X Position (R_J)', $
    ytitle='Z Position (R_J)', $
    title='Total UV Emission: Field-Aligned Model', $
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
  print, 'Model features captured:'
  print, '  - Non-dipolar magnetic field (JRM33+CON2020)'
  print, '  - Field-aligned plasma distribution'
  print, '  - Dawn-dusk asymmetries'
  print, '  - Realistic vertical stratification'
  print, ''
  print, 'Total integrated emission (550-2100 Å):'
  print, '  Equatorial: ', total(spectrum_eq), ' Rayleighs'
  print, '  30° Latitude: ', total(spectrum_off), ' Rayleighs'
  print, '  Dawn: ', total(spectrum_dawn), ' Rayleighs'
  print, '  Dusk: ', total(spectrum_dusk), ' Rayleighs'
  print, ''
  print, 'Dawn/Dusk ratio: ', total(spectrum_dawn)/total(spectrum_dusk)

  stop
end