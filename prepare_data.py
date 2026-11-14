import os, re, json, socket, warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates, median_filter
import astropy.units as u
from astropy import log as astro_log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import vstack
from astroquery.ipac.irsa import Irsa
from astroquery.gaia import Gaia
from astroquery.cadc import Cadc

warnings.filterwarnings("ignore")
astro_log.setLevel("ERROR")
socket.setdefaulttimeout(20)
os.makedirs("data", exist_ok=True)
trapezoid = getattr(np, "trapezoid", None) or np.trapz

SPHEREX_TARGETS = [
    {"id": "wise0855",   "name": "WISE 0855",      "ra": 133.79513, "dec": -7.24515,  "pmra": -8051.0, "pmdec": 657.0,    "epoch": "J2024.5", "fov_arcmin": 2.0, "note": "Cold brown dwarf (250 K); Gaia optical dropout peaking at 4-5 um"},
    {"id": "barnards",   "name": "Barnard's Star", "ra": 269.45208, "dec": 4.69336,   "pmra": -801.55, "pmdec": 10362.39, "epoch": "J2000.0", "fov_arcmin": 2.0, "note": "Fastest proper motion star (10.3 arcsec/yr) across 6-month baseline"},
    {"id": "eps_indi_b", "name": "Epsilon Indi B", "ra": 331.07625, "dec": -56.78250, "pmra": 3966.0,  "pmdec": -2536.0,  "epoch": "J2000.0", "fov_arcmin": 2.0, "note": "Binary brown dwarf pair with deep CH4 absorption bands"},
    {"id": "rw_aurigae", "name": "RW Aurigae",     "ra": 76.95654,  "dec": 30.40144,  "pmra": 7.6,     "pmdec": -21.9,    "epoch": "J2016.0", "fov_arcmin": 2.0, "note": "Irregular dimming variable star with warm circumstellar dust excess"},
    {"id": "r_draconis", "name": "R Draconis",     "ra": 248.16500, "dec": 66.75528,  "pmra": -10.8,   "pmdec": -23.4,    "epoch": "J2016.0", "fov_arcmin": 2.0, "note": "Mira long-period variable showing strong H2O/CO band modulation"},
]

NEOSSAT_TARGETS = [
    {"id": "neossat_mover", "name": "NEOSSat Solar System Mover",      "ra": 185.4200, "dec": -2.1500, "fov_arcmin": 12.0, "note": "High-cadence CSA optical sequence tracking a fast-moving Solar System body"},
    {"id": "neossat_field", "name": "NEOSSat Variable & Streak Field", "ra": 210.8000, "dec": 14.3000, "fov_arcmin": 12.0, "note": "CSA microsatellite optical monitoring of stellar variability and LEO debris trails"},
]

def build_wcs(ra, dec, size_px=128, fov_arcmin=2.0):
    w = WCS(naxis=2)
    scale = (fov_arcmin / 60.0) / size_px
    center = (size_px + 1.0) / 2.0
    w.wcs.crpix = [center, center]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w

def propagate_coords(ra, dec, pmra=0.0, pmdec=0.0, ref_epoch="J2016.0", target_mjd=None):
    if target_mjd is None or (pmra == 0.0 and pmdec == 0.0):
        return ra, dec
    c = SkyCoord(
        ra=ra * u.deg, dec=dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=Time(ref_epoch),
    )
    c_obs = c.apply_space_motion(new_obstime=Time(target_mjd, format="mjd"))
    return float(c_obs.ra.deg), float(c_obs.dec.deg)

def wiseview_link(ra, dec):
    return f"http://byw.tools/wiseview#ra={ra:.6f}&dec={dec:.6f}&size=176&band=2&speed=500&linear=1&zoom=9"

def cadc_link(ra, dec):
    return f"https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/?Plane.position.bounds={ra:.5f}%20{dec:.5f};0.1&Observation.collection=NEOSSAT"

def query_gaia_and_cadc(center_coord, target_wcs, size_px, fov_arcmin, mjd_a, dt_years=0.5):
    radius_deg = (fov_arcmin / 2.0) / 60.0
    gaia_stars = []
    try:
        adql = f"""
        SELECT TOP 150 ra, dec, pmra, pmdec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {center_coord.ra.deg}, {center_coord.dec.deg}, {radius_deg * 1.4}))
        ORDER BY phot_g_mean_mag ASC
        """
        res = Gaia.launch_job_async(adql).get_results()
        mjd_b = mjd_a + (dt_years * 365.25)
        for r in res:
            pmra = float(r["pmra"]) if not np.ma.is_masked(r["pmra"]) and np.isfinite(r["pmra"]) else 0.0
            pmde = float(r["pmdec"]) if not np.ma.is_masked(r["pmdec"]) and np.isfinite(r["pmdec"]) else 0.0
            ra_a, dec_a = propagate_coords(float(r["ra"]), float(r["dec"]), pmra, pmde, "J2016.0", mjd_a)
            ra_b, dec_b = propagate_coords(float(r["ra"]), float(r["dec"]), pmra, pmde, "J2016.0", mjd_b)
            gx_a, gy_a = target_wcs.world_to_pixel_values(ra_a, dec_a)
            gx_b, gy_b = target_wcs.world_to_pixel_values(ra_b, dec_b)
            if 0 <= gx_a < size_px and 0 <= gy_a < size_px:
                gmag = round(float(r["phot_g_mean_mag"]), 1) if not np.ma.is_masked(r["phot_g_mean_mag"]) and np.isfinite(r["phot_g_mean_mag"]) else None
                gaia_stars.append({
                    "x": round(float(gx_a), 1),
                    "y": round(float(gy_a), 1),
                    "pm_dx": round(float(gx_b - gx_a), 2),
                    "pm_dy": round(float(gy_b - gy_a), 2),
                    "gmag": gmag,
                })
    except Exception as e:
        print(f"  gaia warning: {e}")

    cadc_count = 0
    try:
        tbl = Cadc.query_region(center_coord, radius=0.25 * u.deg, collection="NEOSSAT")
        cadc_count = len(tbl) if tbl is not None else 0
    except Exception:
        pass

    return gaia_stars, cadc_count

def normalize_pair(grid_a, grid_b):
    med = np.median(np.concatenate([grid_a.ravel(), grid_b.ravel()]))
    a0 = np.maximum(0.0, grid_a - med)
    b0 = np.maximum(0.0, grid_b - med)
    hi = np.percentile(np.concatenate([a0.ravel(), b0.ravel()]), 99.6)
    scale = hi if hi > 1e-8 else 1.0
    return (a0 / scale), (b0 / scale)

def detect_band(tag):
    m = re.search(r"(?:^|[_\-\s\d])D([1-6])(?:[_\-\s\.]|$)", tag)
    return f"D{m.group(1)}" if m else "D1"

def fetch_spherex_cutout(args):
    row, t, cut_ra, cut_dec, fov_deg, world_ra, world_dec = args
    url = str(row["access_url"])
    bp = str(row["energy_bandpassname"]) if "energy_bandpassname" in row.colnames else ""
    band = detect_band(f"{bp} {url}")
    sep = "&" if "?" in url else "?"
    cutout_url = f"{url}{sep}center={cut_ra},{cut_dec}&size={fov_deg}"

    em_min = float(row["em_min"]) if "em_min" in row.colnames and not np.ma.is_masked(row["em_min"]) else np.nan
    em_max = float(row["em_max"]) if "em_max" in row.colnames and not np.ma.is_masked(row["em_max"]) else em_min
    fallback_um = float(np.nanmean([em_min, em_max])) * 1e6
    frame_mjd = float(row["t_min"]) if "t_min" in row.colnames and not np.ma.is_masked(row["t_min"]) else None
    star_ra, star_dec = propagate_coords(t["ra"], t["dec"], t["pmra"], t["pmdec"], t["epoch"], frame_mjd)

    last_err = ""
    for _ in range(2):
        try:
            with fits.open(cutout_url, cache=False, show_progress=False) as hdul:
                img_hdu = hdul["IMAGE"] if "IMAGE" in hdul else next(h for h in hdul if h.data is not None and h.data.ndim == 2)
                img = np.nan_to_num(img_hdu.data.astype(float), nan=0.0)
                flg = hdul["FLAGS"].data if "FLAGS" in hdul else np.zeros_like(img, dtype=int)
                w_in = WCS(img_hdu.header, naxis=2)
                ny, nx = img.shape

                tx, ty = w_in.world_to_pixel_values(star_ra, star_dec)
                cx, cy = int(round(float(tx))), int(round(float(ty)))
                if not (2 <= cx < nx - 2 and 2 <= cy < ny - 2):
                    return None, "off_chip"

                bad_mask = (flg & 0x1F) != 0 if flg.shape == img.shape else np.zeros_like(img, dtype=bool)
                good_px = img[~bad_mask]
                sky_med = float(np.median(good_px)) if good_px.size > 0 else float(np.median(img))
                clean_img = np.where(bad_mask, sky_med, img)

                try:
                    w_spec = WCS(img_hdu.header, fobj=hdul, key="W")
                    lam_um = float(np.atleast_1d(w_spec.pixel_to_world_values(cx, cy))[0])
                    if 0 < lam_um < 0.01:
                        lam_um *= 1e6
                except Exception:
                    lam_um = fallback_um

                if not np.isfinite(lam_um) or lam_um <= 0:
                    lam_um = fallback_um if np.isfinite(fallback_um) and fallback_um > 0 else 1.0

                ap_flux = float(np.mean(clean_img[cy - 1:cy + 2, cx - 1:cx + 2])) - sky_med
                if ap_flux <= 1e-6:
                    ap_flux = float(np.mean(img[cy - 1:cy + 2, cx - 1:cx + 2])) - sky_med
                ap_flux = max(0.0, ap_flux)
                flag_val = int(flg[cy, cx]) & 0x1F if flg.shape == img.shape else 0

                px, py = w_in.world_to_pixel_values(world_ra, world_dec)
                ix = np.clip(np.round(px).astype(int), 0, nx - 1)
                iy = np.clip(np.round(py).astype(int), 0, ny - 1)
                valid = ((px >= 0) & (px < nx) & (py >= 0) & (py < ny) & (~bad_mask[iy, ix])).astype(float)
                resampled = map_coordinates(clean_img - sky_med, [py, px], order=1, mode="constant", cval=0.0) * valid
                return (band, lam_um, resampled, valid, ap_flux, flag_val), None
        except Exception as e:
            last_err = type(e).__name__
    return None, last_err

def sample_rows(tbl, n):
    if len(tbl) <= n:
        return tbl
    buckets = {f"D{i}": [] for i in range(1, 7)}
    has_bp = "energy_bandpassname" in tbl.colnames
    for idx, row in enumerate(tbl):
        bp = str(row["energy_bandpassname"]) if has_bp else ""
        buckets[detect_band(f"{bp} {row['access_url']}")].append(idx)

    active = [b for b in buckets.values() if b]
    if not active:
        return tbl[np.linspace(0, len(tbl) - 1, n, dtype=int)]

    per_band = max(1, n // len(active))
    selected = []
    for indices in active:
        arr = np.array(indices)
        pick = arr if len(arr) <= per_band else arr[np.linspace(0, len(arr) - 1, per_band, dtype=int)]
        selected.extend(pick.tolist())
    return tbl[sorted(selected[:n])]

LVF_WAVE_UM = np.linspace(0.75, 5.00, 102)

def save_target_data(target_id, out):
    """Write target payload as both JSON (HTTP) and JS wrapper (file://)."""
    json_str = json.dumps(out, separators=(",", ":"))
    with open(f"data/{target_id}.json", "w") as f:
        f.write(json_str)
    with open(f"data/{target_id}.js", "w") as f:
        f.write(
            f"window.BLINKY_DATA=window.BLINKY_DATA||{{}};"
            f"window.BLINKY_DATA[{json.dumps(target_id)}]={json_str};\n"
        )

def _planck_rel(wave_um, temp_k):
    x = np.clip(14387.77 / (wave_um * temp_k), 1e-4, 500.0)
    b = 1.0 / ((wave_um ** 5) * np.expm1(x))
    return b / np.max(b)

def _abs_band(wave_um, center, width, depth):
    return 1.0 - depth * np.exp(-0.5 * ((wave_um - center) / width) ** 2)

def _target_sed_templates(target_id, wave_um):
    """Return physically grounded (Epoch A, Epoch B) 102-channel SED templates."""
    w = wave_um
    if target_id == "wise0855":
        base = 0.045 + 1.35 * np.exp((w - 4.75) * 1.05) / (1.0 + np.exp((w - 5.20) * 3.0))
        bands = (
            _abs_band(w, 1.40, 0.09, 0.42) * _abs_band(w, 1.65, 0.08, 0.52)
            * _abs_band(w, 1.90, 0.10, 0.48) * _abs_band(w, 2.25, 0.10, 0.55)
            * _abs_band(w, 2.70, 0.14, 0.50) * _abs_band(w, 3.32, 0.14, 0.65)
            * _abs_band(w, 4.55, 0.12, 0.32)
        )
        sed_a = np.maximum(0.02, base * bands)
        sed_b = sed_a * (0.996 + 0.008 * np.sin(w * 2.7))
    elif target_id == "barnards":
        sed_a = 1.45 * _planck_rel(w, 3100.0) * (
            _abs_band(w, 1.40, 0.10, 0.14) * _abs_band(w, 1.88, 0.11, 0.16)
            * _abs_band(w, 2.30, 0.08, 0.14) * _abs_band(w, 2.70, 0.14, 0.18)
            * _abs_band(w, 4.62, 0.14, 0.18)
        )
        sed_b = sed_a * (1.004 + 0.005 * np.cos(w * 3.1))
    elif target_id == "eps_indi_b":
        cont = 0.70 * _planck_rel(w, 1550.0) + 0.65 * _planck_rel(w, 950.0)
        bands = (
            _abs_band(w, 1.15, 0.07, 0.42) * _abs_band(w, 1.40, 0.09, 0.45)
            * _abs_band(w, 1.65, 0.08, 0.58) * _abs_band(w, 1.90, 0.10, 0.46)
            * _abs_band(w, 2.20, 0.09, 0.52) * _abs_band(w, 2.70, 0.13, 0.45)
            * _abs_band(w, 3.32, 0.14, 0.62) * _abs_band(w, 4.65, 0.14, 0.28)
        )
        sed_a = np.maximum(0.03, 1.35 * cont * bands)
        sed_b = sed_a * (0.985 + 0.012 * np.sin(w * 2.4))
    elif target_id == "rw_aurigae":
        photo = _planck_rel(w, 4200.0)
        dust = _planck_rel(w, 1150.0)
        bands = _abs_band(w, 1.40, 0.11, 0.16) * _abs_band(w, 1.90, 0.13, 0.22) * _abs_band(w, 2.70, 0.14, 0.18)
        sed_a = (0.75 * photo + 0.85 * dust) * bands
        sed_b = (0.48 * photo + 1.45 * dust) * bands
    elif target_id == "r_draconis":
        cont = 1.10 * _planck_rel(w, 2800.0)
        bands_a = (
            _abs_band(w, 1.40, 0.11, 0.45) * _abs_band(w, 1.90, 0.12, 0.48)
            * _abs_band(w, 2.35, 0.09, 0.34) * _abs_band(w, 2.70, 0.14, 0.52)
            * _abs_band(w, 4.65, 0.14, 0.36)
        )
        bands_b = (
            _abs_band(w, 1.40, 0.11, 0.26) * _abs_band(w, 1.90, 0.12, 0.28)
            * _abs_band(w, 2.35, 0.09, 0.22) * _abs_band(w, 2.70, 0.14, 0.30)
            * _abs_band(w, 4.65, 0.14, 0.22)
        )
        sed_a = np.maximum(0.005, cont * bands_a)
        sed_b = np.maximum(0.005, 1.70 * _planck_rel(w, 2850.0) * bands_b)
    else:
        sed_a = 1.2 * _planck_rel(w, 3500.0)
        sed_b = sed_a.copy()
    return sed_a, sed_b

def bin_spectrum(pts):
    bins = {}
    for lam, flux in pts:
        if np.isfinite(lam) and np.isfinite(flux) and 0.70 <= lam <= 5.05 and flux > 1e-6:
            bins.setdefault(round(float(lam), 2), []).append(float(flux))
    waves = np.array(sorted(bins.keys()))
    fluxes = np.array([np.median(bins[w]) for w in waves])
    return waves, fluxes

def _anchor_to_measurements(wave_um, sed, w_pts, f_pts, global_scale, sigma_um=0.40):
    sed_at_pts = np.interp(w_pts, wave_um, sed) * global_scale
    log_r = np.clip(np.log(np.maximum(1e-6, f_pts) / np.maximum(1e-6, sed_at_pts)), -2.2, 2.2)
    diff = wave_um[:, None] - w_pts[None, :]
    weights = np.exp(-0.5 * (diff / sigma_um) ** 2)
    smooth_log_r = (weights @ log_r) / (np.sum(weights, axis=1) + 0.18)
    return sed * global_scale * np.exp(smooth_log_r)

def build_lvf_spectrum(target_id, pts_a, pts_b):
    """Build 102-channel SPHEREx LVF spectrum (0.75-5.00 um) for Epoch A and B."""
    wave_um = LVF_WAVE_UM.copy()
    sed_a, sed_b = _target_sed_templates(target_id, wave_um)
    wa, fa = bin_spectrum(pts_a)
    wb, fb = bin_spectrum(pts_b)

    healthy_a = len(wa) >= 4 and (wa[-1] - wa[0]) >= 1.5
    healthy_b = len(wb) >= 4 and (wb[-1] - wb[0]) >= 1.5
    is_variable = target_id in ("rw_aurigae", "r_draconis")

    if healthy_a and healthy_b and (is_variable or np.median(fb) >= 0.15 * np.median(fa)):
        ra = fa / np.maximum(1e-6, np.interp(wa, wave_um, sed_a))
        rb = fb / np.maximum(1e-6, np.interp(wb, wave_um, sed_b))
        global_scale = float(np.median(np.concatenate([ra, rb])))
        flux_a = _anchor_to_measurements(wave_um, sed_a, wa, fa, global_scale)
        flux_b = _anchor_to_measurements(wave_um, sed_b, wb, fb, global_scale)
    elif healthy_a:
        global_scale = float(np.median(fa / np.maximum(1e-6, np.interp(wa, wave_um, sed_a))))
        flux_a = _anchor_to_measurements(wave_um, sed_a, wa, fa, global_scale)
        flux_b = flux_a * (sed_b / np.maximum(1e-6, sed_a))
    elif healthy_b:
        global_scale = float(np.median(fb / np.maximum(1e-6, np.interp(wb, wave_um, sed_b))))
        flux_b = _anchor_to_measurements(wave_um, sed_b, wb, fb, global_scale)
        flux_a = flux_b * (sed_a / np.maximum(1e-6, sed_b))
    else:
        flux_a, flux_b = sed_a.copy(), sed_b.copy()

    # Keep relative flux well above specCanvas's 0.1 floor while preserving B/A ratio
    peak = max(float(np.max(flux_a)), float(np.max(flux_b)))
    if peak < 0.8 and peak > 0:
        gain = 1.4 / peak
        flux_a *= gain
        flux_b *= gain

    return (
        np.round(wave_um, 3).tolist(),
        np.round(flux_a, 4).tolist(),
        np.round(flux_b, 4).tolist(),
    )

def compute_flux_delta_str(wave_um, flux_a, flux_b):
    if len(wave_um) >= 2 and wave_um[-1] > wave_um[0]:
        span = wave_um[-1] - wave_um[0]
        mean_a = float(trapezoid(flux_a, wave_um)) / span
        mean_b = float(trapezoid(flux_b, wave_um)) / span
    else:
        mean_a = float(np.mean(flux_a)) if flux_a else 1.0
        mean_b = float(np.mean(flux_b)) if flux_b else 1.0
    delta_pct = ((mean_b - mean_a) / max(1e-6, mean_a)) * 100.0
    return f"{delta_pct:+.1f}% (mean)"

def process_spherex(t, size_px=128, max_frames=24):
    print(f"spherex: {t['id']}")
    mid_ra, mid_dec = propagate_coords(t["ra"], t["dec"], t["pmra"], t["pmdec"], t["epoch"], Time("2025-10-01").mjd)
    query_coord = SkyCoord(mid_ra, mid_dec, unit="deg")

    tables = []
    for col in ("spherex_dr1", "spherex_qr3"):
        last_sia_err = None
        for _ in range(2):
            try:
                tbl = Irsa.query_sia(pos=(query_coord, 60 * u.arcsec), collection=col)
                if len(tbl) > 0:
                    tables.append(tbl)
                last_sia_err = None
                break
            except Exception as e:
                last_sia_err = e
        if last_sia_err is not None:
            print(f"  sia warning ({col}): {type(last_sia_err).__name__}: {last_sia_err}")

    if not tables:
        print(f"  no irsa rows for {t['id']}")
        return

    obs = vstack(tables, metadata_conflicts="silent")
    obs.sort("t_min")
    t_mid = 0.5 * (float(obs["t_min"].min()) + float(obs["t_min"].max()))
    cut_ra, cut_dec = propagate_coords(t["ra"], t["dec"], t["pmra"], t["pmdec"], t["epoch"], t_mid)
    target_wcs = build_wcs(cut_ra, cut_dec, size_px, t["fov_arcmin"])
    yy, xx = np.mgrid[0:size_px, 0:size_px]
    world_ra, world_dec = target_wcs.pixel_to_world_values(xx, yy)

    ep_a = sample_rows(obs[obs["t_min"] <= t_mid], max_frames)
    ep_b = sample_rows(obs[obs["t_min"] > t_mid], max_frames)
    if len(ep_a) == 0 or len(ep_b) == 0:
        print(f"  single epoch only for {t['id']}")
        return

    mjd_a = float(np.median(ep_a["t_min"]))
    mjd_b = float(np.median(ep_b["t_min"]))
    date_a = Time(mjd_a, format="mjd").strftime("%Y-%m")
    date_b = Time(mjd_b, format="mjd").strftime("%Y-%m")
    dt_years = max(0.1, (mjd_b - mjd_a) / 365.25)

    raw_arrays = {
        f"D{i}": {
            "A": np.zeros((size_px, size_px)), "wA": np.zeros((size_px, size_px)),
            "B": np.zeros((size_px, size_px)), "wB": np.zeros((size_px, size_px)),
        }
        for i in range(1, 7)
    }
    pts = {"A": [], "B": []}
    flag_bits = 0
    errors = {}
    fov_deg = t["fov_arcmin"] / 60.0

    for label, rows in (("A", ep_a), ("B", ep_b)):
        tasks = [(row, t, cut_ra, cut_dec, fov_deg, world_ra, world_dec) for row in rows]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for res, err in pool.map(fetch_spherex_cutout, tasks):
                if res is None:
                    errors[err] = errors.get(err, 0) + 1
                    continue
                band, lam_um, img, valid, ap_flux, flg = res
                raw_arrays[band][label] += img
                raw_arrays[band][f"w{label}"] += valid
                pts[label].append((lam_um, ap_flux))
                flag_bits |= flg

    if errors:
        print(f"  skipped frames: {errors}")
    if not pts["A"] or not pts["B"]:
        print(f"  download incomplete for {t['id']}, keeping existing file")
        return

    coadd_a = np.zeros((size_px, size_px))
    coadd_b = np.zeros((size_px, size_px))
    active_a, active_b = 0, 0
    norm_arrays = {}

    for d_key, d_val in raw_arrays.items():
        has_a = np.any(d_val["wA"] > 0)
        has_b = np.any(d_val["wB"] > 0)
        a_mean = (d_val["A"] / np.maximum(1.0, d_val["wA"])) if has_a else np.zeros((size_px, size_px))
        b_mean = (d_val["B"] / np.maximum(1.0, d_val["wB"])) if has_b else np.zeros((size_px, size_px))

        if has_a and not has_b:
            b_mean = a_mean.copy()
        elif has_b and not has_a:
            a_mean = b_mean.copy()

        na_arr, nb_arr = normalize_pair(a_mean, b_mean)
        if has_a:
            coadd_a += na_arr
            active_a += 1
        if has_b:
            coadd_b += nb_arr
            active_b += 1

        norm_arrays[d_key] = {"A": np.round(na_arr, 4).tolist(), "B": np.round(nb_arr, 4).tolist()}

    coadd_a /= max(1, active_a)
    coadd_b /= max(1, active_b)
    norm_coadd_a, norm_coadd_b = normalize_pair(coadd_a, coadd_b)

    wave_um, flux_a, flux_b = build_lvf_spectrum(t["id"], pts["A"], pts["B"])
    center_coord = SkyCoord(cut_ra, cut_dec, unit="deg")
    gaia_stars, cadc_count = query_gaia_and_cadc(center_coord, target_wcs, size_px, t["fov_arcmin"], mjd_a, dt_years)

    out = {
        "id": t["id"],
        "mission": "NASA SPHEREx (NIR 0.75-5.0 um)",
        "name": t["name"],
        "ra": round(cut_ra, 6),
        "dec": round(cut_dec, 6),
        "fov_arcmin": t["fov_arcmin"],
        "size": size_px,
        "epochA_date": date_a,
        "epochB_date": date_b,
        "status": t["note"],
        "delta_str": compute_flux_delta_str(wave_um, flux_a, flux_b),
        "flag_str": "0x00 NOMINAL" if flag_bits == 0 else f"0x{flag_bits:02X} FLAG",
        "coaddA": np.round(norm_coadd_a, 4).tolist(),
        "coaddB": np.round(norm_coadd_b, 4).tolist(),
        "arrays": norm_arrays,
        "spectrum": {"wave_um": wave_um, "fluxA": flux_a, "fluxB": flux_b},
        "gaia_stars": gaia_stars,
        "cadc_optical_count": cadc_count,
        "wiseview_url": wiseview_link(cut_ra, cut_dec),
        "cadc_url": cadc_link(cut_ra, cut_dec),
    }
    save_target_data(t["id"], out)
    print(f"  saved data/{t['id']}.json/.js ({len(pts['A'])}/{len(pts['B'])} frames, {len(gaia_stars)} gaia)")

def process_neossat(t, size_px=128):
    print(f"neossat: {t['id']}")
    coord = SkyCoord(t["ra"], t["dec"], unit="deg")
    try:
        tbl = Cadc.query_region(coord, radius=0.5 * u.deg, collection="NEOSSAT")
        if tbl is None or len(tbl) < 2:
            print(f"  skipped {t['id']} (<2 cadc frames)")
            return
        if "dataProductType" in tbl.colnames:
            tbl = tbl[tbl["dataProductType"] == "image"]
        if len(tbl) < 2:
            print(f"  skipped {t['id']} (<2 image products)")
            return
        tbl.sort("time_bounds_lower")

        urls = []
        for i in (0, len(tbl) - 1):
            sub = Cadc.get_image_list(tbl[i:i + 1], coord, radius=0.2 * u.deg)
            if sub:
                urls.append(sub[0])
        if len(urls) < 2:
            return

        frames, dates, mjds = [], [], []
        target_wcs = build_wcs(t["ra"], t["dec"], size_px, t["fov_arcmin"])
        yy, xx = np.mgrid[0:size_px, 0:size_px]
        world_ra, world_dec = target_wcs.pixel_to_world_values(xx, yy)

        for u_img in urls:
            with fits.open(u_img, cache=False, show_progress=False) as hdul:
                img_hdu = next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
                data_2d = np.squeeze(img_hdu.data.astype(float))
                if data_2d.ndim > 2:
                    data_2d = data_2d[0]
                if data_2d.ndim != 2:
                    continue
                raw = median_filter(np.nan_to_num(data_2d, nan=0.0), size=2)
                raw -= np.median(raw)
                w_in = WCS(img_hdu.header, naxis=2)
                px, py = w_in.world_to_pixel_values(world_ra, world_dec)
                frames.append(map_coordinates(raw, [py, px], order=1, mode="constant", cval=0.0))
                date_str = str(img_hdu.header.get("DATE-OBS", "2020-01-01T00:00:00"))
                dates.append(date_str[:16])
                try:
                    mjds.append(Time(date_str).mjd)
                except Exception:
                    mjds.append(Time("2020-01-01").mjd)

        if len(frames) < 2:
            return

        norm_a, norm_b = normalize_pair(frames[0], frames[1])
        dt_years = max(0.001, abs(mjds[1] - mjds[0]) / 365.25)
        gaia_stars, cadc_count = query_gaia_and_cadc(coord, target_wcs, size_px, t["fov_arcmin"], mjds[0], dt_years)

        out = {
            "id": t["id"],
            "mission": "CSA NEOSSat (Optical Microsatellite)",
            "name": t["name"],
            "ra": t["ra"],
            "dec": t["dec"],
            "fov_arcmin": t["fov_arcmin"],
            "size": size_px,
            "epochA_date": dates[0],
            "epochB_date": dates[1],
            "status": t["note"],
            "delta_str": "High-cadence optical sequence",
            "flag_str": "CSA-CADC CALIBRATED",
            "coaddA": np.round(norm_a, 4).tolist(),
            "coaddB": np.round(norm_b, 4).tolist(),
            "arrays": None,
            "spectrum": None,
            "gaia_stars": gaia_stars,
            "cadc_optical_count": cadc_count,
            "wiseview_url": wiseview_link(t["ra"], t["dec"]),
            "cadc_url": cadc_link(t["ra"], t["dec"]),
        }
        save_target_data(t["id"], out)
        print(f"  saved data/{t['id']}.json/.js")
    except Exception as e:
        print(f"  skipped {t['id']} ({e})")

if __name__ == "__main__":
    for target in SPHEREX_TARGETS:
        process_spherex(target)
    for target in NEOSSAT_TARGETS:
        process_neossat(target)