# Blinky

A browser-based blink comparator and spectral viewer for inspecting multi-epoch infrared and optical sky observations from **NASA SPHEREx**, **ESA Gaia DR3**, and **CSA NEOSSat**.

## Running the App

Open `index.html` directly in a browser, or serve the directory locally:

```bash
python3 -m http.server 8000
```

All five SPHEREx datasets are bundled in `data/` (as `.json` and `.js` files) so the viewer works offline without a backend server.

## Controls

- **Target:** Switch between five SPHEREx near-infrared targets and two NEOSSat optical presets.
- **View:** Toggle **Epoch A**, **Epoch B**, **Blink** (500 ms alternating loop), or **Diff (B − A)** (red shows brightening in Epoch B; blue shows dimming or a star's earlier position).
- **Detector:** View the 6-band coadd or isolate individual SPHEREx detector arrays (`D1`–`D6`, `0.75–5.00 μm`).
- **Gaia DR3 overlay:** Plots optical Gaia DR3 source positions (green circles) and proper-motion vectors (`> 0.3 px`) to separate optical dropouts from visible field stars.
- **Spectrum & External Archives:** Compares Epoch A (dashed blue) and Epoch B (solid red) across 102 wavelength channels (`0.75–5.00 μm`), alongside links to WiseView and CADC.

## Targets

| ID | Object | Mission | Science Case | Data |
| :--- | :--- | :--- | :--- | :--- |
| `wise0855` | WISE 0855−0714 | NASA SPHEREx | ~250 K Y-type brown dwarf; optical dropout peaking at `4–5 μm` | IRSA FITS cutouts + Gaia DR3 |
| `barnards` | Barnard's Star | NASA SPHEREx | High proper motion M4V dwarf (`10.3"/yr`) | IRSA FITS cutouts + Gaia DR3 |
| `eps_indi_b` | Epsilon Indi B | NASA SPHEREx | T1+T6 binary brown dwarf with $\text{CH}_4$ absorption | IRSA FITS cutouts + Gaia DR3 |
| `rw_aurigae` | RW Aurigae | NASA SPHEREx | Irregular T Tauri variable with circumstellar dust excess | IRSA FITS cutouts + Gaia DR3 |
| `r_draconis` | R Draconis | NASA SPHEREx | Mira variable showing $\text{H}_2\text{O}$ and $\text{CO}$ band changes | IRSA FITS cutouts + Gaia DR3 |
| `neossat_mover` | Solar System Mover | CSA NEOSSat | Fast-moving Solar System object sequence | Preview model |
| `neossat_field` | Variable & Streak Field | CSA NEOSSat | Stellar variability and LEO satellite trail | Preview model |

## Data Reduction (`prepare_data.py`)

The files in `data/` are built by `prepare_data.py`:

1. **Archive Queries:** Queries IPAC/IRSA (`spherex_dr1` and `spherex_qr3`) for Level-2 FITS cutouts, ESA Gaia DR3 (`gaiadr3.gaia_source`) for optical astrometry, and CADC (`NEOSSAT`) for optical frame counts.
2. **Astrometry and Reprojection:** Propagates catalog positions to each exposure's observation epoch (`SkyCoord.apply_space_motion`), masks flagged pixels (`FLAGS & 0x1F`), subtracts the frame sky median, and resamples cutouts onto a common `128×128` tangent-plane (`RA---TAN`) grid.
3. **Coadding and Scaling:** Averages exposures by per-pixel coverage weight and normalizes Epoch A and B together against their joint median and 99.6th percentile before `asinh` scaling.
4. **Spectra:** Extracts $3\times 3$ aperture fluxes at the target coordinates per frame and interpolates across the 102-channel (`0.75–5.00 μm`) grid using molecular absorption templates ($\text{H}_2\text{O}$, $\text{CH}_4$, $\text{CO}$) anchored to the measured IRSA points.

To regenerate the datasets from live archives:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 prepare_data.py
```

## Structure

```text
blinky/
├── index.html
├── prepare_data.py
├── requirements.txt
├── README.md
└── data/
```