# Penia Map

## Contents

1. [Project overview](#1-project-overview)
2. [Data format](#2-data-format)
3. [Repository structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Usage](#6-usage)
7. [Signal processing](#7-signal-processing)
8. [Output metrics](#8-output-metrics)
9. [Data quality](#9-data-quality)
10. [Reproducibility](#10-reproducibility)
11. [Citation and license](#11-citation-and-license)

## 1. Project overview

This code was developed in the context of the ASTROPOWER project. The project studies older adults as analog astronauts on Earth, since ageing and microgravity can lead to similar physiological declines. One of these declines, powerpenia (loss of muscle power), is emerging as a new approach that could help detect functional decline earlier. ASTROPOWER aims to develop an acoustic biofeedback tool, intended to guide muscle tension and contraction during exercise and to help counteract the effects of ageing and microgravity.

More specifically, this code was developed within the master’s thesis entitled *“Penia Map: integrating muscle-based biomarkers for predicting biological age”*. The main goal of the thesis is to predict biological age from indicators of the musculoskeletal system. To this end, a dataset was collected to characterise age-related musculoskeletal decline across four dimensions: sarcopenia (loss of muscle mass), dynapenia (loss of muscle strength), powerpenia (loss of muscle power) and osteopenia (reduced bone mineral density). Body composition and anthropometric data were acquired, together with physiological data, namely surface electromyography (EMG) of the forearm muscles and kinematic and kinetic variables extracted during dynamic and isometric handgrip tasks. Demographic information was also recorded. These data were then used to predict biological age proxies through linear regression models.

The main purpose of the code is therefore to extract metrics from dynamic and isometric handgrip contractions, namely from EMG, force and displacement signals (`features_power_dyna.xlsx`); to merge them with body composition, anthropometric and demographic data (`final_dataframe.xlsx`); and, finally, to run regression models for different biological age proxies and save their results (`results.xlsx`).

## 2. Data format

Recordings are **OpenSignals text files** (`.txt`). The number of columns depends on the recording type. Signal values are raw ADC values (bits).

### 2.1 MVC and IHD files (4 columns)

| Column | Description                         |
|--------|-------------------------------------|
| `nSeq` | Sequence number                     |
| `DI`   | Digital I/O states (not used)       |
| `EMG`  | EMG, extensor carpi radialis (bits) |
| `EMG`  | EMG, flexor carpi radialis (bits)   |

### 2.2 DHD_HUB files (7 columns)

| Column | Description                         |
|--------|-------------------------------------|
| `nSeq` | Sequence number                     |
| `DI`   | Digital I/O states (not used)       |
| `EMG`  | EMG, extensor carpi radialis (bits) |
| `EMG`  | EMG, flexor carpi radialis (bits)   |
| `XYZ`  | Accelerometer, X axis (bits)        |
| `XYZ`  | Accelerometer, Y axis (bits)        |
| `XYZ`  | Accelerometer, Z axis (bits)        |

### 2.3 DHD and calibration files (10 columns)

| Column                    | Description                          |
|---------------------------|--------------------------------------|
| `nSeq`                    | Sequence number                      |
| `DI`                      | Digital I/O states (not used)        |
| `DISP`                    | Displacement (bits)                  |
| `FORCE`                   | Force (bits)                         |
| `gACCx`, `gACCy`, `gACCz` | Accelerometer, X / Y / Z axes (bits) |
| `pMAGx`, `pMAGy`, `pMAGz` | Magnetometer, X / Y / Z axes (bits)  |

### 2.4 Excel files

All `.xlsx` files contain an `ID` column identifying the participant.

`bioimpedance_data.xlsx` — anthropometric and body composition data from the bioimpedance scale:

- Height (cm), body mass (kg), body mass index (kg/m²)
- Fat percentage: global, right arm, left arm, right leg, left leg, torso
- Muscle mass (kg): global, right arm, left arm, right leg, left leg, torso
- Estimated bone mass (kg)
- Visceral fat rating (index of visceral fat level)
- Daily calorie intake, DCI (kcal): estimated calories needed to maintain body weight
- Estimated metabolic age (years)
- Global body water (%)

`demographic_data.xlsx` — participant demographics:

- Age (years)
- Gender (female, male)
- Dominant hand (right, left)
- Marital status (single, married, divorced, widowed, de facto union)
- Academic level:
  - 1st cycle (1st–4th grade)
  - 2nd cycle (5th–6th grade)
  - compulsory education (7th–9th grade)
  - secondary education (10th–12th grade)
  - technical and vocational education
  - higher education (bachelor’s degree)
  - postgraduate higher education (master’s or doctoral degree)

`handle_positions.xlsx` — distance between the DHD handles (cm), needed to convert raw displacement into physical units.

`ihd_values.xlsx` — values from the isometric contractions measured with the IHD (kg).

`Acc_calib.txt` — calibration data (bits) of the accelerometer on top of the DHD, needed to apply its transfer function.

## 3. Repository structure

```

├── main.py                          # raw files transform to one feature row per participant (powerpenia and dynapenia domains)
├── constants.py                     # sampling frequency (FS) and column names
├── utils.py                         # builds the per-participant feature row
├── requirements.txt
├── template_report_2.html           # template for participant report generation
│
├── data_loader/
│   ├── data_loading.py              # loads OpenSignals and auxiliary Excel files; nSeq integrity analysis
│   ├── files_validation.py          # sample loss/gap report for all recordings → files_validation.xlsx
│   └── data_visualizer.py           # plotting helpers
│
├── processor/
│   ├── transfer_function.py         # applies all sensor transfer functions
│   ├── acc.py                       # accelerometer conversion
│   ├── displacement.py              # displacement conversion
│   ├── load_cell.py                 # force conversion and smoothing
│   ├── emg.py                       # EMG conversion, filtering and envelope
│   ├── emg_normalization.py         # MVC reference values and %MVC normalisation
│   ├── interpolator.py              # lost-sample insertion and interpolation
│   └── Acc_calib_new.txt            # accelerometer calibration data
│
├── algorithms/
│   ├── synchronization.py           # DHD ↔ DHD_HUB synchronisation with drift correction
│   ├── trimming.py                  # interactive selection / application of trim regions
│   ├── grip_closing_phase_detection.py  # concentric/eccentric phases from the force derivative
│   ├── onset_detection.py           # EMG onset/offset detection (TKEO-based)
│   ├── validate_onsets.py           # manual validation of detected onsets/offsets
│   └── conversion_bonemass_bonedensity.py  # NHANES-based BMD prediction equation
│
├── metrics/
│   ├── powerpenia/
│   │   ├── peak_power_calculation.py    # Peak power calculation
│   │   └── powerpenia_metrics.py        # Peak power normalization using anthropometric measures
│   │
│   ├── sarcopenia/
│   │   └── sarcopenia_metrics.py        # Muscle mass metrics and ALMI calculation
│   │
│   ├── osteopenia/
│   │   └── osteopenia_metrics.py        # Bone mass metrics and estimated BMD calculation
│   │
│   ├── muscle_coordination_metrics.py   # Inter-muscle coordination metrics
│   ├── spectral_density_metrics.py      # Welch PSD, band powers, median and peak frequency
│   └── metrics_aggregation.py           # Merges all metrics into final_dataframe.xlsx (run main.py first)
│
├── models/
│   ├── correlation_calculation.py   # adaptive Pearson/Spearman correlations
│   ├── feature_reduction_and_linear_regression.py  # best-subset linear regression
│   ├── remove_features.py           # removes band-power / mean / normalised-IHD columns
│   └── run_regression.py            # runs the regressions and exports results (run metrics_aggregation first)
│
│
├── participants_characteristics/
│   ├── graph_penia_map.py               # Generates Penia Map radar charts
│   ├── participant_distribution.py      # Age and sex distribution analysis
│   ├── sample_characterization.py       # Sample characterization by sex, BMI and physical activity level
│   └── teste_bmi.py                     # Overweight analysis
│
├── reliability_analysis_tanita/
│   ├── bland_altman.py                  # Generates Bland-Altman plots
│   ├── ICC.py                            # Intraclass correlation coefficient (ICC) calculation
│   └── run_reliability_analysis.py      # Runs reliability analysis comparing Tanita bioimpedance estimates with DXA
│
└── report_participants/
    ├── Add_additional_tanita_info.py    # Adds supplementary Tanita information not used in the models
    ├── Add_participant_name.py          # Adds participant names to reports
    └── Participant_report_generator.py # Generates personalized PDF reports for all participants
```

## 4. Installation

**Requirements:** Python 3.13. All Python dependencies are listed in `requirements.txt`.

```bash
git clone https://github.com/sofiapgarcia/penia_map.git
cd penia_map
git checkout organized
pip install -r requirements.txt
```

## 5. Configuration

Before running, set the paths below (they are defined inside each script).

| Script                                                | Variable          | Points to                                                 |
|-------------------------------------------------------|-------------------|-----------------------------------------------------------|
| `main.py`                                             | *(dataset path)*  | Dataset folder                                            |
| `data_loader/files_validation.py`                     | `base_path`       | Dataset folder                                            |
| `metrics/metrics_aggregation.py`                      | `path_tanita`     | Bioimpedance Excel file                                   |
|                                                       | `path_power_dyna` | Power and dynapenia features                              |
|                                                       | `output_path`     | Where to save the final dataset                           |
| `models/remove_features.py`                           | `INPUT_FILE`      | Final dataset                                             |
|                                                       | `OUTPUT_FILE`     | Where to save the filtered dataset                        |
| `models/run_regression.py`                            | `filepath`        | Dataset used in the regression models                     |
| `algorithms/conversion_bonemass_bonedensity.py`       | `DATA_DIR`        | Data used to obtain the BMD equation                      |
| `participants_characteristics/*` (except `teste_bmi`) | `EXCEL_PATH`      | Final dataframe                                           |
| `participants_characteristics/teste_bmi`              | `EXCEL_PATH`      | Bioimpedance variables                                    |
| `report_participants/age_groups`                      | `FILE_PATH`       | Final dataframe (all participants)                        |
| `report_participants/add_additional_tanita_info`      | `FILE_1`          | Final dataframe with age groups                           |
|                                                       | `FILE_2`          | Bioimpedance variables                                    |
| `report_participants/add_participant_name`            | `DATA_FILE`       | Dataframe with extra bioimpedance variables and age group |
|                                                       | `NAMES_FILE`      | Excel file with participant ID and name                   |

**Fixed parameters**

| Item                      | Value                                | Where                       |
|---------------------------|--------------------------------------|-----------------------------|
| Sampling frequency        | value in Hz                          | `constants.py`              |
| Displacement calibration  | closed-grip position `X_MAX = 29` mm | `processor/displacement.py` |
| Accelerometer calibration | `Acc_calib_new.txt`                  | `processor/acc.py`          |

## 6. Usage

### Main pipeline

Run the steps in this order; each step depends on the previous one.

```bash
python main.py                                   #  features_power_dyna.xlsx
cd metrics && python metrics_aggregation.py      # final_dataframe.xlsx
cd ..
python -m models.remove_features                 #  final_dataframe_filtered.xlsx
python -m models.run_regression                  #  regression results (Excel)
```

### Additional analyses

```bash
python -m data_loader.files_validation                          # loss/gap report → files_validation.xlsx
python -m reliability_analysis_tanita.run_reliability_analysis  # bioimpedance scale vs DXA agreement
python -m teste_bmi                                             # overweight analysis
python -m graph_penia_map                                       # Penia Map radar charts by age group
```

### Sample characterisation

```bash
python -m participant_distribution
python -m sample_characterization
```

### Participant reports

Run in this order:

```bash
python -m age_groups
python -m add_additional_tanita_info
python -m add_participant_name
python -m participant_report_generator
```

## 7. Signal processing

Steps are applied in this order in `main.py`, for each condition and participant.

1.  **Transfer functions** (`processor/transfer_function.py`)

    - Hub accelerometer: per-axis calibration (`Acc_calib_new.txt`), scaled to m/s².
    - DHD accelerometer: `(ADC − 2¹⁵) · 8 / 2¹⁶ · 9.8` (m/s²).
    - Displacement: `x = a·N + b`, with `a = (x0 − X_MAX) / (Nmax − N0)` and `b = −a·N0` (mm).
    - EMG: `((ADC / 2¹⁶ − 0.5) · Vcc) / gain · 1000` (mV).
    - Force: `0.0158709 · N` (N).

2.  **Missing-sample handling** (`processor/interpolator.py`): gaps are detected in `nSeq`, NaN rows are inserted at the missing positions, the counter is unwrapped, and all channels are filled by **linear interpolation**.

3.  **EMG filtering** (`processor/emg.py`): DC removal, 50 Hz notch, 4th-order Butterworth band-pass 10–499 Hz. All filters are zero-phase (`filtfilt`).

4.  **Force processing** (`processor/load_cell.py`): mean removal, then a centred (zero-phase) moving average (default 20 ms).

5.  **Synchronisation** (`algorithms/synchronization.py`): DHD and DHD_HUB are aligned using the accelerometer signals (cross-correlation, expected rhythm 90 bpm), with linear clock-drift correction between a start and an end anchor. Saved parameters are re-applied, and the two recordings are merged into a single DHD dataframe.

6.  **Trimming** (`algorithms/trimming.py`): removes the regions highlighted interactively or listed in `trim_indices.xlsx`.

7.  **Phase detection** (`algorithms/grip_closing_phase_detection.py`, dynamic handgrip)

    - Force derivative, Gaussian-smoothed (σ = 20 ms).
    - The 12 most prominent positive peaks (≥ 1 s apart, ≥ 30 % of the maximum) define the repetitions.
    - Onset, concentric→eccentric turning point and offset are found with a local threshold of 20 % of each repetition’s peak derivative.
    - Samples are labelled `C<n>` (grip closing / concentric), `E<n>` (grip opening / eccentric) or `R` (rest).

8.  **Velocity and power** (`metrics/powerpenia/peak_power_calculation.py`)

    - Velocity = derivative of displacement, smoothed with a 5 ms centred moving average.
    - **Power = velocity × force.**
    - Peak power, maximum velocity and maximum force are taken **within concentric phases only**.

9.  **EMG envelope** (`processor/emg.py`): full-wave rectification + 4th-order Butterworth low-pass at 4 Hz (zero-phase).

10. **MVC normalisation** (`processor/emg_normalization.py`)

    - Extensor reference: maximum of the envelope (and of the filtered signal) in the IHD recording.
    - Flexor reference: maximum in the MVC recording.
    - DHD and IHD signals are expressed as % MVC.

11. **EMG onset/offset detection** (`algorithms/onset_detection.py`): TKEO-based energy envelope (a simplified, deterministic version of the EstOpt approach).

    - DHD: onset searched around the concentric start of each repetition; the concentric end comes from the phase labels.
    - IHD: threshold = 25 % of the median of the top 10 % envelope values; active segments ≥ 300 ms, merged when separated by \< 400 ms; 3 contractions per muscle.
    - All onsets/offsets are then **manually validated** (`algorithms/validate_onsets.py`).

12. **Muscle-coordination metrics** (`metrics/muscle_coordination_metrics.py`): computed for the **peak-power repetition** (DHD) and the **maximum-handgrip repetition** (IHD).

13. **Spectral metrics** (`metrics/spectral_density_metrics.py`): Welch PSD per repetition, between onset and concentric end (DHD) or between onset and offset (IHD). Outputs power in 50 Hz bands, median frequency, peak frequency and flexor/extensor ratios.

## 8. Output metrics

### 8.1 Variable groups (`features_power_dyna.xlsx`)

| Group / prefix                                                                             | Description                                                                | Unit              |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|-------------------|
| `peak_power`, `velocity_at_peak_power`, `force_at_peak_power`, `max_velocity`, `max_force` | Dynamic handgrip power summary (concentric phase)                          | W, m/s, N, m/s, N |
| `rep_pp_dhd_*`                                                                             | Coordination and spectral metrics of the peak-power repetition (DHD)       | see 8.2           |
| `rep_max_ihd_*`                                                                            | Coordination and spectral metrics of the maximum-handgrip repetition (IHD) | see 8.2           |
| `mean_dhd_*`, `mean_ihd_*`                                                                 | Mean of each metric across repetitions                                     | see 8.2           |
| `dhd_Median_Frequency_<muscle>_slope`, `ihd_…`                                             | Slope of the median frequency across repetitions                           | Hz/repetition     |
| `ihd_values_*`                                                                             | Maximum and mean of the isometric handgrip values H1–H3                    | kg                |

### 8.2 Coordination and spectral metrics

**Coordination metrics — dynamic handgrip dynamometer (DHD)**

| Metric                                                   | Description                                                                     | Unit    |
|----------------------------------------------------------|---------------------------------------------------------------------------------|---------|
| `onset_diff_ms`, `onset_diff_ms_abs`                     | Time difference between the EMG onsets of the two muscles (signed and absolute) | ms      |
| `cocontraction_ms`, `cocontraction_normalized`           | Duration of simultaneous activation of both muscles (absolute and normalised)   | ms, –   |
| `emg_peak_extensor`, `emg_peak_flexor`                   | Peak EMG envelope of each muscle                                                | % MVC   |
| `emg_peak_ratio`                                         | Ratio between the flexor and extensor EMG peaks                                 | –       |
| `envelope_peak_delay_ms` (+ `_abs`, normalised versions) | Delay between the envelope peaks of the two muscles                             | ms      |
| `peak_rfd`                                               | Peak rate of force development, over a 20 ms window                             | N/s     |
| `emg_overlap_area` (+ `_normalized`)                     | Area where the two muscles’ envelopes overlap                                   | % MVC·s |

**Coordination metrics — isometric handgrip dynamometer (IHD)**

| Metric             | Description                                              |
|--------------------|----------------------------------------------------------|
| `rms_*`            | Root mean square of the EMG, per muscle                  |
| `iemg_*`           | Integrated EMG, per muscle                               |
| `mean_emg_*`       | Mean EMG, per muscle                                     |
| EMG overlap area   | Area where the two muscles’ envelopes overlap (IHD task) |

**Spectral metrics**

| Metric                    | Description                                            | Unit |
|---------------------------|--------------------------------------------------------|------|
| `<low>-<high>Hz_<muscle>` | Band power in 50 Hz bands from 20 to 470 Hz            | –    |
| `Median_Frequency_*`      | Median frequency of the power spectrum                 | Hz   |
| `Peak_Frequency_*`        | Frequency of the spectral peak                         | Hz   |
| Flexor/extensor ratios    | Ratio between the two muscles for the spectral metrics | –    |

### 8.3 `final_dataframe.xlsx` (after `metrics_aggregation.py`)

Adds, per participant:

- `peak_power_mass` (W/kg) and `peak_power_height_squared` (W/m²): peak power normalised by body mass and by height².
- `ALMI` (kg/m²): appendicular lean mass index — lean mass of arms + legs divided by height².
- Estimated bone mass (kg).
- Bone mineral density (BMD), from the NHANES 2001–2002 prediction equation using height, estimated bone mass, age and sex.
- Demographics: sex (male = 1, female = 0), age, height, body mass, BMI, metabolic age, activity level.
- Muscle mass variables: `Global muscle kg`, `Arm muscle (right) kg`, `Arm muscle (left) kg`, `Leg muscle (right) kg`, `Leg muscle (left) kg`.

### 8.4 Other outputs

| Output                                                            | Content                                                                                                                           |
|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `files_validation.xlsx`                                           | Per recording: wraps, missing samples, expected/acquired samples, loss %, number and length of gaps, invalid jumps, validity flag |
| `sync_indices.xlsx`, `trim_indices.xlsx`, `onsets_validated.xlsx` | Per participant: saved synchronisation, trimming and onset-validation decisions                                                   |
| Regression results (Excel)                                        | Linear regression model results                                                                                                   |
| Sample characterisation                                           | Tables and figures (age histogram, sex donut, radar charts)                                                                       |
| Reliability analysis                                              | Bland–Altman plots and ICC tables                                                                                                 |
| Participant reports                                               | Personalised PDF reports                                                                                                          |

## 9. Data quality

**Lost samples.** `analyze_sequence` uses the `nSeq` counter (wraps at 32768) to count missing samples, gaps and invalid jumps. Missing rows are inserted as NaN and linearly interpolated in all channels. A recording is flagged `invalid_file` if more than 10 % of its samples are missing.

**Signal validation.**

- Loss/gap report for all recordings (`files_validation.py`).
- Optional diagnostic plots in every processing module (`show_plot` flags).
- Manual validation of synchronisation, trimming and EMG onsets/offsets.
