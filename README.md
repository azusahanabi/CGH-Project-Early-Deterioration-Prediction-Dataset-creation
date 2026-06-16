# CGH Inpatient Deterioration Dataset Preparation

This README explains how `dataset_nonsequential_final.ipynb` builds non-sequential modelling datasets for the CGH W36 inpatient deterioration project.

It is written for readers who have not seen the project before. It explains the prediction task, input data, cohort rule, cleaning logic, feature engineering, split design, output files, and mentor-review audit checks.

The final notebook now supports **two related but different dataset tracks**:

1. **Main renewed track: case-level ward-episode outcome prediction**  
   This creates one row per case for each selected feature-window design. It is mainly for **HD/ICU transfer and death**, because these outcomes are not well represented as short future-horizon events inside W36.

2. **Optional continuity track: sliding-window short-horizon prediction**  
   This creates one row per case-window and predicts whether a timed event occurs in the next few hours. This remains useful for **IV increase, O2 increase, and MET activation**, where event timestamps are available inside the W36 monitoring period.

---

## 1. Quick Orientation

### 1.1 One-sentence project objective

The project builds modelling datasets to predict clinical deterioration in CGH W36 inpatients using information available before the prediction point.

Because not all deterioration outcomes happen in the same way, the final notebook separates the prediction task into two tracks.

| Track | Main question | Best suited outcomes |
|---|---|---|
| Case-level ward-episode track | Using a defined part of the W36 stay, can we predict whether this case eventually has HD/ICU transfer or death? | HD/ICU transfer, 1-month death, 3-month death, any death indicator, HD/ICU-or-death composite |
| Sliding-window short-horizon track | At this prediction time, will a new deterioration event happen in the next 4, 6, or 12 hours? | IV increase, O2 increase, MET activation, optional timed-event analysis |

### 1.2 Why a new case-level dataset was needed for HD/ICU transfer and death

The earlier sliding-window design is clinically reasonable for events that happen during the W36 ward stay and have reliable timestamps. However, HD/ICU transfer and death behave differently in this dataset.

Important project observations:

- HD/ICU transfer often happens at or around `Movement End DT`, which is the end of the observable W36 ward episode.
- Death mostly happens after transfer to ICU or after the patient has already left W36.
- Very few death events happen inside the W36 ward itself.
- Some death information exists as 1-month or 3-month death status rather than as a precise in-ward event timestamp.

Therefore, forcing HD/ICU transfer and death into a 4-hour, 6-hour, or 12-hour in-ward future horizon produces zero or very few positive windows. That does not mean HD/ICU transfer or death are unimportant. It means they should be modelled with a different dataset design.

The renewed case-level dataset asks a more suitable question:

> Based on information from selected parts of the W36 stay, can we predict whether the case eventually has HD/ICU transfer or death?

### 1.3 What is one row in each dataset?

| Dataset track | One row means | Example |
|---|---|---|
| Case-level ward-episode dataset | One case under one feature-window design | Case A using `early_12h` features |
| Sliding-window short-horizon dataset | One prediction window for one case | Case A, window from 08:00 to 14:00, predicting 14:00 to 02:00 |

This distinction is central. The case-level dataset is not trying to predict an event in the next few hours. The sliding-window dataset is.

### 1.4 Key time terms

| Term | Meaning |
|---|---|
| `W36 Start DT` | Start of the W36 monitoring episode for one case |
| `Movement End DT` | End of the observable W36 monitoring episode |
| `feature_window_start` / `feature_window_end` | Time interval used to extract case-level features |
| `window_start` / `window_end` | Time interval used for one sliding-window observation window |
| `horizon_end` | End of the future prediction horizon in the sliding-window track |
| `m_hours` | Sliding-window observation length: 4, 6, 12, or 24 hours |
| `h_hours` | Sliding-window prediction horizon: 4, 6, or 12 hours |

---

## 2. Data Sources

### 2.1 Original raw data structure

The project combines two data batches:

- `240822`
- `240920`

Each batch originally contains five source datasets.

| Source type | Description |
|---|---|
| `Respiree_PatientInfo_ano` | Patient admission, ward-stay, demographic, and episode-level information |
| `eHINTS_data_ano` | Deterioration-related outcome fields, including HD/ICU transfer, O2 increase, IV increase, MET activation, and death-related fields |
| `eHINTS_diag_ano` | Diagnosis information |
| `eHINTS_vitalSigns_ano` | Manually charted vital signs |
| `Respiree_vital_sign` | Respiree sensor vital-sign time series |

### 2.2 Files directly loaded by the final notebook

The final notebook does **not** directly reload the ten raw files. It starts from prepared merged CSV files stored under:

```text
/content/drive/My Drive/merged_240920_240822_prepared
```

| Prepared file | Role in final dataset pipeline |
|---|---|
| `case_master_all_240920_240822.csv` | Main case-level table with patient information, W36 episode times, source-availability flags, and deterioration outcome fields |
| `charted_vitals_all_240920_240822.csv` | Manually charted vital-sign records |
| `diagnosis_summary_all_240920_240822.csv` | Case-level diagnosis summary; the final notebook uses `secondary_diagnosis_list` |
| `sensor_timeseries_all_240920_240822.csv` | Respiree sensor vital-sign time series |
| `cohort_summary_all_240920_240822.csv` | Cohort-checking summary file loaded for audit/reference |

### 2.3 Output root folder

The updated notebook saves output under:

```text
/content/drive/My Drive/Dataset/non-sequential
```

The main output subfolders are:

| Folder | Purpose |
|---|---|
| `case_level_ward_outcome/` | Main renewed case-level datasets for HD/ICU/death-style ward-episode outcomes |
| `sliding_window_short_horizon/` | Optional dynamic short-horizon datasets for timed IV/O2/MET-style prediction |
| `mentor_review_audit/` | Additional audit outputs requested before modelling |

---

## 3. Cohort Selection Rule

The modelling cohort keeps only cases that appear in all required project data sources.

A case is included only when all source-availability flags below are equal to 1:

```python
has_ehints_data == 1
has_diagnosis == 1
has_charted_vitals == 1
has_respiree_sensor == 1
```

The retained cases are then filtered to valid W36 monitoring episodes:

```python
W36 Start DT_dt is not missing
Movement End DT_dt is not missing
Movement End DT_dt > W36 Start DT_dt
```

Therefore, each final case has outcome information, diagnosis information, charted vitals, Respiree sensor vitals, and a valid W36 start/end time.

The retained full cohort contains **441 cases** in the generated summaries.

---

## 4. Data Cleaning Steps

### 4.1 Standardize case IDs

All loaded tables receive a cleaned `case_no_deid_clean` column.

The cleaning removes extra spaces and fixes IDs that may be read as decimal-looking values.

```text
123.0 -> 123
```

This reduces failed joins caused by formatting differences across files.

### 4.2 Convert datetime columns

Important case-level, charted-vital, and sensor-vital time columns are converted into datetime format. Invalid datetime values are coerced to missing.

| Column | Purpose |
|---|---|
| `W36 Start DT_dt` | Monitoring episode start |
| `Movement End DT_dt` | Monitoring episode end |
| `O2 increased provision DT_dt` / `o2_increase_event_dt` | O2 increase event time source |
| `Increased IV DT_dt` | IV increase event time |
| `MET activation after W36 Admission_dt` | MET activation event time |
| `Death Date_dt` | Death event time |
| `next hd/icu admission dt_dt` | HD/ICU transfer event time |
| `created_datetime_dt` | Charted-vital measurement time |
| `sensor_datetime_dt` | Sensor-vital measurement time |

### 4.3 Define case-level outcome flags

The notebook first defines deterioration at case level. These flags answer: **did this case ever have this event?**

| Case-level label | Main logic |
|---|---|
| `label_hd_icu` | `HD/ICU == 1` or non-missing next HD/ICU admission time |
| `label_o2_increase` | Uses `o2_increase_confirmed == 1` when available; otherwise falls back to O2 event timestamp |
| `label_iv_increase` | Non-missing IV increase timestamp |
| `label_met` | Non-missing MET activation timestamp |
| `label_death` | Death 1M, Death 3M, or non-missing death date |
| `label_any_deterioration` | Any of the five event labels is positive |

### 4.4 Define usable event timestamps

Window-level labels require event timestamps. A case may be case-level positive but still lack a usable event time.

The notebook creates these event-time columns:

| Event-time column | Event |
|---|---|
| `event_time_hd_icu` | HD/ICU transfer |
| `event_time_o2` | Confirmed O2 increase |
| `event_time_iv` | IV increase |
| `event_time_met` | MET activation |
| `event_time_death` | Death |

For O2 increase, if the case is not confirmed as O2 increase, its O2 timestamp is set to missing for labelling. This avoids false positive O2 windows caused by non-confirmed timestamps.

### 4.5 Remove internal labelling columns before final saving

Event-time columns and `Movement End DT` are needed for labelling, eligibility, and censoring. They are removed from final model-ready CSVs after labels are created.

---

## 5. Secondary Diagnosis Extraction and Categorization

The notebook uses **secondary diagnosis only**.

Project background suggests that secondary diagnoses were mostly available before W36 ward admission, so they can represent baseline comorbidity and risk burden. Primary diagnosis is intentionally excluded because it may be finalized later, including after W36 admission or after discharge, which could introduce leakage.

In plain words:

```text
Use secondary diagnosis as baseline risk context.
Do not use primary diagnosis as a main predictor in the main modelling dataset.
```

### 5.1 Input column

The diagnosis input comes from:

```text
diagnosis_summary_all_240920_240822.csv
```

The notebook uses these columns:

| Column | Role |
|---|---|
| `case_no_deid_clean` | Case identifier |
| `secondary_diagnosis_list` | Pipe-separated secondary diagnosis text |

A typical secondary-diagnosis field may contain multiple diagnoses separated by `|`.

### 5.2 Extraction steps

The extraction process is:

1. Keep `case_no_deid_clean` and `secondary_diagnosis_list`.
2. Standardize the case ID.
3. Split `secondary_diagnosis_list` by `|`.
4. Explode the list so that each row contains one case and one diagnosis phrase.
5. Strip extra spaces.
6. Remove empty or placeholder values such as empty strings, `nan`, `None`, and `**********`.
7. Remove duplicate diagnosis phrases for the same case.
8. Normalize diagnosis text by lowercasing it and removing punctuation.

The normalized text is used only for keyword matching. The original diagnosis description is still used in the frequency-checking file.

### 5.3 Diagnosis groups created

The grouped secondary-diagnosis features include these categories:

| Feature group | Main clinical meaning |
|---|---|
| `secondary_dx_infection_sepsis` | Infection, sepsis, bacteraemia, UTI-related terms |
| `secondary_dx_pneumonia` | Pneumonia and bronchopneumonia |
| `secondary_dx_respiratory` | Respiratory failure, COPD, asthma, hypoxia, pulmonary oedema, pleural effusion, and related terms |
| `secondary_dx_cardiac` | Heart failure, ischemic heart disease, myocardial infarction, arrhythmia, coronary disease, and related terms |
| `secondary_dx_thrombosis_embolism` | Pulmonary embolism, thrombosis, DVT |
| `secondary_dx_renal_any` | Any renal, kidney, nephropathy, or dialysis-related term |
| `secondary_dx_acute_kidney_failure` | AKI, acute kidney injury, acute renal failure |
| `secondary_dx_ckd_stage_3_5_or_dialysis` | CKD stage 3-5 or dialysis |
| `secondary_dx_diabetes_complication` | Diabetes with complications or poor control |
| `secondary_dx_hypertension` | Hypertension-related terms |
| `secondary_dx_electrolyte_acid_base_volume` | Electrolyte, acid-base, dehydration, volume depletion, or fluid overload terms |
| `secondary_dx_anemia_bleeding_coagulation` | Anaemia, bleeding, thrombocytopenia, pancytopenia, neutropenia, and coagulation terms |
| `secondary_dx_malignancy` | Cancer, carcinoma, neoplasm, lymphoma, leukaemia, tumour/tumor |
| `secondary_dx_metastatic_cancer` | Metastatic or secondary malignant disease |
| `secondary_dx_neuro_delirium_stroke` | Delirium, dementia, stroke, seizure, Parkinson-related, neurological terms |
| `secondary_dx_liver_cirrhosis_failure` | Cirrhosis, hepatic failure, liver failure, portal hypertension |
| `secondary_dx_gi_acute` | GI haemorrhage, obstruction, perforation, peritonitis, ileus, melena, haematemesis |
| `secondary_dx_frailty_pressure_ulcer_dysphagia_fall` | Frailty, pressure ulcer, dysphagia, fall, malnutrition, cachexia, walking difficulty |
| `secondary_dx_hypotension_shock` | Hypotension and shock-related terms |

### 5.4 Case-level diagnosis features

After grouping, the notebook creates one row per case with these features:

| Feature | Meaning |
|---|---|
| `secondary_diagnosis_count` | Number of unique cleaned secondary-diagnosis phrases for the case |
| `secondary_dx_*` | One dummy variable per diagnosis group |
| `secondary_dx_severe_group_count` | Number of secondary-diagnosis groups flagged for the case |

If a full-cohort case has no usable secondary diagnosis after cleaning, the notebook still keeps the case and fills diagnosis features with 0. This avoids silently dropping cases during diagnosis merging.

---

## 6. Vital Sign Processing

The notebook creates one unified long-format vital-sign table.

```text
case_no_deid_clean | vital_time | vital_name | vital_value
```

### 6.1 Charted vital signs

Charted vital signs are filtered to full-cohort cases, assigned `vital_time`, and converted into long format if needed.

Expected charted vital types include:

```text
HR, RR, SpO2, temperature, SBP, DBP
```

The notebook also standardizes source-specific or duplicated names.

```text
charted_HR -> HR
charted_charted_HR -> HR
```

### 6.2 Sensor vital signs

Sensor records are retained only when:

```python
is_linked_to_case == 1
within_patientinfo_episode == 1
sensor_datetime_dt is not missing
```

Sensor columns are converted from wide format to long format.

Sensor vital types include:

```text
RR, HR, SpO2, skin_temperature, body_temperature
```

### 6.3 Unified vital-sign names

Clinically comparable charted and sensor variables are merged into one stream.

| Source variables | Unified variable |
|---|---|
| Charted HR + sensor HR | `HR` |
| Charted RR + sensor RR | `RR` |
| Charted SpO2 + sensor SpO2 | `SpO2` |
| Charted temperature + sensor body temperature | `temperature` |
| Sensor skin temperature | `skin_temperature` |
| Charted SBP | `SBP` |
| Charted DBP | `DBP` |

`skin_temperature` is kept separate because it is not the same clinical variable as body temperature.

### 6.4 Vital-sign summary features

For each vital sign inside each selected observation window, the notebook creates these summaries.

| Feature suffix | Meaning |
|---|---|
| `_mean` | Average value in the window |
| `_min` | Minimum value in the window |
| `_max` | Maximum value in the window |
| `_std` | Standard deviation in the window; set to 0 if only one value exists |
| `_count` | Number of measurements in the window |
| `_last` | Last observed value before the window end |
| `_first` | First observed value in the window |
| `_delta` | Last value minus first value |
| `_slope` | Change per hour from first to last value |
| `_time_since_last` | Hours between last measurement and the window end |
| `_missing_flag` | 1 if this vital sign is missing in the window; otherwise 0 |

### 6.5 Baseline/recent split and short-recent features

Each observation window is split into two equal parts:

```text
baseline = earlier half of the observation window
recent   = later half of the observation window
```

The notebook calculates vital summaries for both halves, then creates recent-minus-baseline contrast features for:

```text
mean, min, max, count
```

It also creates fixed short-recent summaries immediately before the prediction or feature-window end:

| Prefix | Time range | Included summaries |
|---|---|---|
| `last60min_` | Last 60 minutes before window end | mean, min, max, std, count, last |
| `last30min_` | Last 30 minutes before window end | mean, min, max, std, count, last |

The 60-minute and 30-minute features are used for comparison. The 60-minute window may be more stable, while the 30-minute window may capture sharper acute changes but can be sparse.

### 6.6 Vital availability features

The notebook adds simple missingness and availability features.

| Feature | Meaning |
|---|---|
| `available_vital_type_count` | Number of vital-sign types available in the window |
| `missing_vital_type_count` | Number of vital-sign types missing in the window |
| `core_available_vital_count` | Number of available core vitals among HR, RR, SpO2, temperature, SBP, DBP |
| `core_missing_vital_count` | Number of missing core vitals |
| `all_core_vitals_available_flag` | 1 if all core vitals are available |

---

## 7. Main Renewed Dataset: Case-Level Ward-Episode Outcome Prediction

### 7.1 Feature-window designs

The case-level track creates one row per case for each feature-window design.

| Feature-window name | Meaning | Interpretation |
|---|---|---|
| `early_6h` | First 6 hours after W36 admission | Early, deployable-style prediction |
| `early_12h` | First 12 hours after W36 admission | Early, deployable-style prediction with more vital-sign context |
| `early_24h` | First 24 hours after W36 admission | Early-ish, but excludes short W36 stays shorter than 24 hours |
| `full_stay` | All available W36 stay information | Retrospective upper-bound signal analysis |
| `last_3h` | Last 3 hours before Movement End | Retrospective acute-signal analysis |
| `last_6h` | Last 6 hours before Movement End | Retrospective acute-signal analysis |
| `last_12h` | Last 12 hours before Movement End | Retrospective acute-signal analysis |
| `last_24h` | Last 24 hours before Movement End | Retrospective acute-signal analysis; excludes short stays |

The early windows are the cleanest for real early-warning interpretation. Full-stay and last-window datasets are useful for understanding signal strength, but they should not be presented as deployable early-warning models.

### 7.2 Case-level targets

The case-level output contains these target columns:

| Target | Meaning |
|---|---|
| `case_y_hd_icu` | Case eventually had HD/ICU transfer |
| `case_y_death_1m` | Case had 1-month death indicator |
| `case_y_death_3m` | Case had 3-month death indicator |
| `case_y_death` | Any available death indicator |
| `case_y_hd_icu_or_death` | Composite of HD/ICU transfer or death |
| `case_y_o2` | Optional case-level O2 increase outcome |
| `case_y_iv` | Optional case-level IV increase outcome |
| `case_y_met` | Optional case-level MET activation outcome |
| `case_y_any_deterioration` | Any case-level deterioration outcome |

Each target has a matching `eligible_case_y_*` column. If an exact event time is known and the event already happened before the feature-window end, the row is not considered a valid prediction row for that target.

### 7.3 Case-level dataset summary

The generated case-level outputs contain the following case counts and outcome counts across all splits.

| Feature window | Rows/cases | HD/ICU positives | Death positives | HD/ICU or death positives | HD/ICU or death rate | Any deterioration positives | Any deterioration rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early_6h | 441 | 69 | 88 | 140 | 31.7% | 190 | 43.1% |
| early_12h | 440 | 68 | 87 | 139 | 31.6% | 189 | 43.0% |
| early_24h | 422 | 64 | 86 | 134 | 31.8% | 184 | 43.6% |
| full_stay | 441 | 69 | 88 | 140 | 31.7% | 190 | 43.1% |
| last_3h | 441 | 69 | 88 | 140 | 31.7% | 190 | 43.1% |
| last_6h | 441 | 69 | 88 | 140 | 31.7% | 190 | 43.1% |
| last_12h | 440 | 68 | 87 | 139 | 31.6% | 189 | 43.0% |
| last_24h | 422 | 64 | 86 | 134 | 31.8% | 184 | 43.6% |

Interpretation:

- The early 6-hour, full-stay, last 3-hour, and last 6-hour datasets retain all 441 cases.
- The early 12-hour and last 12-hour datasets retain 440 cases.
- The early 24-hour and last 24-hour datasets retain 422 cases because some W36 episodes are shorter than 24 hours.
- The HD/ICU-or-death composite has about 31.6% to 31.8% positives across feature-window designs, which is much more usable than attempting HD/ICU or death as short-horizon in-ward timed labels.

### 7.4 Case-level output files

Folder pattern:

```text
case_level_ward_outcome/{feature_window_name}/
```

File pattern:

```text
train_{feature_window_name}.csv
validation_{feature_window_name}.csv
test_{feature_window_name}.csv
all_splits_{feature_window_name}.csv
```

Summary files:

| File | Purpose |
|---|---|
| `case_level_dataset_summary.csv` | Row counts, case counts, positive case counts, and positive rates by feature window and split |
| `case_level_missingness_summary.csv` | Missingness rates for generated features |
| `case_level_split_leakage_check.csv` | Confirms train/validation/test case split overlap is zero |
| `all_case_level_feature_windows_stacked.csv` | Stacked audit file across all feature-window designs |

---

## 8. Optional Sliding-Window Short-Horizon Dataset

The sliding-window track is kept for IV/O2/MET-style short-horizon prediction and for continuity with earlier work.

### 8.1 Window settings

| Setting | Values |
|---|---|
| Observation windows `m_hours` | 4, 6, 12, 24 |
| Prediction horizons `h_hours` | 4, 6, 12 |
| Sliding step | 1 hour |

For one case:

```text
window_start = W36 Start DT + k hours
window_end   = window_start + m_hours
horizon_end  = window_end + h_hours
```

A candidate window is first created only if:

```text
window_end <= Movement End DT
```

It is saved into a final labelled dataset only if the full future horizon is observable:

```text
horizon_end <= Movement End DT
```

### 8.2 Sliding-window target columns

The final sliding-window datasets contain six target columns.

| Target | Meaning |
|---|---|
| `y_any` | Any new not-yet-occurred deterioration event inside the prediction horizon |
| `y_hd_icu` | New HD/ICU transfer inside the prediction horizon |
| `y_o2` | New confirmed O2 increase inside the prediction horizon |
| `y_iv` | New IV increase inside the prediction horizon |
| `y_met` | New MET activation inside the prediction horizon |
| `y_death` | New death event inside the prediction horizon |

Each target has a matching eligibility column:

```text
eligible_y_any
eligible_y_hd_icu
eligible_y_o2
eligible_y_iv
eligible_y_met
eligible_y_death
```

During modelling, eligibility columns should be used for filtering valid rows for the selected target. They should not be used as predictors.

### 8.3 Positive, negative, ineligible, and censored windows

For an event-specific target such as `y_o2`, the notebook applies this rule:

| Situation | Result |
|---|---|
| Full prediction horizon is not observable before Movement End | Window is censored/discarded |
| Target event already happened before `window_end` | `eligible_y_* = 0`, `y_* = missing` |
| Target event happens in `[window_end, horizon_end)` | `eligible_y_* = 1`, `y_* = 1` |
| Target event does not happen in `[window_end, horizon_end)` | `eligible_y_* = 1`, `y_* = 0` |

A window can be ineligible for one target but still valid for another. For example, a row can be ineligible for `y_o2` if O2 already happened, but still eligible for `y_iv`, `y_met`, `y_hd_icu`, or `y_death`.

### 8.4 Cascade modelling design

The sliding-window dataset is cascade-capable.

Earlier deterioration events are kept as valid history if they happened before `window_end`. For example:

```text
prior_o2 = 1
time_since_prior_o2_hours = hours since previous O2 increase
```

These features are leakage-safe because they are known at prediction time.

---

## 9. Mentor-Review Areas to Address

This section summarizes the audit checks requested before moving into modelling.

### 9.1 Area 1: Secondary diagnosis hit rates

Requested check:

> Review the audit outputs before modelling. Report the fraction of cases with zero secondary diagnosis flags after cleaning, and whether any groups are too rare to be informative.

Result:

- Total retained cases checked: **441**
- Cases with zero secondary-diagnosis group flags after cleaning: **46**
- Fraction with zero secondary-diagnosis group flags: **10.4%**

This means most cases receive at least one grouped secondary-diagnosis flag, but about one in ten cases has no keyword-group match. These cases may still have raw secondary diagnoses; they simply did not match the current grouped keyword definitions.

Lowest-count diagnosis groups:

| Diagnosis group | Cases | Case rate | Any deterioration rate |
| --- | --- | --- | --- |
| secondary_dx_liver_cirrhosis_failure | 13 | 2.9% | 69.2% |
| secondary_dx_thrombosis_embolism | 13 | 2.9% | 53.8% |
| secondary_dx_metastatic_cancer | 22 | 5.0% | 95.5% |
| secondary_dx_gi_acute | 23 | 5.2% | 73.9% |
| secondary_dx_respiratory | 23 | 5.2% | 82.6% |
| secondary_dx_hypotension_shock | 27 | 6.1% | 81.5% |

Interpretation:

- The notebook's rare-group threshold was **9 cases**. No group fell below that threshold.
- However, the lowest-count groups are still small in practical modelling terms. For example, `secondary_dx_thrombosis_embolism` has 13 cases, about 3.0% of the cohort.
- These groups should not be treated as strong standalone predictors. They can be retained for tree-based models or regularized models, but their coefficients or feature importance should be interpreted cautiously.
- If model instability appears later, rare groups can be combined, excluded, or treated as exploratory.

### 9.2 Area 2: O2 increase fallback logic

Requested check:

> Check how many cases are using the fallback path, and whether their outcome rates differ meaningfully from confirmed cases.

Result:

| Audit item | Value | Interpretation |
|---|---:|---|
| Cases with O2 timestamp | 20 | Cases with a non-missing O2 timestamp field |
| Confirmed O2 cases | 20 | Cases with `o2_increase_confirmed == 1` |
| Timestamp but not confirmed cases | 0 | Potentially noisy timestamp-only cases |
| Fallback cases used as positive | 0 | Cases labelled positive by timestamp fallback |

In the current generated dataset, **no case used the fallback path**. The confirmation column exists, and all 20 cases with O2 timestamps are confirmed O2 cases. Therefore, there is no evidence that timestamp-only unconfirmed O2 records introduced noise in this run.

Outcome-rate comparison by O2 evidence group:

| O2 evidence group | Cases | Any deterioration rate | HD/ICU rate | IV rate | MET rate | Death rate |
| --- | --- | --- | --- | --- | --- | --- |
| confirmed_o2_with_timestamp | 20 | 100.0% | 25.0% | 50.0% | 10.0% | 55.0% |
| not_confirmed_no_timestamp | 421 | 40.4% | 15.2% | 14.5% | 0.5% | 18.3% |

Interpretation:

- Confirmed O2 cases are a small group: 20 cases out of 441.
- Confirmed O2 cases have high co-occurrence with other adverse outcomes in this cohort, including IV increase and death.
- O2 may still be a weak event-specific sliding-window target because it has few positive cases, not because the fallback logic added noisy timestamp-only positives.

### 9.3 Area 3: Horizon censoring by `m/h` combination

Requested check:

> Summarize censoring rates across the 12 sliding-window dataset combinations. Long observation windows paired with longer prediction horizons could distort positive rates if censoring is high.

Horizon-censoring summary across all splits:

| m hours | h hours | Windows before filter | Windows after filter | Removed windows | Removed rate |
| --- | --- | --- | --- | --- | --- |
| 4 | 4 | 74971 | 73207 | 1764 | 2.4% |
| 4 | 6 | 74971 | 72325 | 2646 | 3.5% |
| 4 | 12 | 74971 | 69687 | 5284 | 7.0% |
| 6 | 4 | 74089 | 72325 | 1764 | 2.4% |
| 6 | 6 | 74089 | 71445 | 2644 | 3.6% |
| 6 | 12 | 74089 | 68812 | 5277 | 7.1% |
| 12 | 4 | 71445 | 69687 | 1758 | 2.5% |
| 12 | 6 | 71445 | 68812 | 2633 | 3.7% |
| 12 | 12 | 71445 | 66222 | 5223 | 7.3% |
| 24 | 4 | 66222 | 64552 | 1670 | 2.5% |
| 24 | 6 | 66222 | 63729 | 2493 | 3.8% |
| 24 | 12 | 66222 | 61283 | 4939 | 7.5% |

Interpretation:

- Censoring is present but modest across all 12 combinations.
- Removed rates range from **2.4%** to **7.5%**.
- As expected, longer prediction horizons remove more late-stay windows because more future follow-up is required before `Movement End DT`.
- The highest censoring rate is for `m=24, h=12`, at **7.5%**.
- These rates should be reported, but they are not high enough by themselves to invalidate the sliding-window datasets.

### 9.4 Area 4: `eligible_y_any` logic validation

Requested check:

> Verify that a patient who already had one event can still be eligible for `y_any` if other events have not yet occurred.

Validation result from the patient trace audit:

| Check | Value |
|---|---:|
| Rows checked | 710 |
| `eligible_y_any` mismatch rows | 0 |
| Mixed-eligibility rows where `y_any` remained eligible | 660 |
| Validation passed | Yes |

Interpretation:

- The validation found **0 mismatches**.
- The notebook correctly allows mixed eligibility. For example, if IV has already occurred, the row becomes ineligible for `y_iv`, but it can still remain eligible for `y_any` if O2, MET, HD/ICU, or death has not yet occurred.
- This is the correct behaviour for a composite target asking whether **any new not-yet-occurred deterioration event** happens in the prediction horizon.
- No logic gap was found before modelling.

### 9.5 Area 5: Recommended starting configuration

For the sliding-window IV/O2/MET track, a reasonable starting configuration is:

```text
m = 6 hours, h = 12 hours
```

Reasoning:

- `m=6` gives more physiological context than `m=4` while still staying close to the current clinical state.
- `h=12` gives a clinically useful warning horizon and captures more positive event windows than shorter horizons.
- Horizon censoring for `m=6, h=12` is **7.1%**, which is acceptable and similar to other 12-hour horizon settings.
- In the generated summaries, `m=6, h=12` retained 68,812 windows and had 69 `y_any` positive cases, 58 `y_iv` positive cases, 19 `y_o2` positive cases, and 4 `y_met` positive cases across all splits.
- HD/ICU and death still have zero short-horizon sliding-window positives, so they should not be the main targets for this track.

For the case-level HD/ICU/death track, a reasonable starting feature window is:

```text
early_12h
```

Reasoning:

- `early_12h` remains close to deployable early prediction.
- It provides more vital-sign information than `early_6h`.
- It loses only one case compared with `early_6h` in the generated summaries.
- It has 139 HD/ICU-or-death positive cases across all splits, about 31.6% of the retained rows.

Suggested first modelling plan:

1. Start HD/ICU/death modelling with `early_12h` case-level data.
2. Use `early_6h` as a more aggressive early-warning comparison.
3. Use `early_24h`, `full_stay`, and last-window designs as sensitivity or signal-analysis experiments.
4. Start IV/O2/MET sliding-window modelling with `m=6, h=12`.

---

## 10. One Actual Patient Trace Through the Sliding-Window Pipeline

The mentor requested one actual patient trace from W36 start to Movement End.

The generated audit trace uses:

| Item | Value |
|---|---|
| Case ID | `4366de1ae1` |
| Observation window | `m = 6 hours` |
| Prediction horizon | `h = 12 hours` |
| Candidate windows generated | 722 |
| Labelled windows after horizon filter | 710 |
| Censored windows | 12 |

### 10.1 Trace result summary

| Target | Positive windows | Ineligible windows | Censored windows | Main interpretation |
|---|---:|---:|---:|---|
| `y_any` | 36 | 0 | 12 | Any new event occurred in 36 labelled windows |
| `y_hd_icu` | 0 | 0 | 12 | No HD/ICU event occurred inside labelled horizons for this case |
| `y_o2` | 12 | 0 | 12 | O2 increase appeared in 12 future horizons |
| `y_iv` | 12 | 498 | 12 | IV became ineligible after IV had already occurred |
| `y_met` | 12 | 660 | 12 | MET became ineligible after MET had already occurred |
| `y_death` | 0 | 0 | 12 | No death event occurred inside labelled horizons for this case |

This trace confirms the intended pipeline behaviour:

- windows are generated hourly from W36 start until the observation window can no longer fit before Movement End;
- late windows are censored when the full 12-hour prediction horizon is not observable;
- event-specific targets become ineligible after that event has already happened;
- the composite `y_any` can remain eligible even when one event-specific target is already ineligible;
- positive windows are assigned only when an event occurs inside `[window_end, horizon_end)`.

### 10.2 Trace timeline figure

The notebook saves a timeline visualization for the selected case:

```text
mentor_review_audit/patient_trace_timeline_case_4366de1ae1_m6_h12.png
```

Local copy in this folder:

```text
patient_trace_timeline_case_4366de1ae1_m6_h12.png
```

![Patient trace timeline](patient_trace_timeline_case_4366de1ae1_m6_h12.png)

The CSV trace remains the formal audit output. The figure is mainly for discussion because it visually shows observation windows, prediction horizons, event times, and windows censored by Movement End.

---

## 11. Train / Validation / Test Split

The split is performed at case level, not window level.

This prevents windows from the same case appearing in both training and testing data.

The split ratio is:

```text
70% train
15% validation
15% test
```

The split is stratified by:

```python
label_any_deterioration
```

The notebook uses:

```python
random_state = 42
```

After case-level split assignment, every generated case-level row or sliding-window row inherits the split of its parent case.

The case-level split leakage check found zero overlap between train, validation, and test cases for all eight case-level feature-window designs.

---

## 12. Feature Groups for Modelling

A direct way to understand the final modelling features is to group them by how they are created and whether they should be used as predictors.

| Feature group | Examples | Predictor? | Interpretation |
|---|---|---:|---|
| Metadata | `case_no_deid_clean`, `data_split`, `feature_window_name`, `m_hours`, `h_hours`, `window_start`, `window_end`, `horizon_end` | No | Identifies case, split, and row design |
| Targets and eligibility | `case_y_hd_icu`, `y_o2`, `eligible_y_o2` | No | Labels and filtering indicators |
| Demographics / patient context | `Age`, `Gender`, `Race`, `Admit Type Description`, `data_batch` | Yes | Case-level baseline context |
| Time context | `hours_since_w36_start`, feature-window duration variables | Usually optional | Useful in sliding-window models; interpret carefully in case-level models |
| Full-window vital features | `HR_mean`, `RR_max`, `SpO2_slope`, `temperature_time_since_last` | Yes | Overall physiological status across the selected observation window |
| Baseline/recent vital features | `baseline_HR_mean`, `recent_HR_mean` | Yes | Earlier-half versus later-half status inside the same window |
| Baseline/recent contrast features | `recent_minus_baseline_HR_mean`, `recent_minus_baseline_RR_count` | Yes | Recent change compared with the patient's own earlier status |
| Short-recent 60-minute features | `last60min_HR_last`, `last60min_SpO2_min` | Optional | Near-term physiology before prediction time |
| Short-recent 30-minute features | `last30min_RR_max`, `last30min_temperature_mean` | Optional | Very immediate physiology before prediction time |
| Vital availability features | `available_vital_type_count`, `all_core_vitals_available_flag` | Yes | Missingness and measurement availability signals |
| Secondary diagnosis features | `secondary_dx_respiratory`, `secondary_dx_cardiac`, `secondary_diagnosis_count` | Optional | Grouped secondary-diagnosis risk markers |
| Known intermediate event features | `known_o2_before_feature_end`, `time_since_known_met_hours` | Optional for case-level models | Earlier O2/IV/MET history before case-level feature-window end |
| Prior-event cascade features | `prior_o2`, `time_since_prior_o2_hours`, `prior_event_count` | Yes for cascade sliding-window models | Deterioration history already known before prediction time |

The word “optional” means the notebook creates these features, but the modelling stage should compare feature sets rather than assume every feature group is always beneficial.

---

## 13. How to Use the Final CSVs for Modelling

### 13.1 Case-level HD/ICU/death modelling

A typical workflow is:

1. Choose one feature-window folder, such as `case_level_ward_outcome/early_12h`.
2. Load `train_early_12h.csv`, `validation_early_12h.csv`, and `test_early_12h.csv`.
3. Choose one target, such as `case_y_hd_icu_or_death`.
4. Filter to eligible rows for that target using the matching `eligible_case_y_*` column, or use `prepare_case_level_model_data(...)` from the notebook.
5. Choose a feature set, such as `patient_dx_vital_with_baseline_recent` or `full_case_level_feature_set`.
6. Train and evaluate the model.

Important reminders:

- Do not use target columns as predictors.
- Do not use eligibility columns as predictors.
- Do not use event timestamps as predictors.
- Treat full-stay and last-window results as retrospective or upper-bound signal analysis.

### 13.2 Sliding-window IV/O2/MET modelling

A typical workflow is:

1. Choose one `m_hours` and one `h_hours` dataset folder, such as `m6_h12`.
2. Load the corresponding train, validation, and test CSV files.
3. Choose one target, such as `y_any`, `y_iv`, or `y_o2`.
4. Filter to eligible rows for that target, or use `prepare_model_data(...)` from the notebook.
5. Select one feature set, such as `baseline_patient_vital` or `cascade_final60_dx_last60`.
6. Train and evaluate the model.

Important reminders:

- Use the matching `eligible_y_*` column when training event-specific models.
- For first-deterioration modelling, filter to rows where `prior_event_count == 0` and `any_prior_deterioration_flag == 0`.
- HD/ICU and death are not useful event-specific short-horizon targets in the current generated sliding-window summaries because they have zero positive windows.

---

## 14. Output Files

### 14.1 Case-level ward-outcome outputs

```text
/content/drive/My Drive/Dataset/non-sequential/case_level_ward_outcome/
```

Important files:

| File or folder | Purpose |
|---|---|
| `early_6h/`, `early_12h/`, `early_24h/` | Early case-level feature-window datasets |
| `full_stay/` | Full W36 stay retrospective dataset |
| `last_3h/`, `last_6h/`, `last_12h/`, `last_24h/` | Last-window retrospective acute-signal datasets |
| `case_level_dataset_summary.csv` | Case-level row counts and positive counts |
| `case_level_missingness_summary.csv` | Case-level feature missingness summary |
| `case_level_split_leakage_check.csv` | Case split overlap check |
| `all_case_level_feature_windows_stacked.csv` | Stacked case-level audit file |

### 14.2 Sliding-window short-horizon outputs

```text
/content/drive/My Drive/Dataset/non-sequential/sliding_window_short_horizon/
```

Important files:

| File or folder | Purpose |
|---|---|
| `m4_h4/`, `m4_h6/`, ..., `m24_h12/` | Sliding-window train/validation/test CSVs for each `m/h` combination |
| `feature_bases_internal/` | Reusable feature bases before horizon labels are added |
| `case_split_summary.csv` | Case-level split size and timed-positive counts |
| `dataset_generation_summary.csv` | Sliding-window row counts and positive-window/case counts |
| `missingness_summary.csv` | Sliding-window missingness summary |
| `horizon_censoring_summary.csv` | Windows removed because the full future horizon was not observable |

### 14.3 Mentor-review audit outputs

```text
/content/drive/My Drive/Dataset/non-sequential/mentor_review_audit/
```

Important files:

| File | Purpose |
|---|---|
| `secondary_dx_zero_group_flag_summary.csv` | Fraction of cases with zero secondary-diagnosis group flags |
| `secondary_dx_rare_group_review.csv` | Diagnosis group size and rare-group review |
| `o2_increase_label_policy_summary.csv` | O2 confirmation/fallback policy counts |
| `o2_increase_evidence_group_summary.csv` | Outcome rates by O2 evidence group |
| `eligible_y_any_validation_trace_case_*.csv` | `eligible_y_any` validation summary |
| `eligible_y_any_mixed_examples_trace_case_*.csv` | Examples of mixed eligibility where `y_any` remains eligible |
| `patient_trace_case_*.csv` | Readable patient trace across all candidate windows |
| `patient_trace_actual_pipeline_labelled_case_*.csv` | Selected case after the actual labelling function |
| `patient_trace_timeline_case_*.png` | Timeline visualization for discussion |

---

## 15. Known Limitations and Future Improvements

1. **HD/ICU transfer and death are not suitable as short-horizon in-ward window targets in the current data.**

   This is the main reason the renewed case-level dataset was created. HD/ICU transfer often occurs at Movement End, and death usually happens after ICU transfer or outside W36. The case-level design is more appropriate for these outcomes.

2. **Raw outcome prevalence does not always translate into usable timed labels.**

   Some raw case-level outcomes may lack reliable timestamps. These outcomes can be counted in case-level summaries but cannot always be converted into short-horizon window labels.

3. **O2 has few confirmed positive cases.**

   The audit found no noisy fallback-positive cases in this run, but only 20 confirmed O2 cases exist in the retained cohort. O2-specific modelling should therefore be treated as exploratory unless more positives are available.

4. **MET is very rare as a sliding-window event-specific target.**

   The sliding-window summary shows only 4 MET positive cases across generated combinations. MET may contribute to `y_any`, but standalone MET modelling is likely unstable.

5. **Secondary diagnosis grouping is keyword-based.**

   Grouped features are interpretable, but keyword matching may miss clinically relevant diagnoses or include imperfect matches. Rare groups should be interpreted cautiously.

6. **Charted and sensor vital signs are merged into one clinical stream.**

   This improves interpretability, but charted and sensor measurements may differ in frequency, noise level, and clinical meaning. Future work may compare merged versus source-separated vital streams.

---

## 16. Reproducibility Notes

Main dataset-building notebook:

```text
dataset_nonsequential_final.ipynb
```

Key implementation choices:

- Output root is `/content/drive/My Drive/Dataset/non-sequential`.
- Case-level ward-outcome datasets are generated by default.
- Optional sliding-window short-horizon datasets are not regenerated by default.
- Train/validation/test split is done by case, not by row/window.
- Split ratio is 70% train, 15% validation, and 15% test.
- Primary diagnosis is excluded from model features.
- Secondary diagnosis is grouped into clinically meaningful dummy variables.
- Charted and sensor HR/RR/SpO2/body temperature are unified into clinical vital streams.
- Sensor skin temperature remains separate from body temperature.
- Vital features include summary, trend, missingness, baseline/recent, and short-recent features.
- Event timestamps are used for label construction, eligibility, and audit checks, then removed from final model-ready CSVs.
- Eligibility columns are used for target-specific filtering, not as model predictors.
- Case-level early windows are most suitable for deployable HD/ICU/death-style prediction.
- Sliding-window datasets are most suitable for IV/O2/MET-style timed short-horizon prediction.

---

## 17. Reader Confusion Checklist

| Confusion question | Clarification in this README |
|---|---|
| Why is there a new dataset for HD/ICU/death? | Explained that HD/ICU transfer often occurs at Movement End and death mostly occurs after ICU transfer or outside W36 |
| What is one row in the final dataset? | Separated case-level rows from sliding-window rows |
| Are the notebook inputs raw files or prepared merged files? | Separated original raw data structure from prepared files loaded by the final notebook |
| Why are HD/ICU/death poor sliding-window targets? | Explained the mismatch between outcome timing and short in-ward horizons |
| What is the difference between `case_y_*` and `y_*`? | Explained case-level targets versus window-level targets |
| Why does raw prevalence differ from positive windows? | Explained timed events, eligibility, and horizon observability |
| Why do some target values become missing? | Explained target-specific ineligibility after an event already happened |
| Are prior events leakage? | Explained that prior events before prediction time are valid cascade features |
| What should not be used as predictors? | Listed metadata, target, eligibility, and event-time columns as non-predictors |
| What did the mentor-review audits find? | Added separate sections for all five requested areas plus one-patient trace |
| Which modelling setup should start first? | Recommended `early_12h` for case-level HD/ICU/death and `m=6, h=12` for sliding-window IV/O2/MET |
