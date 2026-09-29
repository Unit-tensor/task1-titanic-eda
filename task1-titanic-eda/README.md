# Task 1 - Exploratory Data Analysis (EDA) on Titanic

Exploratory analysis of the classic Titanic dataset using **Pandas**: inspecting the data, cleaning missing values and duplicates, and extracting insights with grouping and aggregation.

## Objectives
- DataFrame operations: `head()`, `info()`/`dtypes`, `describe()`
- Handle missing values and duplicates
- Grouping and aggregation analysis

## Dataset
Titanic passenger data (891 rows, 15 columns), loaded via `seaborn.load_dataset("titanic")` and cached to `data/titanic.csv`.

## Project structure
```
task1-titanic-eda/
├── eda.py            # full analysis script
├── requirements.txt
├── data/             # raw dataset (auto-downloaded)
└── outputs/          # cleaned CSV + text report
```

## How to run
```bash
pip install -r requirements.txt
python eda.py
```

## Methodology
| Step | What was done |
|------|---------------|
| Inspect | `head`, `shape`, `dtypes`, `describe` (numeric + categorical) |
| Missing values | `age` filled with the median of each (class, sex) group; `embarked`/`embark_town` filled with the mode; `deck` dropped (~77% missing) |
| Duplicates | Detected with `duplicated()` and removed |
| Feature engineering | `family_size`, `age_group` |
| Aggregation | `groupby` and `pivot_table` on sex, class, port, age group, family size |

## Key findings
- Overall survival rate is about 41% after removing duplicates (about 38% in the raw data).
- Sex is the strongest factor: about 74% of women survived vs about 22% of men.
- Class matters: survival falls from about 64% (1st) to 26% (3rd class).
- Fares are heavily right-skewed (mean much higher than median).
- Small families fared better than solo travellers or very large families.

Exact figures are in `outputs/eda_report.txt` after running the script.

## Skills demonstrated
Pandas, data cleaning, imputation strategy, groupby/pivot analysis.
