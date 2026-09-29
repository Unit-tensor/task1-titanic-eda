"""Task 1 - Exploratory Data Analysis (EDA) on the Titanic dataset."""
from pathlib import Path
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "titanic.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
DATA.parent.mkdir(exist_ok=True)

report = []
def log(title, obj=""):
    text = f"\n=== {title} ===\n{obj}"
    print(text)
    report.append(text)

# 1. Load ---------------------------------------------------------------
if DATA.exists():
    df = pd.read_csv(DATA)
else:
    df = sns.load_dataset("titanic")
    df.to_csv(DATA, index=False)

# 2. Basic DataFrame operations -----------------------------------------
log("HEAD", df.head().to_string())
log("SHAPE (rows, cols)", df.shape)
log("DTYPES", df.dtypes.to_string())
log("DESCRIBE (numeric)", df.describe().round(2).to_string())
log("DESCRIBE (categorical)", df.describe(include=["object", "category", "bool"]).to_string())

# 3. Missing values -----------------------------------------------------
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
log("MISSING VALUES", pd.DataFrame({"count": missing, "percent": missing_pct})
    .query("count > 0").to_string())

# Age: fill with the median age of each (pclass, sex) group
df["age"] = df["age"].fillna(df.groupby(["pclass", "sex"])["age"].transform("median"))
# Embarked / embark_town: fill with the mode (only 2 rows missing)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
# Deck: ~77% missing -> too sparse to impute, so drop it
df = df.drop(columns=["deck"])
log("MISSING AFTER CLEANING", int(df.isna().sum().sum()))

# 4. Duplicates ---------------------------------------------------------
n_dup = int(df.duplicated().sum())
log("DUPLICATE ROWS FOUND", n_dup)
df = df.drop_duplicates().reset_index(drop=True)
log("SHAPE AFTER DROPPING DUPLICATES", df.shape)

# 5. Feature engineering -------------------------------------------------
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

# 6. Grouping & aggregation --------------------------------------------
log("OVERALL SURVIVAL RATE", f"{df['survived'].mean():.2%}")
for col in ["sex", "pclass", "embarked", "age_group", "alone"]:
    g = (df.groupby(col, observed=True)["survived"]
           .agg(passengers="count", survivors="sum", survival_rate="mean").round(3))
    log(f"SURVIVAL BY {col.upper()}", g.to_string())

log("SURVIVAL BY SEX x CLASS (pivot)",
    df.pivot_table(values="survived", index="sex", columns="pclass", aggfunc="mean").round(3).to_string())
log("AVERAGE FARE & AGE BY CLASS",
    df.groupby("pclass").agg(avg_fare=("fare", "mean"), median_fare=("fare", "median"),
                             avg_age=("age", "mean")).round(2).to_string())
log("FAMILY SIZE vs SURVIVAL",
    df.groupby("family_size")["survived"].agg(["count", "mean"]).round(3).to_string())

# 7. Key insights --------------------------------------------------------
sex = df.groupby("sex")["survived"].mean()
cls = df.groupby("pclass")["survived"].mean()
insights = f"""
1. Only {df['survived'].mean():.1%} of passengers survived.
2. Women survived at {sex['female']:.1%} vs men at {sex['male']:.1%} ("women and children first").
3. 1st class survival was {cls[1]:.1%} vs {cls[3]:.1%} in 3rd class - wealth mattered.
4. Fare is strongly right-skewed: mean {df['fare'].mean():.1f} vs median {df['fare'].median():.1f}.
5. Passengers travelling alone survived less than small families (see FAMILY SIZE table).
"""
log("KEY INSIGHTS", insights)

df.to_csv(OUT / "titanic_cleaned.csv", index=False)
(OUT / "eda_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nSaved outputs/titanic_cleaned.csv and outputs/eda_report.txt")
