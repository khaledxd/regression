# 🏠 House Price Prediction

A machine learning project predicting residential property prices using a large real-world dataset of 33,000+ Australian house sales. The project covers data cleaning, missing value handling, feature engineering, and benchmarking of multiple regression models.

---

## 📊 Dataset

- **Source:** Australian residential property sales records
- **Size:** 33,656 properties × 19 features
- **Price range:** $51,000 – $2,440,000 (mean: ~$637,000)
- **Suburbs covered:** 321 unique suburbs

**Key features:**

| Feature | Description |
|---|---|
| `PRICE` | Sale price (target variable) |
| `BEDROOMS` / `BATHROOMS` / `GARAGE` | Property room counts |
| `LAND_AREA` / `FLOOR_AREA` | Property size metrics |
| `BUILD_YEAR` | Year the property was built |
| `CBD_DIST` | Distance to Central Business District |
| `NEAREST_STN_DIST` | Distance to nearest train station |
| `NEAREST_SCH_DIST` | Distance to nearest school |
| `SUBURB` | Suburb name (321 unique values) |
| `DATE_SOLD` | Month and year of sale |
| `LATITUDE` / `LONGITUDE` | Geographic coordinates |

**Missing values handled:**
- `GARAGE` — 2,478 missing (7.4%) → imputed with column mean
- `BUILD_YEAR` — 3,155 missing (9.4%) → imputed with most frequent value
- `NEAREST_SCH_RANK` — 10,952 missing (32.5%) → dropped due to high missingness rate

---

## 🔍 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Shape, dtypes, descriptive statistics inspection
- Missing value audit across all 19 columns
- Correlation heatmap — identified positive relationships between Bathrooms/Land Area and Price; noted negative relationship with Build Year and School Rank
- Duplicate check

### 2. Missing Value Handling
- `BUILD_YEAR` → most frequent imputation (`SimpleImputer`)
- `GARAGE` → mean imputation
- `NEAREST_SCH_RANK` → dropped (32.5% missing)

### 3. Feature Engineering
- Parsed `DATE_SOLD` (including stripping `\r` characters from raw data) → extracted sale `year`
- `Home_Age` = 2024 − BUILD_YEAR (property age in years)
- `Rooms` = BEDROOMS + BATHROOMS + 2 (composite room count)
- `Price_Per_Area` = PRICE / FLOOR_AREA (price efficiency metric)
- Dropped high-cardinality text columns: `ADDRESS`, `POSTCODE`, `NEAREST_STN`, `NEAREST_SCH`

### 4. Visualization
- 2×2 subplot grid:
  - Distribution of property prices (histogram + KDE)
  - Bedrooms vs Price (scatter plot)
  - Price vs Land Area (scatter plot)
  - Garage Spaces vs Price (box plot)

### 5. Data Transformation
- One-Hot Encoding applied to `SUBURB` (321 suburbs → binary columns)
- StandardScaler applied to all features before modeling

### 6. Model Benchmarking

Three regression models compared in a single loop:

| Model | Metrics Reported |
|---|---|
| Random Forest Regressor | MAE, MSLE, R² |
| Decision Tree Regressor | MAE, MSLE, R² |
| Support Vector Regressor | MAE, MSLE, R² |

---

## 📁 Project Structure

```
house-prices-prediction/
│
├── house-prices.ipynb     # Main notebook
├── house-prices.csv       # Dataset
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/house-prices-prediction.git
```

2. Install the required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the notebook in Jupyter
```bash
jupyter notebook house-prices.ipynb
```

4. Make sure `house-prices.csv` is in the same folder as the notebook, then run all cells

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | Imputation, encoding, scaling, regression models, metrics |

---

## 💡 Key Findings

- **Land area and number of bathrooms** show the strongest positive correlation with sale price
- **Distance to CBD** is an important location factor — properties closer to the city center command higher prices
- **Home age** derived from `BUILD_YEAR` captures depreciation effects on price
- The dataset covers a wide price range ($51k to $2.44M), reflecting diverse suburb profiles across 321 locations
- Random Forest Regressor achieved the strongest performance among the three models benchmarked

---

## 👤 Author

**Khaled Abdulaziz**
