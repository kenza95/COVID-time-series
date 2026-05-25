# =================================================================================================================================================
# TIME SERIES ANALYSIS & FORECASTING (COVID CONTEXT)

# Dataset: Banque de France FIBEN dataset provides firms "value added" from 2003 to 2019.
# Firms "value added" is computed by Banque de France using firms balance sheets when they become available (N+1). 
# Firms are classified within 5 economic sectors. 
# Although the available dataset is at the firm level (with multiple firms per sector and year), we aggregate it to the sector–year level to construct sectoral time series.

# The objective is to estimate the historical relationship between:
# Banque de France (BdF) sectoral value added
# INSEE sectoral value added

# This relationship is then used to forecast BdF sectoral value added for 2020, a year for which BdF data is not available yet but INSEE estimates exist.
# We estimate separate time series regressions for each of the 5 economic sectors using the aggregated sectoral time series.
# =================================================================================================================================================

#To execute, run this in zsh terminal: python3 -m pip install numpy pandas matplotlib statsmodels
#Then: python3 main.py         

# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
import os

# =================================================================================================================================================
# PDF FORMATTING FUNCTION FOR PRODUCING FINAL OUTPUTS AS PDF REPORTS PER SECTOR
# =================================================================================================================================================

def format_pdf_page():
    """Creates a clean PDF page"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    return fig, ax

"""
# =================================================================================================================================================
# LOAD AND CLEAN RAW DATA (LOCAL SETUP)
# The raw FIBEN dataset is stored locally and is too large to include in the Git repository.
# This section explains how the final "timeseries" dataset is constructed from raw data.
# It is intended for local execution using absolute file paths.
# If running from the Git repository, skip to the "LOAD CLEAN DATA" section below.
# =================================================================================================================================================

# The firms dataset contains 5234974 rows and 26 columns.
FIBEN_dataset = pd.read_stata("/Users/kenzahaouche/COVID-time-series/FIBEN_2003_2019.dta")
print("Shape:", FIBEN_dataset.shape)
print("Columns:", FIBEN_dataset.columns)
print(FIBEN_dataset.head())
print(FIBEN_dataset.info())

# We delete columns that are not needed for the analysis.  
FIBEN_dataset = FIBEN_dataset.iloc[:, :7]
# We drop value added at firm-level (va_BdFn), we keep the sectoral value added(sectoral_VA_BdF).
FIBEN_dataset = FIBEN_dataset.drop(columns=["va_BdFn"]) 

# We rename columns for clarity. INSEE sectoral value added was previously merged with the FIBEN dataset based on year and sector identifiers.
FIBEN_dataset.columns = [
    "year", "sector", "nbr_firms", "sectoral_VA_BdF",
    "yearly_GDP_INSEE", "sectoral_VA_INSEE"
] 
print("Years:", FIBEN_dataset["year"].unique())
print("Sectors:", FIBEN_dataset["sector"].unique())

# We aggregate the dataset to the sector–year level to construct sectoral time series.
FIBEN_dataset = FIBEN_dataset.drop_duplicates(subset=["sector", "year"])
FIBEN_dataset = FIBEN_dataset.sort_values(["sector", "year"]).reset_index(drop=True)

# We delete rows contaning NAs in the columns that will be used in the time series models.
FIBEN_dataset = FIBEN_dataset.dropna(
    subset=["year", "sector", "sectoral_VA_BdF", "sectoral_VA_INSEE"]
)

# We check there are no non-positive values in the variables that will be log-transformed
print("Check for non-positive values (<= 0) before log transformation:\n")
print("sectoral_VA_BdF:", (FIBEN_dataset["sectoral_VA_BdF"] <= 0).sum())
print("sectoral_VA_INSEE:", (FIBEN_dataset["sectoral_VA_INSEE"] <= 0).sum())

# We attribute correct formats to the columns that will be used in the time series models.
FIBEN_dataset["year"] = pd.to_datetime(FIBEN_dataset["year"]).dt.year
FIBEN_dataset["sectoral_VA_BdF"] = pd.to_numeric(FIBEN_dataset["sectoral_VA_BdF"], errors="coerce")
FIBEN_dataset["sectoral_VA_INSEE"] = pd.to_numeric(FIBEN_dataset["sectoral_VA_INSEE"], errors="coerce")
print(FIBEN_dataset.head())

# We map sector codes 1 to 5 to their explicit names. 
sector_map = {
    1: "Agriculture",
    2: "Industry",
    3: "Construction",
    4: "Market services",
    5: "Non-market services"
}
FIBEN_dataset["sector"] = FIBEN_dataset["sector"].map(sector_map)

# The BdF sectoral value added data for 2019, is incomplete (unreliable) as firms balance sheets were not yet all available at the time of analysis. We discard them and will instead forecast them later.
FIBEN_dataset = FIBEN_dataset[FIBEN_dataset["year"] != 2019]

# We export the clean sector-year level data into CSV. We will use it for forecasting below.
timeseries = FIBEN_dataset.to_csv("/Users/kenzahaouche/COVID-time-series/dataset/timeseries.csv", index=False) 
"""

# =================================================================================================================================================
# LOAD CLEAN DATA
# If you are running it from Git repo, start here.
# =================================================================================================================================================

# Dataset is stored in dataset folder of the Git repo. Run this directly after cloning the repo.
timeseries = pd.read_csv("dataset/timeseries.csv") 

# =================================================================================================================================================
# LOAD INSEE FORECASTS FOR 2020 (sectoral_VA_INSEE)
# =================================================================================================================================================

# We load INSEE forecasts for 2020 sectoral value added (in billions). These estimates are used as inputs in the time series models to forecast BdF sectoral value added for 2020.
forecast_insee_2020 = {
    "Agriculture": 37,
    "Industry": 257.8,
    "Construction": 93.7,
    "Market services": 1096.1,
    "Non-market services": 438.7
}

# =================================================================================================================================================
# MODEL ESTIMATION AND FORECASTING
# =================================================================================================================================================

# We create a dictionary of datasets, one per sector, to run separate time series models for each sector.
datasets_by_sector = {
    s: df.copy() for s, df in timeseries.groupby("sector")
}
sectors = list(datasets_by_sector.keys())

# We can loop over the 5 economic sectors as the methodology of model estimation and forecasting is identical for each.
for sector_name in sectors:

    print("\n==============================")
    print("SECTOR:", sector_name)
    print("\n==============================")

    # We create one dataset per sector 
    sectoraldataset = datasets_by_sector[sector_name].sort_values("year").copy()

    # We create LOG variables for the time series models for i) sectoral_VA_BdF and ii) sectoral_VA_INSEE.
    sectoraldataset["log_sectoral_VA_BdF"] = np.log(sectoraldataset["sectoral_VA_BdF"])
    sectoraldataset["log_sectoral_VA_INSEE"] = np.log(sectoraldataset["sectoral_VA_INSEE"])

    # We create LAGGED variables for the time series models for i) sectoral_VA_BdF and ii) sectoral_VA_INSEE.
    sectoraldataset["lag1_sectoral_VA_BdF"] = sectoraldataset["log_sectoral_VA_BdF"].shift(1)
    sectoraldataset["lag2_sectoral_VA_BdF"] = sectoraldataset["log_sectoral_VA_BdF"].shift(2)
    sectoraldataset["lag1_sectoral_VA_INSEE"] = sectoraldataset["log_sectoral_VA_INSEE"].shift(1)
    sectoraldataset["lag2_sectoral_VA_INSEE"] = sectoraldataset["log_sectoral_VA_INSEE"].shift(2)

    model_data = sectoraldataset.dropna()

    # We run ADF TEST to test for stationarity of the dependent variable i.e. sectoral_VA_BdF. 
    adf_result = adfuller(model_data["log_sectoral_VA_BdF"])
    stationarity = "Stationary" if adf_result[1] < 0.05 else "Non-stationary"

    # We test three different auto-regressive models to compare their results and select the best using AIC criteria.
    models = {
        "Model 0": smf.ols(
            "log_sectoral_VA_BdF ~ lag1_sectoral_VA_BdF + lag2_sectoral_VA_BdF + log_sectoral_VA_INSEE",
            data=model_data
        ).fit(),

        "Model 1": smf.ols(
            "log_sectoral_VA_BdF ~ lag1_sectoral_VA_BdF + lag2_sectoral_VA_BdF + log_sectoral_VA_INSEE + lag1_sectoral_VA_INSEE",
            data=model_data
        ).fit(),

        "Model 2": smf.ols(
            "log_sectoral_VA_BdF ~ lag1_sectoral_VA_BdF + lag2_sectoral_VA_BdF + log_sectoral_VA_INSEE + lag1_sectoral_VA_INSEE + lag2_sectoral_VA_INSEE",
            data=model_data
        ).fit()
    }

    best_model_name = min(models, key=lambda k: models[k].aic)
    best_model = models[best_model_name]

    # We forecast sectoral_VA_BdF for 2019 using the best model among the three. 
    row_2019 = model_data.iloc[[-1]].copy()
    forecast_2019 = np.exp(best_model.predict(row_2019).values[0])

    # We forecast sectoral_VA_BdF for 2020 using the best model among the three, while reinjecting the 2019 forecast above in the model. 
    row_2020 = model_data.iloc[[-1]].copy()
    row_2020["lag1_sectoral_VA_BdF"] = np.log(forecast_2019)
    row_2020["log_sectoral_VA_INSEE"] = np.log(forecast_insee_2020[sector_name])
    forecast_2020 = np.exp(best_model.predict(row_2020).values[0])

    # =================================================================================================================================================
    # PDF REPORT
    # We create a summary PDF report for each sector with: i) the result of the ADF test, ii) the estimates of each of the three models, 
    # iii) the best model name based on AIC, iv) the forecasted values for sectoral_VA_BdF for 2019 and 2020, 
    # and iv) plots of the fitted regression and the forecasted values.
    # =================================================================================================================================================

    # We create a PDF report named after the sector and stored in outputs folder of the Git repo. 
    pdf = PdfPages(f"outputs/Report_{sector_name}.pdf")
    
    # We create a separate cover page with sector name and main results.
    fig, ax = format_pdf_page()

    ax.set_title(f"{sector_name} — Sectoral Report", fontsize=18, fontweight="bold")

    text = f"""
ADF TEST
ADF: {adf_result[0]:.4f}
p-value: {adf_result[1]:.4f}
The time series is {stationarity} (5% significance level)

Best Model: {best_model_name}

Forecasted value for sectoral_VA_BdF 2019: {forecast_2019:.2f}
Forecasted value for sectoral_VA_BdF 2020: {forecast_2020:.2f}
"""

    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, family="monospace")

    pdf.savefig(fig)
    plt.close()

    # We produce summary tables of the estimation results for the three autoregressive models.
    for m in models:

        fig, ax = format_pdf_page()

        ax.set_title(f"{sector_name} — {m}", fontsize=14, fontweight="bold")

        ax.text(
            0.5, 0.5,
            models[m].summary().as_text(),
            ha="center",
            va="center",
            fontsize=7,
            family="monospace"
        )

        pdf.savefig(fig)
        plt.close()

    # We create a fit plot to compare actual and fitted values for sectoral_VA_BdF between 2003 and 2018.
    fig = plt.figure(figsize=(12, 6))

    plt.plot(model_data["year"], model_data["log_sectoral_VA_BdF"], label="Actual")
    plt.plot(model_data["year"], best_model.fittedvalues, label="Fitted")

    plt.title(f"{sector_name} - actual vs fitted values (best model)")
    plt.xlabel("Year")
    plt.ylabel("Log(sectoral_VA_BdF)")
    plt.legend()
    plt.grid()

    pdf.savefig(fig)
    plt.close()

    # On a similar fit plot, we add the forecasted values for sectoral_VA_BdF for 2019 and 2020.
    fig = plt.figure(figsize=(12, 6))

    plt.plot(model_data["year"], model_data["log_sectoral_VA_BdF"], label="Actual")
    plt.plot(model_data["year"], best_model.fittedvalues, label="Fitted")

    plt.scatter(2019, np.log(forecast_2019), color="black", label="2019 forecast")
    plt.scatter(2020, np.log(forecast_2020), color="red", label="2020 forecast")

    plt.title(f"{sector_name} - with forecasted values")
    plt.xlabel("Year")
    plt.ylabel("Log(sectoral_VA_BdF)")
    plt.legend()
    plt.grid()

    pdf.savefig(fig)
    plt.close()

    pdf.close()

    print("Saved:", sector_name)

