"""Veri ön işleme modülü — MUCars-2024 fiyat kategorisi sınıflandırması."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "cars_dataframe.csv"

PRICE_BINS = [0, 60_000, 110_000, 170_000, np.inf]
PRICE_LABELS = ["Dusuk", "Orta", "Yuksek", "Luks"]

TOP_EQUIPMENT = [
    "Electric Windows",
    "CD/MP3/Bluetooth",
    "Air Conditioning",
    "Leather Seats",
    "Airbags",
    "Alloy Wheels",
    "Rear Camera",
    "Parking Sensors",
    "Navigation System/GPS",
    "Central Locking",
]

CATEGORICAL_COLS = [
    "Brand",
    "Model",
    "Condition",
    "Gearbox",
    "Fuel",
    "Origin",
    "First Owner",
    "Location",
    "Sector",
]

NUMERIC_COLS = [
    "Year",
    "Mileage_Numeric",
    "Fiscal_Power_Numeric",
    "Car_Age",
    "Equipment_Count",
    "Number of Doors",
]

RARE_THRESHOLD = 50


def parse_mileage(value: str | float) -> float | np.nan:
    """Mileage aralığından orta noktayı çıkarır."""
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value).replace("\u00a0", " ").replace(" ", "")
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return np.nan
    nums = [int(n) for n in numbers]
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    return float(nums[0])


def parse_fiscal_power(value: str | float) -> float | np.nan:
    """Fiscal Power değerinden sayısal CV çıkarır."""
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value)
    if "Plus de" in text or "plus de" in text.lower():
        match = re.search(r"(\d+)", text)
        return float(match.group(1)) + 5 if match else 45.0
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def parse_equipment_list(value: str | float) -> list[str]:
    """Equipment sütunundaki liste string'ini parse eder."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def group_rare_categories(series: pd.Series, threshold: int = RARE_THRESHOLD) -> pd.Series:
    """Frekansı düşük kategorileri 'Other' altında birleştirir."""
    counts = series.value_counts()
    rare = counts[counts < threshold].index
    return series.where(~series.isin(rare), "Other")


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_price(df: pd.DataFrame, max_price: int = 1_000_000) -> pd.DataFrame:
    """Geçerli fiyatı olan ve aykırı değerleri temizlenmiş kayıtları döndürür."""
    data = df.copy()
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna(subset=["Price"])
    data = data[data["Price"] > 1_000]
    data = data[data["Price"] <= max_price]
    return data.reset_index(drop=True)


def assign_price_category(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile tabanlı 4 fiyat kategorisi atar."""
    data = df.copy()
    data["Price_Category"] = pd.cut(
        data["Price"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        include_lowest=True,
        right=True,
    )
    return data.dropna(subset=["Price_Category"]).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Özellik mühendisliği adımlarını uygular."""
    data = df.copy()

    data["Mileage_Numeric"] = data["Mileage"].apply(parse_mileage)
    data["Fiscal_Power_Numeric"] = data["Fiscal Power"].apply(parse_fiscal_power)
    data["Car_Age"] = 2024 - pd.to_numeric(data["Year"], errors="coerce")

    equipment_lists = data["Equipment"].apply(parse_equipment_list)
    data["Equipment_Count"] = equipment_lists.apply(len)
    for item in TOP_EQUIPMENT:
        col_name = f"Equip_{item.replace('/', '_').replace(' ', '_')}"
        data[col_name] = equipment_lists.apply(lambda lst, i=item: int(i in lst))

    for col in CATEGORICAL_COLS:
        data[col] = data[col].fillna("Unknown").astype(str)
        data[col] = group_rare_categories(data[col])

    data["Number of Doors"] = pd.to_numeric(data["Number of Doors"], errors="coerce")

    equip_cols = [c for c in data.columns if c.startswith("Equip_")]
    numeric_cols = NUMERIC_COLS + equip_cols

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median())

    return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Model için kullanılacak özellik sütunlarını döndürür."""
    equip_cols = [c for c in df.columns if c.startswith("Equip_")]
    return CATEGORICAL_COLS + NUMERIC_COLS + equip_cols


def encode_features(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Kategorik sütunları LabelEncoder ile sayısallaştırır."""
    data = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            data[col] = data[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    return data, encoders


def prepare_dataset(
    path: Path | str = DATA_PATH,
    max_price: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], list[str]]:
    """Ham CSV'den model-ready DataFrame üretir."""
    raw = load_raw_data(path)
    cleaned = clean_price(raw, max_price=max_price)
    labeled = assign_price_category(cleaned)
    featured = engineer_features(labeled)
    encoded, encoders = encode_features(featured, fit=True)
    feature_cols = get_feature_columns(featured)
    return encoded, encoders, feature_cols


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Price_Category",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test bölmesi."""
    X = df[feature_cols]
    y = df[target_col].astype(str)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_eda_dataframe(path: Path | str = DATA_PATH, max_price: int = 1_000_000) -> pd.DataFrame:
    """EDA için etiketli ama encode edilmemiş veri döndürür."""
    raw = load_raw_data(path)
    cleaned = clean_price(raw, max_price=max_price)
    labeled = assign_price_category(cleaned)
    return engineer_features(labeled)
