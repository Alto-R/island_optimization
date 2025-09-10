import pandas as pd
from pathlib import Path
import re

# 优先 rapidfuzz；无则回退 difflib
try:
    from rapidfuzz import process, fuzz
    _USE_RAPIDFUZZ = True
    print('rapidfuzz')
except ImportError:
    import difflib
    _USE_RAPIDFUZZ = False

# --- 配置项 ---
SCENARIOS = [
    "output_0", "output_2020", "output_2050",
    "output_future_2030", "output_future_2040", "output_future_2050",
]
AFFORDABILITY_ALPHA = 0.10
META_FILE = "filtered_island_1898.csv"
GNP_FILE  = "GNP/GNP.csv"
ROOT = Path(".")

# =============================================================================
# --- 核心修改：终极版别名映射字典 (基于您的所有文件) ---
COUNTRY_ALIAS_MAP = {
    # --- 按您的要求添加 ---
    "Taiwan": "China",

    # --- 修正官方名称不一致 (基于最终诊断) ---
    "Bahamas": "Bahamas, The",
    "Cape Verde": "Cabo Verde",
    "Congo": "Congo, Dem. Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Hong Kong": "Hong Kong SAR, China",
    "Iran": "Iran, Islamic Rep.",
    "Macedonia": "North Macedonia",
    "Russia": "Russian Federation",
    "South Korea": "Korea, Rep.",
    "Saint Lucia": "St. Lucia",
    "Syria": "Syrian Arab Republic",
    "Turkey": "Turkiye",
    "Venezuela": "Venezuela, RB",
    "Yemen": "Yemen, Rep.",

    # --- 修正领土/地区名称 (基于最终诊断) ---
    "American Samoa": "American Samoa",
    "Aruba": "Aruba",
    "Bermuda": "Bermuda",
    "British Virgin Islands": "British Virgin Islands",
    "Cayman Islands": "Cayman Islands",
    "Cook Islands": "New Zealand",
    "Curacao": "Curacao",
    "Faroe Islands": "Faroe Islands",
    "French Polynesia": "French Polynesia",
    "Greenland": "Greenland",
    "Guadeloupe": "France",
    "Guam": "Guam",
    "Guernsey": "Guernsey", # 您的GNP文件中有此条目
    "Jersey": "Jersey",     # 您的GNP文件中有此条目
    "Martinique": "France",
    "Mayotte": "France",
    "Montserrat": "United Kingdom",
    "New Caledonia": "New Caledonia",
    "Niue": "New Zealand",
    "Northern Mariana Islands": "Northern Mariana Islands",
    "Puerto Rico": "Puerto Rico",
    "Reunion": "France",
    "Saint Helena": "St. Helena",
    "Saint Pierre and Miquelon": "France",
    "Sint Maarten (Dutch part)": "Sint Maarten (Dutch part)",
    "Turks and Caicos Islands": "Turks and Caicos Islands",
    "Virgin Islands, U.S.": "Virgin Islands (U.S.)",
    "Wallis and Futuna": "France",
    
    # --- 修正已不存在的地理实体 ---
    "Netherlands Antilles": "Curacao",

    # --- 修正复合名称的特殊情况 ---
    "Virgin Islands; U.S.": "Virgin Islands (U.S.)",
}

# --- 核心修改：修复 load_gnp_2020 函数 ---
def load_gnp_2020(gnp_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gnp_csv)
    df.columns = [str(col).strip() for col in df.columns]
    
    # 在进行任何操作前，先将Country Name列的空格清理干净
    if "Country Name" in df.columns:
        df["Country Name"] = df["Country Name"].str.strip()

    if "Country Name" not in df.columns or "Country Code" not in df.columns:
        raise ValueError("GNP.csv 缺少必要列：'Country Name' 或 'Country Code'")

    # 后续逻辑保持不变
    if "Year" in df.columns and "Value" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df_2020 = df.loc[df["Year"] == 2020].copy()
        df_2020 = df_2020.rename(columns={"Value": "value_2020"})
    elif "2020" in df.columns:
        df_2020 = df.rename(columns={"2020": "value_2020"})
    else:
        raise ValueError("GNP.csv 未找到 'Year'/'Value' 列或 '2020' 列")
        
    df_2020["value_2020"] = pd.to_numeric(df_2020["value_2020"], errors="coerce")
    # 注意：我们在这里保留了 value_2020 为 NaN 的行，以便在主流程中给出更精确的诊断
    return df_2020[["Country Name", "Country Code", "value_2020"]].drop_duplicates(subset=["Country Name"]).reset_index(drop=True)

# --- 核心函数升级：attach_income_from_gnp_2020 (增加诊断)---
def attach_income_from_gnp_2020(meta_df: pd.DataFrame, gnp2020_df: pd.DataFrame,
                                country_col_in_meta: str = "Country",
                                score_cutoff: int = 85) -> pd.DataFrame:
    meta = meta_df.copy()
    candidates = list(gnp2020_df["Country Name"].astype(str).unique())
    # 修改gnp_map的创建，以处理可能存在的NaN值
    gnp_map = gnp2020_df.set_index("Country Name")[["Country Code", "value_2020"]].to_dict(orient="index")
    
    results = []
    for _, r in meta.iterrows():
        q_original = str(r.get(country_col_in_meta, "") or "")
        best_name, score, match_method = None, 0, "No Match"
        
        # 匹配流程
        if q_original in COUNTRY_ALIAS_MAP:
            mapped_name = COUNTRY_ALIAS_MAP[q_original]
            if mapped_name in gnp_map:
                best_name, score, match_method = mapped_name, 100, "Alias"
        if best_name is None:
            q_processed = preprocess_country_name(q_original)
            if q_processed in gnp_map:
                best_name, score, match_method = q_processed, 100, "Exact (Processed)"
        if best_name is None:
            q_processed = preprocess_country_name(q_original)
            b_name, b_score = fuzzy_match_country(q_processed, candidates, score_cutoff=score_cutoff)
            if b_name:
                best_name, score, match_method = b_name, b_score, "Fuzzy"
        
        # 获取信息并检查2020年数据是否有效
        income_val = float("nan")
        info = gnp_map.get(best_name) if best_name else None
        if info:
            # 只有当GNI值不是NaN时，才认为匹配成功
            if pd.notna(info["value_2020"]):
                income_val = float(info["value_2020"])
            else:
                match_method = "Match OK, but 2020 GNI is Null" # 给出更精确的诊断
        
        results.append({
            "_match_country_name": best_name,
            "_match_country_code": info["Country Code"] if info else None,
            "_match_score": score,
            "_match_method": match_method,
            "income_per_capita": income_val
        })
        
    match_df = pd.DataFrame(results, index=meta.index)
    return pd.concat([meta, match_df], axis=1)
    
# --- 其余所有辅助函数保持不变 ---
def preprocess_country_name(name: str):
    if not isinstance(name, str) or not name.strip(): return ""
    name = name.split(';')[0].split('/')[0].split(',')[0].strip()
    name = re.sub(r"\(.*\)", "", name).strip()
    return name

def fuzzy_match_country(query: str, candidates: list, score_cutoff: int = 80):
    if not isinstance(query, str) or not query.strip(): return None, 0
    if _USE_RAPIDFUZZ:
        best = process.extractOne(query, candidates, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        return (best[0], int(best[1])) if best else (None, 0)
    else:
        best = difflib.get_close_matches(query, candidates, n=1, cutoff=score_cutoff/100.0)
        if best:
            return best[0], int(difflib.SequenceMatcher(None, query, best[0]).ratio() * 100)
        return None, 0

def load_cost(cost_csv: Path) -> float:
    df = pd.read_csv(cost_csv)
    mask = df["Cost_Item"].astype(str).str.strip().eq("--- TOTAL ANNUAL COST ---")
    return float(df.loc[mask, "Cost_Value"].values[0]) if mask.any() else float(df["Cost_Value"].iloc[-1])

def load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "E_demand" not in df.columns: raise ValueError(f"{results_csv} 缺少列 E_demand")
    if "P_Dload_E" not in df.columns: df["P_Dload_E"] = 0.0
    return df

def compute_billed_kwh(df: pd.DataFrame) -> float:
    return float(((df["E_demand"] - df["P_Dload_E"]).clip(lower=0) * 3).sum())

def compute_consumption_pc(df: pd.DataFrame, population: float) -> float:
    total_demand = float((df["E_demand"] * 3).sum())
    return total_demand / population if population > 0 else 0.0

def compute_affordable_tariff(income_pc: float, cons_pc_kwh: float, alpha: float = AFFORDABILITY_ALPHA) -> float:
    return (alpha * income_pc) / cons_pc_kwh if cons_pc_kwh > 0 else float("inf")

def evaluate_one(island_id, lat, lon, income_pc, population, scen_dir: Path):
    stem = f"{lat}_{lon}"
    cost_csv, results_csv = scen_dir / f"{stem}_best_cost.csv", scen_dir / f"{stem}_results.csv"
    if not cost_csv.exists() or not results_csv.exists(): return None
    total_cost = load_cost(cost_csv)
    res_df = load_results(results_csv)
    billed_kwh = compute_billed_kwh(res_df)
    cons_pc_kwh = compute_consumption_pc(res_df, population)
    tariff_breakeven = total_cost / billed_kwh if billed_kwh > 0 else float("inf")
    tariff_affordable = compute_affordable_tariff(income_pc, cons_pc_kwh)
    vg = tariff_breakeven - tariff_affordable
    return billed_kwh, cons_pc_kwh, tariff_breakeven, tariff_affordable, vg

# --- 主流程 (保持不变) ---
def main():
    meta = pd.read_csv(ROOT / META_FILE)
    gnp2020 = load_gnp_2020(ROOT / GNP_FILE)
    meta2 = attach_income_from_gnp_2020(meta, gnp2020, country_col_in_meta="Country", score_cutoff=85)
    
    miss = meta2["income_per_capita"].isna()
    if miss.any():
        print("[WARN] 最终无法匹配 GNI 的岛屿 (原因见 _match_method 列):")
        columns_to_show = ["ID", "Country", "_match_country_name", "_match_score", "_match_method"]
        unmatched_df = meta2.loc[miss, columns_to_show]
        print(unmatched_df) # 打印所有，而不仅仅是前20条
        output_filename = "unmatched_countries_final_v3.csv"
        unmatched_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n[INFO] 剩余 {miss.sum()} 条未匹配的岛屿列表已保存至: {output_filename}")

    # (后续计算部分保持不变)
    rows = []
    for _, row in meta2.loc[~miss].iterrows():
        island_id, lat, lon, population_raw = row["ID"], float(row["Lat"]), float(row["Long"]), float(row["pop"])
        population = min(population_raw, 500.0) if population_raw >= 500 else population_raw
        income_pc = float(row["income_per_capita"])
        scen_2020_dir = ROOT / "output_2020"
        results_csv_2020 = scen_2020_dir / f"{lat}_{lon}_results.csv"
        if not results_csv_2020.exists(): continue
        try:
            res_df_2020 = load_results(results_csv_2020)
            cons_pc_kwh_2020_baseline = compute_consumption_pc(res_df_2020, population)
        except Exception as e:
            print(f"[ERROR] Island {island_id}: {e}, skipping.")
            continue
        for scen in SCENARIOS:
            scen_dir = ROOT / scen
            if not scen_dir.exists(): continue
            res = evaluate_one(island_id, lat, lon, income_pc, population, scen_dir)
            if res is None: continue
            billed_kwh, _, tariff_breakeven, _, _ = res
            tariff_affordable_fixed = compute_affordable_tariff(income_pc, cons_pc_kwh_2020_baseline)
            vg_fixed = tariff_breakeven - tariff_affordable_fixed
            rows.append({
                "island_id": island_id, "lat": lat, "lon": lon, "Country": row["Country"],
                "_match_country_name": row["_match_country_name"], "_match_country_code": row["_match_country_code"],
                "_match_score": row["_match_score"], "scenario": scen, "population_raw": population_raw,
                "population_calc": population, "income_per_capita_2020": income_pc,
                "consumption_pc_kwh": cons_pc_kwh_2020_baseline, "billed_kwh": billed_kwh,
                "tariff_breakeven": tariff_breakeven, "tariff_affordable": tariff_affordable_fixed,
                "viability_gap": vg_fixed
            })

    summary = pd.DataFrame(rows)
    summary.to_csv("island_viability_summary_electric.csv", index=False)
    print("[OK] saved to island_viability_summary_electric.csv")

if __name__ == "__main__":
    main()