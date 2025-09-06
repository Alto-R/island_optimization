import pandas as pd
from pathlib import Path

# 优先 rapidfuzz；无则回退 difflib
try:
    from rapidfuzz import process, fuzz
    _USE_RAPIDFUZZ = True
except Exception:
    import difflib
    _USE_RAPIDFUZZ = False

SCENARIOS = [
    "output_0",
    "output_2020",
    "output_2050",
    "output_future_2030",
    "output_future_2040",
    "output_future_2050",
]

AFFORDABILITY_ALPHA = 0.10
META_FILE = "filtered_island_1898.csv"   # 需含: ID, Lat, Long, Country, pop
GNP_FILE  = "GNP/GNP.csv"                    # 需含: Country Name, Country Code, 以及 Year/Value（长表）或 '2020'（宽表）
ROOT = Path(".")

# -----------------------------
# 成本与结果读取 + 指标计算
# -----------------------------
def load_cost(cost_csv: Path) -> float:
    df = pd.read_csv(cost_csv)
    mask = df["Cost_Item"].astype(str).str.strip().eq("--- TOTAL ANNUAL COST ---")
    if mask.any():
        return float(df.loc[mask, "Cost_Value"].values[0])
    return float(df["Cost_Value"].iloc[-1])

def load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "E_demand" not in df.columns:
        raise ValueError(f"{results_csv} 缺少列 E_demand")
    if "P_Dload_E" not in df.columns:
        df["P_Dload_E"] = 0.0
    return df

def compute_billed_kwh(df: pd.DataFrame) -> float:
    # 售电量 = sum_t (E_demand - P_Dload_E) * 3
    return float(((df["E_demand"] - df["P_Dload_E"]).clip(lower=0) * 3).sum())

def compute_consumption_pc(df: pd.DataFrame, population: float) -> float:
    # 人均年用电量（不减削减）= sum_t E_demand*3 / population
    total_demand = float((df["E_demand"] * 3).sum())
    return total_demand / population if population > 0 else 0.0

def compute_affordable_tariff(income_pc: float, cons_pc_kwh: float, alpha: float = AFFORDABILITY_ALPHA) -> float:
    if cons_pc_kwh <= 0:
        return float("inf")
    return (alpha * income_pc) / cons_pc_kwh

def evaluate_one(island_id, lat, lon, income_pc, population, scen_dir: Path):
    stem = f"{lat}_{lon}"
    cost_csv = scen_dir / f"{stem}_best_cost.csv"
    results_csv = scen_dir / f"{stem}_results.csv"
    if not cost_csv.exists() or not results_csv.exists():
        return None
    total_cost = load_cost(cost_csv)
    res_df = load_results(results_csv)
    billed_kwh = compute_billed_kwh(res_df)
    cons_pc_kwh = compute_consumption_pc(res_df, population)
    tariff_breakeven = total_cost / billed_kwh if billed_kwh > 0 else float("inf")
    tariff_affordable = compute_affordable_tariff(income_pc, cons_pc_kwh)
    vg = tariff_breakeven - tariff_affordable
    return billed_kwh, cons_pc_kwh, tariff_breakeven, tariff_affordable, vg

# -----------------------------
# 只取 2020 年的 GNP（固定列名）
# -----------------------------

def load_gnp_2020(gnp_csv: Path) -> pd.DataFrame:
    """
    从 GNP.csv 读取 2020 年 GNI per capita（current US$）。
    固定国家列名：'Country Name', 'Country Code'。
    支持：
      A) 长表：含 'Year' 与 'Value' -> 过滤 Year==2020
      B) 宽表：含 '2020' 列 -> 直接读取该列
    返回列：['Country Name','Country Code','value_2020']，其中 value_2020 为 float。
    """
    df = pd.read_csv(gnp_csv)
    if "Country Name" not in df.columns or "Country Code" not in df.columns:
        raise ValueError("GNP.csv 缺少必要列：'Country Name' 或 'Country Code'")

    # 情况 A：长表（Year/Value）
    if "Year" in df.columns:
        if "Value" not in df.columns:
            raise ValueError("GNP.csv 为长表结构，但缺少 'Value' 列")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df_2020 = df.loc[df["Year"] == 2020, ["Country Name", "Country Code", "Value"]].dropna(subset=["Value"]).copy()
        df_2020 = df_2020.rename(columns={"Value": "value_2020"})
        df_2020["value_2020"] = pd.to_numeric(df_2020["value_2020"], errors="coerce")
        df_2020 = df_2020.dropna(subset=["value_2020"]).drop_duplicates(subset=["Country Name"])
        return df_2020.reset_index(drop=True)

    # 情况 B：宽表（有 '2020' 列）
    if "2020" in df.columns:
        df["2020"] = pd.to_numeric(df["2020"], errors="coerce")
        df_2020 = df[["Country Name", "Country Code", "2020"]].dropna(subset=["2020"]).copy()
        df_2020 = df_2020.rename(columns={"2020": "value_2020"})
        df_2020 = df_2020.drop_duplicates(subset=["Country Name"])
        return df_2020.reset_index(drop=True)

    raise ValueError("GNP.csv 未找到 'Year' 列或 '2020' 列，无法提取 2020 数据")

# -----------------------------
# 模糊匹配（原始国家名，不做标准化）
# -----------------------------
def fuzzy_match_country(query: str, candidates: list, score_cutoff: int = 80):
    """
    用原始国家名做模糊匹配；返回 (best_name, score) 或 (None, 0)
    """
    if not isinstance(query, str) or not query.strip():
        return None, 0

    if _USE_RAPIDFUZZ:
        best = process.extractOne(query, candidates, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        if best is None:
            return None, 0
        return best[0], int(best[1])
    else:
        best = difflib.get_close_matches(query, candidates, n=1, cutoff=score_cutoff/100.0)
        if best:
            return best[0], int(difflib.SequenceMatcher(None, query, best[0]).ratio() * 100)
        return None, 0

def attach_income_from_gnp_2020(meta_df: pd.DataFrame, gnp2020_df: pd.DataFrame,
                                country_col_in_meta: str = "Country",
                                score_cutoff: int = 85) -> pd.DataFrame:
    """
    将 meta['Country'] 与 GNP['Country Name'] 做模糊匹配，合并 2020 年 GNI per capita 为 income_per_capita。
    """
    meta = meta_df.copy()
    candidates = list(gnp2020_df["Country Name"].astype(str).unique())

    matched_name = []
    matched_code = []
    matched_score = []
    matched_value = []

    # 为快速索引构建映射：Country Name -> (Country Code, value_2020)
    gnp_map = gnp2020_df.set_index("Country Name")[["Country Code", "value_2020"]].to_dict(orient="index")

    for _, r in meta.iterrows():
        q = str(r.get(country_col_in_meta, "") or "")
        best_name, score = fuzzy_match_country(q, candidates, score_cutoff=score_cutoff)
        matched_name.append(best_name)
        matched_score.append(score)
        if best_name is None:
            matched_code.append(None)
            matched_value.append(float("nan"))
        else:
            info = gnp_map.get(best_name, None)
            if info is None:
                matched_code.append(None)
                matched_value.append(float("nan"))
            else:
                matched_code.append(info["Country Code"])
                matched_value.append(float(info["value_2020"]))

    meta["_match_country_name"] = matched_name
    meta["_match_country_code"] = matched_code
    meta["_match_score"] = matched_score
    meta["income_per_capita"] = matched_value
    return meta

# -----------------------------
# 主流程
# -----------------------------
def main():
    # 1) 岛屿元数据
    meta = pd.read_csv(ROOT / META_FILE)
    for c in ["ID", "Lat", "Long", "Country", "pop"]:
        if c not in meta.columns:
            raise ValueError(f"{META_FILE} 需包含列：{c}")

    # 2) 读 GNP 2020
    gnp2020 = load_gnp_2020(ROOT / GNP_FILE)

    # 3) 模糊匹配并合并 2020 GNI per capita
    meta2 = attach_income_from_gnp_2020(meta, gnp2020, country_col_in_meta="Country", score_cutoff=80)

    # 告警：未匹配成功的行
    miss = meta2["income_per_capita"].isna()
    if miss.any():
        print("[WARN] 未匹配到 2020 GNI 的岛屿（前 20 条）：")
        print(meta2.loc[miss, ["ID", "Country", "_match_country_name", "_match_score"]].head(20))
        print(f"[WARN] 共 {miss.sum()} 条未匹配，将从计算中跳过。")

    # 4) 遍历情景做可行性计算
    rows = []
    for _, row in meta2.loc[~miss].iterrows():
        island_id = row["ID"]
        lat = float(row["Lat"])
        lon = float(row["Long"])
        population_raw = float(row["pop"])
        population = min(population_raw, 500.0) if population_raw >= 500 else population_raw
        income_pc = float(row["income_per_capita"])  # 2020 年的人均 GNI（单位=GNP.csv单位）

        for scen in SCENARIOS:
            scen_dir = ROOT / scen
            if not scen_dir.exists():
                continue
            res = evaluate_one(island_id, lat, lon, income_pc, population, scen_dir)
            if res is None:
                continue
            billed_kwh, cons_pc_kwh, tariff_breakeven, tariff_affordable, vg = res
            rows.append({
                "island_id": island_id,
                "lat": lat,
                "lon": lon,
                "Country": row["Country"],
                "_match_country_name": row["_match_country_name"],
                "_match_country_code": row["_match_country_code"],
                "_match_score": row["_match_score"],
                "scenario": scen,
                "population_raw": population_raw,
                "population_calc": population,
                "income_per_capita_2020": income_pc,
                "consumption_pc_kwh": cons_pc_kwh,  # 不减削减量
                "billed_kwh": billed_kwh,           # 已减削减量
                "tariff_breakeven": tariff_breakeven,
                "tariff_affordable": tariff_affordable,
                "viability_gap": vg
            })

    summary = pd.DataFrame(rows)
    summary.to_csv("island_viability_summary_electric.csv", index=False)
    print("[OK] saved to island_viability_summary_electric.csv")

if __name__ == "__main__":
    main()

