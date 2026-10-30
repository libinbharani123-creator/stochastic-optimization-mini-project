#!/usr/bin/env python3
"""Build daily_sales.csv from the UCI Online Retail dataset."""
import io
import zipfile
import urllib.request
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def build(stock_code: str = "85123A", out: str = "daily_sales.csv") -> None:
    with urllib.request.urlopen(URL) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    xlsx = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
    with z.open(xlsx) as fh:
        df = pd.read_excel(fh, engine="openpyxl")

    df = df[(df["StockCode"].astype(str) == stock_code) & (df["Quantity"] > 0)]
    daily = (df.assign(date=df["InvoiceDate"].dt.normalize())
               .groupby("date")["Quantity"].sum()
               .asfreq("D", fill_value=0)
               .rename("units_sold").reset_index())
    daily["promo"] = 0
    daily["stockout_flag"] = 0
    daily.to_csv(out, index=False)
    print(f"wrote {out}: {len(daily)} days, mean {daily.units_sold.mean():.1f}")


if __name__ == "__main__":
    build()
