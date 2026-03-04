# MajsterPlus Data Dictionary

**Dataset version**: 1.0

---

## customers.csv

**Rows**: 5,000 | **Columns**: 21 (20 features + 1 target)

| # | Column | Type | Range / Values | Description |
|---|--------|------|---------------|-------------|
| 1 | `customer_id` | string | "MP00001"–"MP05000" | Unique customer identifier; join key to `transactions.csv` |
| 2 | `registration_date` | string | e.g. "15-sty-2022" | Date the customer registered. Uses Polish month abbreviations (sty, lut, mar, kwi, maj, cze, lip, sie, wrz, paz, lis, gru) |
| 3 | `age` | int | 18–78 | Customer age in years |
| 4 | `gender` | categorical | "M", "K" | Gender — "M" = male, "K" = female (Polish: Kobieta) |
| 5 | `city` | categorical | 12 values | Customer's city of residence (Warszawa, Krakow, Lodz, Wroclaw, Poznan, Gdansk, Szczecin, Bydgoszcz, Lublin, Bialystok, Katowice, Gdynia) |
| 6 | `loyalty_member` | categorical | "Tak", "Nie" | Loyalty programme membership — "Tak" = Yes, "Nie" = No (Polish) |
| 7 | `total_spend` | string | e.g. "PLN 1,234.50" | Cumulative spend. Stored as a formatted currency string with "PLN" prefix and comma thousands separator |
| 8 | `purchase_count` | int | 1–87 | Total number of transactions |
| 9 | `avg_basket_value` | float | typically 35.0–890.0 | Average transaction value in PLN |
| 10 | `days_since_last_purchase` | int | 0–540 | Days elapsed since the customer's most recent purchase (relative to reference date 2025-01-15) |
| 11 | `product_categories_bought` | int | 1–8 | Number of distinct product categories purchased |
| 12 | `online_ratio` | float | 0.0–1.0 | Proportion of purchases made online |
| 13 | `satisfaction_score` | float | 1.0–5.0 (valid range) | Customer satisfaction rating on a 1-to-5 scale |
| 14 | `customer_service_contacts` | int | 0–15 | Number of contacts with customer service |
| 15 | `newsletter_subscriber` | categorical | "Tak", "Nie", "Nie dotyczy" | Newsletter subscription status — "Nie dotyczy" = Not applicable (Polish) |
| 16 | `monthly_income_bracket` | categorical | "A"–"E" | Self-reported income bracket (A = lowest, E = highest) |
| 17 | `district_type` | categorical | "urban", "suburban", "rural" | Type of residential area |
| 18 | `store_distance_km` | float | 0.5–65.0 | Distance to the nearest MajsterPlus store in kilometres |
| 19 | `referral_source` | categorical | "friend", "online_ad", "walk_in", "social_media" | How the customer first learned about MajsterPlus |
| 20 | `account_age_days` | int | ~16–1096 | Number of days since registration (relative to reference date 2025-01-15) |
| 21 | `is_lapsed` | int | 0 / 1 | **Target variable** — 1 = customer has lapsed (no purchase in 90+ days), 0 = active |

---

## transactions.csv

**Rows**: ~25,000 | **Columns**: 8 | **Join key**: `customer_id`

| # | Column | Type | Range / Values | Description |
|---|--------|------|---------------|-------------|
| 1 | `transaction_id` | string | "T000001"–"T025000" | Unique transaction identifier |
| 2 | `customer_id` | string | "MP00001"–"MP05000" | Join key to `customers.csv` |
| 3 | `transaction_date` | string | Polish date format | Date of transaction (same Polish month format as `registration_date`) |
| 4 | `store_id` | string | "S01"–"S12" | Store where the transaction took place |
| 5 | `product_category` | categorical | Tools, Paint, Plumbing, Electrical, Garden, Flooring, Building Materials, Lighting | Product category of the purchase |
| 6 | `amount` | float | > 0 | Transaction value in PLN |
| 7 | `items_count` | int | 1–20 | Number of items in the transaction |
| 8 | `payment_method` | categorical | cash, card, online, mobile | Payment method used |

---

## Store Mapping

| Store ID | City |
|----------|------|
| S01 | Warszawa |
| S02 | Krakow |
| S03 | Lodz |
| S04 | Wroclaw |
| S05 | Poznan |
| S06 | Gdansk |
| S07 | Szczecin |
| S08 | Bydgoszcz |
| S09 | Lublin |
| S10 | Bialystok |
| S11 | Katowice |
| S12 | Gdynia |

---

## Known Data Quality Issues

The dataset reflects real-world data collection — it contains **10 quality issues** that require cleaning before analysis. Below is what has been documented by the data engineering team.

| # | Issue | Affected Column(s) |
|---|-------|--------------------|
| 1 | Polish date format (e.g. "15-sty-2024") | `registration_date`, `transaction_date` |
| 2 | Currency stored as formatted string (e.g. "PLN 1,234.50") | `total_spend` |
| 3 | Polish categorical values ("Tak"/"Nie") | `loyalty_member`, `newsletter_subscriber` |
| 4 | Missing values (~3%) | `age` |
| 5 | Missing values (~8%) | `online_ratio` |
| 6 | Missing values (~12%) | `monthly_income_bracket` |
| 7 | Extreme outliers (values far above normal range) | `avg_basket_value` |
| 8 | Values outside valid 1.0–5.0 range | `satisfaction_score` |
| 9 | Missing values (~6%) | `satisfaction_score` |
| 10 | Missing values (~4%) | `referral_source` |
