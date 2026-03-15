# A/B Testing Dataset: Data Description Report

## Overview

- Rows: **30,000**

- Columns: **24**

- Time range: **2025-01-01 00:01:18.569492** to **2025-02-19 21:27:27.575775**

- Overall conversion rate: **0.1512**

- Overall bounce rate: **0.1891**

- Total revenue (USD): **786,256.59**


## Variant Summary

| variant_group | sessions | users | conversions | conversion_rate | bounces | bounce_rate | avg_time_spent_min | avg_pages | total_revenue | revenue_per_session | conversion_ci_low | conversion_ci_high |
| ------------- | -------- | ----- | ----------- | --------------- | ------- | ----------- | ------------------ | --------- | ------------- | ------------------- | ----------------- | ------------------ |
| Cold          | 9928     | 7306  | 1795        | 0.180802        | 1853    | 0.186644    | 10.5345            | 5.49275   | 305188        | 30.7401             | 0.173355          | 0.188495           |
| Heat          | 10026    | 7273  | 1521        | 0.151706        | 1911    | 0.190604    | 10.4813            | 5.5016    | 262492        | 26.1811             | 0.144817          | 0.158861           |
| Vibrant       | 10046    | 7324  | 1219        | 0.121342        | 1908    | 0.189926    | 10.5568            | 5.50995   | 218577        | 21.7576             | 0.115101          | 0.127872           |


## Statistical Tests

### Conversion Rate (Chi-square)

- chi2: **137.6157**

- dof: **2**

- p-value: **1.30957e-30**


### Pairwise Conversion Tests (Two-proportion z-test + Holm-Bonferroni)

| comparison      | diff_p    | z       | p_value     | holm_reject | holm_threshold |
| --------------- | --------- | ------- | ----------- | ----------- | -------------- |
| Cold_vs_Vibrant | 0.0594599 | 11.7382 | 0           | True        | 0.0166667      |
| Heat_vs_Vibrant | 0.0303637 | 6.26486 | 3.7316e-10  | True        | 0.025          |
| Cold_vs_Heat    | 0.0290962 | 5.52063 | 3.37795e-08 | True        | 0.05           |


### Revenue per Session (Welch t-test + Mann-Whitney U + Holm-Bonferroni)

| comparison      | mean_a  | mean_b  | t_p_value   | u_p_value   | holm_reject |
| --------------- | ------- | ------- | ----------- | ----------- | ----------- |
| Cold_vs_Vibrant | 30.7401 | 21.7576 | 4.02734e-06 | 6.41819e-22 | True        |
| Heat_vs_Vibrant | 26.1811 | 21.7576 | 0.0197761   | 5.25273e-06 | True        |
| Cold_vs_Heat    | 30.7401 | 26.1811 | 0.0212042   | 3.37544e-07 | True        |


## Data Dictionary (Observed)

| column                | dtype          | missing | missing_pct | unique | example                          | top_values                                                                                                                                                     | min | p25  | median | p75   | max  | mean     |
| --------------------- | -------------- | ------- | ----------- | ------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | ---- | ------ | ----- | ---- | -------- |
| user_id               | str            | 0       | 0           | 12984  | U10477                           | U13006 (9); U13329 (8); U04903 (8); U05015 (8); U02083 (8)                                                                                                     |     |      |        |       |      |          |
| session_id            | str            | 0       | 0           | 30000  | S000001                          | S000001 (1); S000002 (1); S000003 (1); S000004 (1); S000005 (1)                                                                                                |     |      |        |       |      |          |
| sign_in               | str            | 0       | 0           | 2      | Email                            | Email (20975); Guest (9025)                                                                                                                                    |     |      |        |       |      |          |
| name                  | str            | 0       | 0           | 23706  | Victor Navarro-Noël              | 鈴木 花子 (15); 佐藤 花子 (15); 佐藤 くみ子 (13); 佐藤 京助 (13); 鈴木 智也 (13)                                                                                                    |     |      |        |       |      |          |
| demographic_age       | int64          | 0       | 0           | 67     | 31                               |                                                                                                                                                                | 14  | 30   | 47     | 64    | 80   | 46.9055  |
| demographic_age_group | str            | 0       | 0           | 3      | Adult                            | Adult (20139); Old (7156); Teenage (2705)                                                                                                                      |     |      |        |       |      |          |
| demographic_gender    | str            | 0       | 0           | 3      | Female                           | Male (13606); Female (13487); No Answer (2907)                                                                                                                 |     |      |        |       |      |          |
| email                 | str            | 0       | 0           | 20974  | victornavarronoël251@hotmail.com | Not Provided (9025); 鈴木美加子904@gmail.com (2); 吉田翔太957@hotmail.com (2); victornavarronoël251@hotmail.com (1); 王秀云617@gmail.com (1)                               |     |      |        |       |      |          |
| location              | str            | 0       | 0           | 26     | Rome                             | Manchester (1200); Chicago (1199); Toronto (1194); Mumbai (1191); Delhi (1185)                                                                                 |     |      |        |       |      |          |
| country               | str            | 0       | 0           | 15     | Italy                            | India (3529); USA (3444); Canada (2374); UK (2324); Japan (2323)                                                                                               |     |      |        |       |      |          |
| device_type           | str            | 0       | 0           | 3      | Desktop                          | Mobile (18008); Desktop (8953); Tablet (3039)                                                                                                                  |     |      |        |       |      |          |
| timestamp             | datetime64[us] | 0       | 0           | 30000  | 2025-01-18 21:30:04.168185       | 2025-01-18 21:30:04.168185 (1); 2025-01-12 04:36:36.971286 (1); 2025-01-20 06:49:20.481864 (1); 2025-02-01 08:59:46.938408 (1); 2025-02-16 16:02:00.285005 (1) |     |      |        |       |      |          |
| variant_group         | str            | 0       | 0           | 3      | Heat                             | Vibrant (10046); Heat (10026); Cold (9928)                                                                                                                     |     |      |        |       |      |          |
| time_spent            | float64        | 0       | 0           | 1901   | 2.65                             |                                                                                                                                                                | 1   | 5.78 | 10.58  | 15.27 | 20   | 10.5242  |
| pages_visited         | int64          | 0       | 0           | 10     | 7                                |                                                                                                                                                                | 1   | 3    | 5      | 8     | 10   | 5.50147  |
| conversion_flag       | int64          | 0       | 0           | 2      | 0                                |                                                                                                                                                                | 0   | 0    | 0      | 0     | 1    | 0.151167 |
| conversion_type       | str            | 0       | 0           | 3      | NCT                              | NCT (25465); Purchase (2992); Signup (1543)                                                                                                                    |     |      |        |       |      |          |
| traffic_source        | str            | 0       | 0           | 4      | Organic                          | Organic (15084); Social (5992); Paid (5950); Referral (2974)                                                                                                   |     |      |        |       |      |          |
| product_purchased     | str            | 0       | 0           | 74     | NPP                              | NPP (27008); JBL Flip 6 (63); Anker Soundcore 3 (55); Bang & Olufsen Beosound A5 (53); LG XBOOM Go PL7 (53)                                                    |     |      |        |       |      |          |
| revenue_usd           | float64        | 0       | 0           | 41     | 0.0                              |                                                                                                                                                                | 0   | 0    | 0      | 0     | 2499 | 26.2086  |
| payment_type          | str            | 0       | 0           | 3      | NPT                              | NPT (27008); Card (2095); COD (897)                                                                                                                            |     |      |        |       |      |          |
| card_type             | str            | 0       | 0           | 4      | NCAT                             | NCAT (27905); Visa (710); Master (707); Amex (678)                                                                                                             |     |      |        |       |      |          |
| coupon_applied        | str            | 0       | 0           | 3      | ND                               | ND (27008); No (2126); Yes (866)                                                                                                                               |     |      |        |       |      |          |
| bounce_flag           | int64          | 0       | 0           | 2      | 1                                |                                                                                                                                                                | 0   | 0    | 0      | 0     | 1    | 0.189067 |