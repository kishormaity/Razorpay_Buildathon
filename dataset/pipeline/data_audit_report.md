# IEEE-CIS Data Audit Report

Generated on: 2026-08-23 18:46:04

This report summarizes the structure, volume, missingness, and cardinality of key fields in the raw IEEE-CIS dataset.

---

## 1. High-Level Data Volumes

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Transactions** | 590,540 | Total rows in `train_transaction.csv` |
| **Total Identities** | 144,233 | Total rows in `train_identity.csv` |
| **Identity Overlap** | 118,666 (20.09%) | Transactions linked with identity records |
| **Fraud Cases** | 20,663 (3.499%) | Target label (`isFraud == 1`) prevalence |
| **Temporal Span** | 182.00 days | Range of `TransactionDT` seconds |

> [!IMPORTANT]
> **`TransactionDT` is a relative time variable representing elapsed seconds from an anonymized reference point. It should primarily be used for temporal ordering and relative-time feature engineering. Any calendar-date projection is artificial and should not be interpreted as the real transaction date.**

---

## 2. Complete Feature Inventory (Grouped summary)

The dataset contains **394 columns** in transactions and **41 columns** in identity attributes (totaling **434 unique features** merged on `TransactionID`).

### Core Columns Summary:

| Feature | Data Type | Missing Count | Missing % | Recommendation / Role |
| :--- | :--- | :--- | :--- | :--- |
| `TransactionID` | `int64` | 0 | 0.00% | Primary Join Key / Identifier |
| `isFraud` | `int64` | 0 | 0.00% | Target variable (**DO NOT USE AS MODEL INPUT**) |
| `TransactionDT` | `int64` | 0 | 0.00% | Temporal Ordering / Behavioral Features |
| `TransactionAmt` | `float64` | 0 | 0.00% | Transaction Amount (Numerical) |
| `ProductCD` | `str` | 0 | 0.00% | Channel/Product Code (Categorical) |
| `card1` | `int64` | 0 | 0.00% | Card/Group Identifier (**Entity candidate: CARD**) |
| `card4` | `str` | 1,577 | 0.27% | Card Brand (Categorical) |
| `card6` | `str` | 1,571 | 0.27% | Card Type (Categorical) |
| `addr1` | `float64` | 65,706 | 11.13% | Billing Region (**Entity candidate: REGION**) |
| `addr2` | `float64` | 65,706 | 11.13% | Billing Country (Categorical) |
| `P_emaildomain` | `float64` | 94,456 | 15.99% | Purchaser Email Domain (**Entity: EMAIL_DOMAIN**) |
| `R_emaildomain` | `float64` | 453,249 | 76.75% | Recipient Email Domain (Categorical) |
| `id_02` | `float64` | 3,361 | 2.33% | Identity ID (**Entity candidate: DEVICE**) |
| `DeviceInfo` | `str` | 25,567 | 17.73% | Hardware Model name (Categorical / Entity) |

### Large Feature Blocks Summary:

| Feature Block | Count | Type | Avg Missing % | Role |
| :--- | :--- | :--- | :--- | :--- |
| **`C1` - `C14`** | 14 | Numerical | 0.00% | Counting features (e.g., card counts) |
| **`D1` - `D15`** | 15 | Numerical | 58.15% | Timedeltas / relative duration features |
| **`M1` - `M9`** | 9 | Categorical | 49.92% | Match features (e.g., names matching) |
| **`V1` - `V339`** | 339 | Numerical | 43.04% | Engineered Vesta features (Rank/Match) |

---

## 3. Missingness Analysis

Columns are grouped by missingness thresholds:

### Critical Missingness (>95% missing)
Total columns: **9**
> [!NOTE]
> Columns with >95% missing values are generally excluded from ML models unless they represent highly specific fraud signals (like certain advanced browser/network flags).

### High Missingness (50% - 95% missing)
Total columns: **177**
> [!WARNING]
> These fields (such as identity attributes) are only present when specific devices or checks are logged. We must not drop them automatically; the **absence** of these features is itself a high-signal indicator (as fraud rate varies on checks).

### Moderate/Low Missingness (<50% missing)
Total columns: **249**
* Most core transaction attributes, amount, product types, and `card1` fall here and are highly usable.

---

## 4. Numerical Feature Analysis

Key numerical stats calculated on full records:

| Feature | Min | Max | Mean | Std | Missing % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TransactionAmt` | 0.25 | 31937.39 | 135.03 | 239.16 | 0.00% |
| `TransactionDT` | 86,400 | 15,811,131 | 7,372,311.3 | 4,617,223.6 | 0.00% |
| `card1` | 1000 | 18396 | 9898.7 | 4901.2 | 0.00% |

---

## 5. Categorical Feature Analysis & Fraud Rates

Detailed breakdown of top values by transactional volume and target fraud rate.

### `ProductCD` (Product Type/Channel)
| Value | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| `W` | 439,670 | 8,969 | 2.040% |
| `C` | 68,519 | 8,008 | 11.687% |
| `R` | 37,699 | 1,426 | 3.783% |
| `H` | 33,024 | 1,574 | 4.766% |
| `S` | 11,628 | 686 | 5.900% |

### `card4` (Card Brand)
| Value | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| `visa` | 384,767 | 13,373 | 3.476% |
| `mastercard` | 189,217 | 6,496 | 3.433% |
| `american express` | 8,328 | 239 | 2.870% |
| `discover` | 6,651 | 514 | 7.728% |

### `card6` (Card Type)
| Value | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| `debit` | 439,938 | 10,674 | 2.426% |
| `credit` | 148,986 | 9,950 | 6.678% |
| `debit or credit` | 30 | 0 | 0.000% |
| `charge card` | 15 | 0 | 0.000% |

### `DeviceType` (Device Class)
| Value | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| `desktop` | 85,165 | 5,554 | 6.521% |
| `mobile` | 55,645 | 5,657 | 10.166% |

### `P_emaildomain` (Purchaser Email - Top 10)
| Value | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| `gmail.com` | 228,355 | 9,943 | 4.354% |
| `yahoo.com` | 100,934 | 2,297 | 2.276% |
| `hotmail.com` | 45,250 | 2,396 | 5.295% |
| `anonymous.com` | 36,998 | 859 | 2.322% |
| `aol.com` | 28,289 | 617 | 2.181% |
| `comcast.net` | 7,888 | 246 | 3.119% |
| `icloud.com` | 6,267 | 197 | 3.143% |
| `outlook.com` | 5,096 | 482 | 9.458% |
| `msn.com` | 4,092 | 90 | 2.199% |
| `att.net` | 4,033 | 30 | 0.744% |

### `R_emaildomain` (Recipient Email - Top 10)
| Value | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| `gmail.com` | 57,147 | 6,811 | 11.918% |
| `hotmail.com` | 27,509 | 2,140 | 7.779% |
| `anonymous.com` | 20,529 | 598 | 2.913% |
| `yahoo.com` | 11,842 | 610 | 5.151% |
| `aol.com` | 3,701 | 129 | 3.486% |
| `outlook.com` | 2,507 | 414 | 16.514% |
| `comcast.net` | 1,812 | 21 | 1.159% |
| `yahoo.com.mx` | 1,508 | 16 | 1.061% |
| `icloud.com` | 1,398 | 180 | 12.876% |
| `msn.com` | 852 | 1 | 0.117% |

---

## 6. Target Analysis & Class Imbalance

* **Total Non-Fraud (`isFraud == 0`)**: 569,877 (96.501%)
* **Total Fraud (`isFraud == 1`)**: 20,663 (3.499%)
* **Class Imbalance Ratio**: 1 : 27

> [!CAUTION]
> **Leakage Check**: The target variable `isFraud` has a direct 1:1 mapping with the outcome. Under no circumstances should `isFraud` be loaded as a training feature.

---

## 7. Temporal Analysis

We evaluated `TransactionDT` relative to hour and day intervals:

### Fraud Rate by Relative Hour of Day:
| Relative Hour | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| Hour `0` | 37,795 | 1,186 | 3.138% |
| Hour `1` | 32,797 | 1,027 | 3.131% |
| Hour `2` | 26,732 | 1,002 | 3.748% |
| Hour `3` | 20,802 | 797 | 3.831% |
| Hour `4` | 14,839 | 770 | 5.189% |
| Hour `5` | 9,701 | 682 | 7.030% |
| Hour `6` | 6,007 | 467 | 7.774% |
| Hour `7` | 3,704 | 393 | 10.610% |
| Hour `8` | 2,591 | 241 | 9.301% |
| Hour `9` | 2,479 | 223 | 8.996% |
| Hour `10` | 3,627 | 193 | 5.321% |
| Hour `11` | 6,827 | 265 | 3.882% |
| Hour `12` | 12,451 | 379 | 3.044% |
| Hour `13` | 20,315 | 465 | 2.289% |
| Hour `14` | 28,328 | 686 | 2.422% |
| Hour `15` | 33,859 | 860 | 2.540% |
| Hour `16` | 38,698 | 1,142 | 2.951% |
| Hour `17` | 40,723 | 1,284 | 3.153% |
| Hour `18` | 41,639 | 1,467 | 3.523% |
| Hour `19` | 42,115 | 1,463 | 3.474% |
| Hour `20` | 41,782 | 1,432 | 3.427% |
| Hour `21` | 41,641 | 1,416 | 3.400% |
| Hour `22` | 41,139 | 1,345 | 3.269% |
| Hour `23` | 39,949 | 1,478 | 3.700% |

### Fraud Rate by Day of Week:
| Day Index | Total Transactions | Fraud Transactions | Fraud Rate |
| :--- | :--- | :--- | :--- |
| Day `0` | 86,377 | 3,211 | 3.717% |
| Day `1` | 98,502 | 3,550 | 3.604% |
| Day `2` | 79,834 | 2,963 | 3.711% |
| Day `3` | 70,223 | 2,503 | 3.564% |
| Day `4` | 85,433 | 2,687 | 3.145% |
| Day `5` | 84,815 | 2,803 | 3.305% |
| Day `6` | 85,356 | 2,946 | 3.451% |

---

## 8. Entity Analysis (Cardinality)

* **`CARD` (card1)**: 13,553 unique identifiers.
* **`DEVICE` (id_02)**: 115,655 unique identities.
* **`DEVICE` (DeviceInfo)**: 1,786 unique hardware models.
* **`REGION` (addr1)**: 332 billing regions.
* **`EMAIL_DOMAIN` (P_emaildomain)**: 59 domains.

---

## 9. Graph Connectivity Analysis

We mapped links across entities to evaluate shared risk topology:

### Card-to-Entity Links:
* **Average Txs per Card**: 43.57 (Max: 14,932)
* **Average Unique Devices per Card**: 10.32 (Max: 9,291)
* **Average Unique Regions per Card**: 2.77 (Max: 64)
* **Average Unique Emails per Card**: 2.82 (Max: 43)

### Device-to-Entity Links:
* **Average Unique Cards per Device**: 1.21 (Max: 6)
* **Average Txs per Device**: 1.22 (Max: 11)

> [!TIP]
> **Abuse Ring Topology**: The high maximum values (e.g. some device identifiers linked to multiple unique cards, and some cards linked to multiple regions/emails) confirm that the dataset contains the graph structure needed to run community detection (e.g. Leiden or Louvain) for abuse ring identification.

---

## 10. Leakage Analysis

We verified the timeline of variables:
1. **Target variable (`isFraud`)**: Excluded from model features.
2. **Transaction telemetry**: (`TransactionAmt`, `card1-card6`, `addr1-addr2`, email domains) are known at transaction time.
3. **Identity parameters**: (`id_01` to `id_38`, `DeviceInfo`) are collected at transaction time via device fingerprinting and are safe.
4. **Conclusion**: No future-event attributes or analyst actions are recorded in the raw transaction telemetry, avoiding runtime data leakage.

---

## 11. Full List of Dataset Columns

### Transaction Dataset Columns (394 columns):
```json
[
  "TransactionID",
  "isFraud",
  "TransactionDT",
  "TransactionAmt",
  "ProductCD",
  "card1",
  "card2",
  "card3",
  "card4",
  "card5",
  "card6",
  "addr1",
  "addr2",
  "dist1",
  "dist2",
  "P_emaildomain",
  "R_emaildomain",
  "C1",
  "C2",
  "C3",
  "C4",
  "C5",
  "C6",
  "C7",
  "C8",
  "C9",
  "C10",
  "C11",
  "C12",
  "C13",
  "C14",
  "D1",
  "D2",
  "D3",
  "D4",
  "D5",
  "D6",
  "D7",
  "D8",
  "D9",
  "D10",
  "D11",
  "D12",
  "D13",
  "D14",
  "D15",
  "M1",
  "M2",
  "M3",
  "M4",
  "M5",
  "M6",
  "M7",
  "M8",
  "M9",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
  "V7",
  "V8",
  "V9",
  "V10",
  "V11",
  "V12",
  "V13",
  "V14",
  "V15",
  "V16",
  "V17",
  "V18",
  "V19",
  "V20",
  "V21",
  "V22",
  "V23",
  "V24",
  "V25",
  "V26",
  "V27",
  "V28",
  "V29",
  "V30",
  "V31",
  "V32",
  "V33",
  "V34",
  "V35",
  "V36",
  "V37",
  "V38",
  "V39",
  "V40",
  "V41",
  "V42",
  "V43",
  "V44",
  "V45",
  "V46",
  "V47",
  "V48",
  "V49",
  "V50",
  "V51",
  "V52",
  "V53",
  "V54",
  "V55",
  "V56",
  "V57",
  "V58",
  "V59",
  "V60",
  "V61",
  "V62",
  "V63",
  "V64",
  "V65",
  "V66",
  "V67",
  "V68",
  "V69",
  "V70",
  "V71",
  "V72",
  "V73",
  "V74",
  "V75",
  "V76",
  "V77",
  "V78",
  "V79",
  "V80",
  "V81",
  "V82",
  "V83",
  "V84",
  "V85",
  "V86",
  "V87",
  "V88",
  "V89",
  "V90",
  "V91",
  "V92",
  "V93",
  "V94",
  "V95",
  "V96",
  "V97",
  "V98",
  "V99",
  "V100",
  "V101",
  "V102",
  "V103",
  "V104",
  "V105",
  "V106",
  "V107",
  "V108",
  "V109",
  "V110",
  "V111",
  "V112",
  "V113",
  "V114",
  "V115",
  "V116",
  "V117",
  "V118",
  "V119",
  "V120",
  "V121",
  "V122",
  "V123",
  "V124",
  "V125",
  "V126",
  "V127",
  "V128",
  "V129",
  "V130",
  "V131",
  "V132",
  "V133",
  "V134",
  "V135",
  "V136",
  "V137",
  "V138",
  "V139",
  "V140",
  "V141",
  "V142",
  "V143",
  "V144",
  "V145",
  "V146",
  "V147",
  "V148",
  "V149",
  "V150",
  "V151",
  "V152",
  "V153",
  "V154",
  "V155",
  "V156",
  "V157",
  "V158",
  "V159",
  "V160",
  "V161",
  "V162",
  "V163",
  "V164",
  "V165",
  "V166",
  "V167",
  "V168",
  "V169",
  "V170",
  "V171",
  "V172",
  "V173",
  "V174",
  "V175",
  "V176",
  "V177",
  "V178",
  "V179",
  "V180",
  "V181",
  "V182",
  "V183",
  "V184",
  "V185",
  "V186",
  "V187",
  "V188",
  "V189",
  "V190",
  "V191",
  "V192",
  "V193",
  "V194",
  "V195",
  "V196",
  "V197",
  "V198",
  "V199",
  "V200",
  "V201",
  "V202",
  "V203",
  "V204",
  "V205",
  "V206",
  "V207",
  "V208",
  "V209",
  "V210",
  "V211",
  "V212",
  "V213",
  "V214",
  "V215",
  "V216",
  "V217",
  "V218",
  "V219",
  "V220",
  "V221",
  "V222",
  "V223",
  "V224",
  "V225",
  "V226",
  "V227",
  "V228",
  "V229",
  "V230",
  "V231",
  "V232",
  "V233",
  "V234",
  "V235",
  "V236",
  "V237",
  "V238",
  "V239",
  "V240",
  "V241",
  "V242",
  "V243",
  "V244",
  "V245",
  "V246",
  "V247",
  "V248",
  "V249",
  "V250",
  "V251",
  "V252",
  "V253",
  "V254",
  "V255",
  "V256",
  "V257",
  "V258",
  "V259",
  "V260",
  "V261",
  "V262",
  "V263",
  "V264",
  "V265",
  "V266",
  "V267",
  "V268",
  "V269",
  "V270",
  "V271",
  "V272",
  "V273",
  "V274",
  "V275",
  "V276",
  "V277",
  "V278",
  "V279",
  "V280",
  "V281",
  "V282",
  "V283",
  "V284",
  "V285",
  "V286",
  "V287",
  "V288",
  "V289",
  "V290",
  "V291",
  "V292",
  "V293",
  "V294",
  "V295",
  "V296",
  "V297",
  "V298",
  "V299",
  "V300",
  "V301",
  "V302",
  "V303",
  "V304",
  "V305",
  "V306",
  "V307",
  "V308",
  "V309",
  "V310",
  "V311",
  "V312",
  "V313",
  "V314",
  "V315",
  "V316",
  "V317",
  "V318",
  "V319",
  "V320",
  "V321",
  "V322",
  "V323",
  "V324",
  "V325",
  "V326",
  "V327",
  "V328",
  "V329",
  "V330",
  "V331",
  "V332",
  "V333",
  "V334",
  "V335",
  "V336",
  "V337",
  "V338",
  "V339"
]
```

### Identity Dataset Columns (41 columns):
```json
[
  "TransactionID",
  "id_01",
  "id_02",
  "id_03",
  "id_04",
  "id_05",
  "id_06",
  "id_07",
  "id_08",
  "id_09",
  "id_10",
  "id_11",
  "id_12",
  "id_13",
  "id_14",
  "id_15",
  "id_16",
  "id_17",
  "id_18",
  "id_19",
  "id_20",
  "id_21",
  "id_22",
  "id_23",
  "id_24",
  "id_25",
  "id_26",
  "id_27",
  "id_28",
  "id_29",
  "id_30",
  "id_31",
  "id_32",
  "id_33",
  "id_34",
  "id_35",
  "id_36",
  "id_37",
  "id_38",
  "DeviceType",
  "DeviceInfo"
]
```

