# Avian Functional Trait Database Explorer

## Objective

Design and implement a reproducible data ingestion, analysis, and visualisation pipeline to explore large avian trait databases, with the goal of understanding ecomorphological relationships, data quality issues, and variability across avian taxa.

---

## Tasks

- Identify and collect bird traits from various datasets  
- Harmonise bird trait datasets from multiple sources  
- Perform exploratory data analysis  
- Compute summary statistics and exploratory relationships between functional traits and ecological roles.  
- Build an interactive dashboard that enables filtering, comparison, and  visual exploration of trait distributions and phylogenetic correlations

---
## Plan
- Identify which system of Bird ID to use to aminatin consistency between systems. Merge continuous variables using averages in case of conflicts. If measures of variation is available, take that into account as well. Merge catergorical variables by using the most complete database and fill the missing gaps from other databases. In case of conflict, prefer most recent databse.
- Once we have a mega data sheet with all avaiable data for each species and a metadata sheet mapping different bird IDs and common names to a single chosen Bird ID, we will use this data to analyse and look for patterns in the data.
- We will combine the Bird data and photos soucred from the internet to display an interactive dashboard which displays pictures and descriptors for each species and where it is loctaed. We will also have a section of the dashboard where viewers can choose to see relationships between different variables for the birds.
---
## Project Structure
```text
/
│
├── data/               # Raw and processed datasets
│   ├── raw/            # Raw data files
│   └── preprocessed/   # Cleaned / processed data
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── src/                # Source code (pipeline, preprocessing, models)
├── results/            # Figures, outputs, and reports
├── requirements.txt    # Python dependencies
└── README.md
```
---

## Contributors

|     Name        |  GitHub Username 
|-----------------|-------------------|
| Anil Thakor     | [AnilThakor23](https://github.com/AnilThakor23) 
| Anagha          | [anagha2104](https://github.com/anagha2104) 
| Om Kekan        | [Omiiii1407](https://github.com/Omiiii1407) 
| Pratyush Biswal | [Pratyush-B-2DS8](https://github.com/Pratyush-B-2DS8) 

---
