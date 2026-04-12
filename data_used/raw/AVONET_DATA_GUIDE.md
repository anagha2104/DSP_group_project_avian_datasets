# ⭐ AVONET DATASET — FILE CONTENTS SUMMARY

## PURPOSE

This workspace contains the AVONET bird trait dataset along with taxonomy mappings, metadata, and phylogenetic trees used for ecomorphological and evolutionary analyses.

---

## CORE DATA FILES

### 📁 AVONET_Raw_Data.csv
- Contains bird morphological and ecological traits
- Includes multiple taxonomy labels (Species1/2/3 etc.)
- Contains Avibase IDs linking taxonomies
- PRIMARY FILE for most analyses



### 📁 AVONET1_BirdLife.csv
- Trait data aligned to BirdLife taxonomy
- BirdLife species and classification system



### 📁 AVONET2_eBird.xlsx  (Excel file)
- Trait data aligned to eBird taxonomy
- Species, order, and family follow eBird system



### 📁 AVONET3_BirdTree.xlsx (Excel file)
- Trait data aligned to BirdTree taxonomy
- Names matched for phylogenetic compatibility


---


## MAPPING & METADATA

### 🔗 BirdLife_BirdTree_Taxonomy_Relation.csv

- Crosswalk between BirdLife and BirdTree taxonomies
- Maps species names across the two systems
- Use when converting between taxonomies



### 🔗 Source_Details.csv
- Detailed provenance of each trait column
-Original data sources
-publication and dataset information
-Use for citations and data transparency



### 🔗 Trait_Database_History.csv
- Historical summary of contributing trait databases
- Authors, years, and species coverage
- Helps understand how AVONET was compiled



### 🔗 Family_Specimen_Stats.csv
- Specimen coverage statistics by bird family
- Number of species and sampling proportions
- Useful for detecting sampling bias

---

## PHYLOGENETIC TREES

### 🌳 Phylogeny_Hackett_Species_Tree.nex
- Species-level evolutionary tree
- Contains relationships among bird species
- Includes branch lengths (evolutionary time)
- PRIMARY tree for phylogenetic analyses



### 🌳 Phylogeny_BirdLife_Family_Tree.nex
- Family-level evolutionary tree
- Shows relationships among bird families
- Too coarse for species-level studies
- Use only for high-level comparisons


---

## Data Source

All AVONET data used in this workspace were obtained from the Open Traits Network:
https://opentraits.org/datasets/avonet.html .
AVONET is a global bird trait database providing morphological, ecological, and taxonomic information for over 11,000 bird species.

---

## IMPORTANT NOTES

- Species names must exactly match when using phylogenetic trees.
- Do NOT mix taxonomies without using the crosswalk file.
- Some species may be missing from the phylogenetic tree — always verify overlap.

---













