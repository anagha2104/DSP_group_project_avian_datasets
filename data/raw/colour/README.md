# Illustrations of handbook of the birds of the world: Data sets of RGB values and color classification of bird

[https://doi.org/10.5061/dryad.70rxwdc6s](https://doi.org/10.5061/dryad.70rxwdc6s)

## Description of the data and file structure

We compiled two datasets comprising a total of 18,757 illustrations representing 10,290 species, including 5,288 subspecies (belonging to 2,746 species), 2,618 sexually dimorphic species, 456 species with multiple color morphs, and 231 species with different age classes (adult, juvenile, or immature). One dataset consists of the RGB data extracted from bird plates originally published in the *Handbook of the Birds of the World Alive* (HBW Alive) (del Hoyo et al. 2019), which helps future researchers to selectively choose and process raw data based on research needs. These electronic illustrations were downloaded in 2019, when they were still publicly available. Although *Handbook of the Birds of the World Alive* was later integrated into *Birds of the World* ([https://doi.org/10.2173/bow](https://doi.org/10.2173/bow)) (Billerman et al. 2025), our study relies on the earlier version. We here acknowledge the editors, illustrators, and publishers for making these resources accessible. The second dataset comprises color proportion data uniformly classified into 24 color categories. We believe these datasets will facilitate in-depth studies addressing a wide range of scientific questions.

### Files and variables

#### File: Data_S1.zip

├─ RGB_values.zip
│   └─ RGB_values
├─ RGB_values_for_color_classification.csv
├─ Information_for_Illustrations_and_proportion_of_24_colors.csv
└─ Extract_rgb.py

**Description:** 

**Identity and Size:** RGB_values.zip, 80.9 MB, this compressed file contains 1,000-pixel RGB data of 18,757 illustrations in .csv format. Each file is named according to the “Illustration_id” in “Information_for_Illustrations_and_proportion_of_24_colors.csv”.

RGB_values_for_color_classification.csv, 906 bytes. This file contains the information of RGB values of each 24 color categories.

Information_for_Illustrations_and_proportion_of_24_colors.csv, 5.2 MB. This file includes information on species (common and scientific names), subspecies, morphology, sex, credit of illustration, and the proportion of 24 color categories for 18,757 illustrations.

Extract_rgb.py, 4 KB, this is the code for extracting RGB data from illustrations.

**Format and storage mode:** All files are in comma-separated values files (.csv) format, packaged in a compressed folder (.zip), including 18,757 individual .csv files, and one Python code file (.py).

**Header information:** Variable names and descriptions for the file RGB_values.zip. Each file in the zip archive is named according to the Illustration_id, which matches the Illustration_id column in Information_for_Illustrations_and_proportion_of_24_colors.csv.

| Variable | Description                  |
| :------- | :--------------------------- |
| r        | R values of RGB color space. |
| g        | G values of RGB color space. |
| b        | B values of RGB color space. |

Variable names and descriptions for the file RGB_values_for_color_classification.csv.

| Variable             | Description                                                                                                                                                          |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Color classification | We used the K-means clustering algorithm to cluster the colors of avian plumage to 24 categories. |
| R                    | R values of each color category. |
| G                    | G values of each color category. |
| B                    | B values of each color category. |
| Colors               | The variable colors represent the categorical classification of plumage coloration, with each entry corresponding to a specific color name (e.g., blue, red, green). |

Variable names and descriptions for the file Information_for_Illustrations_and_proportion_of_24_color.csv.

| Variable          | Description                                                                                                                                                                                         |
| :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Com\_name         | Common name of each species, following the nomenclature used in HBW Alive (*Handbook of the Birds of the World Alive*). |
| Sci\_name         | Scientific name of each species (genus and species). |
| Subspecies        | Subspecies name, if applicable; otherwise refers to the nominate form. |
| Illustration\_id  | Unique identifier for each individual bird illustration in the dataset. |
| Credit            | Illustration artist. |
| Sex               | The sex represented in the illustration: “Male”, “Female”, or "Monomorphic" (used for species with no apparent sexual dichromatism). |
| Age               | Age class depicted in the illustration, e.g., “Adult”, “Juvenile”, or “Not specified” if unclear. |
| Morph             | Plumage morph (e.g., light or dark morph) if illustrated and available; “None” if species has no recognized morphs. |
| Order             | Taxonomic order based on 2019 ebird taxonomy. |
| Family            | Taxonomic family based on 2019 ebird taxonomy. |
| Group             | Common taxonomic grouping (e.g., “Ducks, Geese, and Waterfowl”, “Pigeons and Doves”) based on widely used English names for ease of interpretation. |
| Color1 to color24 | Proportion of each color category derived from RGB clustering of the illustration; see Table 1 for definitions of color categories. Values range from 0 to 1 and sum to 1 across the 24 categories. |

**Alphanumeric attributes:** Mixed.

**Special characters/fields:** Missing information was classified as “NA”.

For more details, see support information.

## Code/software

All data can be opened by software Excel, R program, Python, or any software for .csv files.

## Access information

Data was derived from the following sources: *Handbook of the Birds of the World Alive (HBW)*

**Copyright and Usage Notice**

The plumage color data used in this study were extracted from illustrations originally published in the *Handbook of the Birds of the World Alive* (del Hoyo et al. 2019) series in 2019, when the materials were publicly accessible.

No original images were reproduced or distributed; only RGB values and categorical color information were obtained.

The Cornell Lab of Ornithology manages electronic copyright for *Birds of the World* scientific illustrations, while Lynx Edicions holds print copyright. We encourage readers to consult the current usage guidelines of *Birds of the World* (Billerman et al. 2025) for any further applications. To obtain the rights to publish the assets in print, users should refer to Lynx Edicions.

## Related Publication

We warmly welcome the community to utilize this dataset for research and conservation purposes. We are also open to various forms of academic collaboration—feel free to reach out if you have any inquiries or collaborative proposals.

The data presented here support the findings of our study published in *Ecology*. For a detailed description of the data collection and color classification methodology, please refer to:\
Han, Y., Huang, W., & Liu, Y. (2026). Illustrations of Handbook of the Birds of the World: Datasets of RGB values and color classification of birds. *Ecology*, 107(2), e70320. [https://doi.org/10.1002/ecy.70320](https://doi.org/10.1002/ecy.70320)
