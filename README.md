# Utilizing large language models for semantic analysis in psychological measurement: An examination of subjective time and depression scales

Thiago Augusto de Souza Bonifácio (1*), Rodrigo da Motta Cabral de Carvalho (1) , and André Mascioli Cravo (2)


1 Graduate Program in Neuroscience and Cognition, Federal University of ABC, São Paulo, Brazil
2 Center for Mathematics, Computing and Cognition, Federal University of ABC, São Paulo, Brazil

## Overview

This repository contains the implementation and datasets for the paper "Utilizing large language models for semantic analysis in psychological measurement: An examination of subjective time and depression scales". This study uses large language models, specifically the Sentence-T5 (ST5), to identify redundancy and overlap in self-report psychological measures. By analyzing semantic similarities between scale items and construct definitions, the study aims to improve the selection and use of existing measures. The analysis found significant overlap among some scales, suggesting redundancy that could impact research validity. The results demonstrate that LLMs can help refine psychological assessments, making research more efficient and rigorous. Further studies are needed to validate these findings across different psychological constructs.

## Table of Contents

1. [Installation](#installation)
2. [Data Description](#data-description)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)



## Installation

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/rodrigo-motta/brain-ising-gnn.git
    cd brain-ising-gnn
    ```

2. Create a enviroment

    ```bash
    conda create -n time_llm python=3.9
    ``` 

3. Install the required packages (the working directory should be the project):
    ```bash
    pip install -r requirements.txt
    ```

## Data Description

1.	Subjective Time Scales: The data for subjective time scales were gathered through a systematic literature search across databases like PsycNET, PubMed, Scopus, and Web of Science. This search identified 30 self-report scales related to subjective time, published between 1994 and 2023, focusing on various constructs like time perception, temporal orientation, and time perspective.

2.	Depression Scales: The depression scales included were well-established measures, such as the Beck Depression Inventory-II, Inventory of Depressive Symptomatology, Montgomery-Asberg Depression Rating Scale, and others. These scales were used to compare similarities and assess potential content overlap.

### Data Access

Under request.

## Usage

### Fundamentals

All the analysis is done in the notebook section that utilizes utils.py and dataset.py.

1. dataset.py: A class to load, preprocess, and manipulate textual data from a given module, specifically focusing on scales and sentences. This class provides methods to remove stopwords, lemmatize text, concatenate scales, and scramble sentences within the scales.

2. utils.py: All functions utilized in the notebooks for analysis (e.g. scramble, get embeddings, cluster, ...)

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes. For significant changes, please open an issue to discuss what you would like to change.

## License 

This project is licensed under the MIT License.

## Acknowledgments

This study was supported by the São Paulo Research Foundation (FAPESP) under various grants.
