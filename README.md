# MedLink: De-Identified Patient Health Record Linkage

This repository contains data and code for the KDD'23 paper
titled [MedLink: De-Identified Patient Health Record Linkage](https://dl.acm.org/doi/10.1145/3580305.3599427).

You can also check out a 2-minute pitch talk on this project available
on [YouTube](https://www.youtube.com/watch?v=dqm3s6vYReI).

## Dependencies

```
python==3.8.18
torch==2.0.1
beir=2.0.0
```

## Repository Structure

- `data/`: Contains data and resource files
    - `mimic3/`: Data related to the MIMIC-III dataset
    - `resource/`: Resource files for medical codes
- `GloVe/`: Source code for GloVe
- `src/`: Source code for MedLink
    - `preprocess/`: Notebooks for data preprocessing
    - `dataset/`: Vocabulary, tokenizer, and data loader
    - `model/`: Model architecture
    - `run_bm25_neg_smp.py`: Script for BM25 hard negative sampling
    - `run_medlink.py`: Script for MedLink training and evaluation
    - `metrics.py`: Metrics for model evaluation
    - `helper.py`: Helper class for model training, evaluation, and inference
    - `utils.py`: Utility functions
    - `credentials.py`: [Optional] Credentials for Neptune.ai, including NEPTUNE_PROJECT, NEPTUNE_API_KEY, and NAME
- `run_medlink.sh`: Bash script for running run_medlink.py

## How to Reproduce

Follow these steps to reproduce the results:

1. Obtain the MIMIC-III dataset and place it under `data/mimic3/raw/`.
2. Run the following notebooks under src/preprocess in the specified order to prepare the data:
    1. Run `preprocess.ipynb`
    2. Run `split_data.ipynb`
    3. Run `glove.ipynb`
    4. Run `candidate_gen.ipynb`
3. Obtain BM25 hard negatives using `src/run_bm25_neg_smp.py`.
4. Train the MedLink model using `run_medlink.sh`.

## Citation

```
@inproceedings{10.1145/3580305.3599427,
    author = {Wu, Zhenbang and Xiao, Cao and Sun, Jimeng},
    title = {MedLink: De-Identified Patient Health Record Linkage},
    year = {2023},
    isbn = {9798400701030},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3580305.3599427},
    doi = {10.1145/3580305.3599427},
    abstract = {A comprehensive patient health history is essential for patient care and healthcare research. However, due to the distributed nature of healthcare services, patient health records are often scattered across multiple systems. Existing record linkage approaches primarily rely on patient identifiers, which have inherent limitations such as privacy invasion and identifier discrepancies. To tackle this problem, we propose linking de-identified patient health records by matching health patterns without strictly relying on sensitive patient identifiers. Our model MedLink solves two challenges faced with the patient linkage task: (1) the challenge of identifying the same patients based on data collected in different timelines as disease progression makes the record matching difficult, and (2) the challenge of identifying distinct health patterns as common medical codes dominate health records and overshadow the more informative low-prevalence codes. To address these challenges, MedLink utilizes bi-directional health prediction to predict future codes forwardly and past codes backwardly, thus accounting for the health progression. MedLink also has a prevalence-aware retrieval design to focus more on the low-prevalence but informative codes during learning. MedLink can be trained end-to-end and is lightweight for efficient inference on large patient databases. We evaluate MedLink against leading baselines on real-world patient datasets, including the critical care dataset MIMIC-III and a large health claims dataset. Results show that MedLink outperforms the best baseline by 4\% in top-1 accuracy with only 8\% memory cost. Additionally, when combined with existing identifier-based linkage approaches, MedLink can improve their performance by up to 15\%.},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages = {2672–2682},
    numpages = {11},
    keywords = {entity resolution, patient identification, patient deduplication, patient linkage, record linkage, electronic health record},
    location = {Long Beach, CA, USA},
    series = {KDD '23}
}
```