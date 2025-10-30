# PRRS-Pipeline-Demo
AI-driven pipeline for PRRS virus epidemiology and pathogen informatics
🦠 PRRSV-2 Variant Classification: A Mini-Phylodynamic Pipeline
I. Introduction
This repository, PRRS-Seq-Intro, serves as a demonstration of a complete, though simplified, computational biology pipeline for PRRSV-2 variant classification using viral sequence data. It integrates standard sequence analysis techniques with a machine learning for prediction (XGBoost/stacking) approach, setting the stage for more complex phylodynamic inference (BEAST) and farm-level modeling.
II. Core Objectives
1.	Reproducibility: Create a fully self-contained pipeline runnable on any standard Python environment (Windows/VS Code).
2.	Bioinformatics Competence: Use Biopython for robust sequence data handling.
3.	Machine Learning Integration: Implement a sequence feature-based classification model (XGBoost).
4.	Phylodynamic Introduction: Model the initial steps of data preparation for advanced phylodynamics analysis.
III. Methodology
The pipeline (pipeline_prrs.py) follows a standard data flow:
1.	Sequence Acquisition (Mock): Loading FASTA data, simulating the process of integrating data from sources like MSHMP or public GenBank.
2.	Alignment (Simulated): A placeholder for calling advanced alignment tools. In a full production environment, IQ-TREE would be used for phylogenomic tree inference, and MAFFT or Clustal Omega would handle the primary alignment.
3.	Feature Engineering: Extraction of simple features (GC content, sequence length) used as predictors.
4.	Classification: Training an XGBoost Classifier to predict the viral genotype. This process is a prerequisite for subsequent network analysis and epidemiological linkage studies.
IV. Data Structure and Setup
File	Description
data_mock.fasta.txt	Contains the 9 mock PRRSV-2 sequences used for classification.
pipeline_prrs.py	The main, single-script pipeline executing all steps.
requirements.txt	Lists all necessary Python dependencies (Biopython, scikit-learn, XGBoost, pandas, numpy).
.gitignore	Ensures environment files (.venv/) are not committed.

Setup Instructions
1.	Create and Activate Virtual Environment:
Bash
python -m venv .venv
# Activate (Windows/PowerShell): .\.venv\Scripts\Activate.ps1
# Activate (Linux/macOS): source .venv/bin/activate
2.	Install Dependencies:
Bash
pip install -r requirements.txt
3.	Execute Pipeline:
Bash
python pipeline_prrs.py
V. Results Summary
The pipeline successfully processed 9 sequences (3 per genotype), splitting the data into 6 training samples and 3 testing samples. The XGBoost model trained and reported an accuracy score of 0.3333 on the test set. This successfully validates the end-to-end data processing and machine learning integration logic.
VI. 🔑 Keywords
phylodynamics, PRRSV-2 variant classification, MSHMP, sequence analysis, farm-level modeling, phylodynamic inference (BEAST), IQ-TREE, network analysis, machine learning for prediction (XGBoost/stacking)
