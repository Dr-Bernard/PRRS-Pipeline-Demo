# -*- coding: utf-8 -*-
"""
PRRSV-2 Variant Classification: A Mini-Phylodynamic Pipeline Model

Senior Computational Biologist / ML Engineer Demonstration Script.
This script executes a mock end-to-end pipeline:
Sequence Acquisition -> Alignment (Simulated) -> Feature Extraction -> ML Classification (XGBoost)
"""
import os
import random
from Bio import SeqIO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# --- Configuration ---
FASTA_FILE = 'data_mock.fasta.txt'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def calculate_gc_content(seq):
    """Calculates the percentage of G and C nucleotides in a sequence."""
    if not seq:
        return 0
    return (seq.count('G') + seq.count('C')) / len(seq) * 100

def mock_alignment(seqs):
    """
    Step 2: Sequence Analysis & Alignment (Mock Simulation)
    In a full pipeline, this is where external tools like MAFFT or Clustal Omega
    (often called via Biopython's wrappers) would perform sequence alignment.
    For this self-contained script, we assume the input sequences are 'aligned'
    or simply pass the raw sequences through.
    """
    print(f"[STEP 2/5] Simulating alignment for {len(seqs)} sequences...")
    # In a real scenario, aligned sequences would be returned here.
    return seqs

def feature_engineering(records):
    """
    Step 3: Feature Engineering
    Extract simple sequence features and generate mock target/irrelevant variables.
    """
    print(f"[STEP 3/5] Extracting features and generating mock variables...")
    data = []
    for record in records:
        sequence = str(record.seq).upper()
        header_parts = record.id.split('_')
        
        # --- Real Features ---
        gc_content = calculate_gc_content(sequence)
        seq_length = len(sequence)
        
        # --- Mock Target Variable (Genotype Classification) ---
        # Infer class from header (A, B, C prefixes) for reproducibility
        genotype_class = 'Genotype_' + header_parts[2][0] 
        
        # --- Mock Irrelevant/Advanced Variable (Simulating Complex Output) ---
        # Simulating a complex, farm-level variable derived from a separate model (e.g., MSHMP)
        # Note: This variable is irrelevant for the ML model but demonstrates data integration.
        clinical_severity_score = random.uniform(1.0, 9.9) 
        
        data.append({
            'sequence_id': record.id,
            'gc_content': gc_content,
            'sequence_length': seq_length,
            'genotype_class': genotype_class,
            'clinical_severity_score': clinical_severity_score
        })
        
    return pd.DataFrame(data)

def machine_learning_classification(df):
    """
    Step 4: Machine Learning Classification (XGBoost)
    Predict the genotype class from the engineered sequence features.
    """
    print(f"[STEP 4/5] Training XGBoost Classifier...")
    
    # Prepare Data
    X = df[['gc_content', 'sequence_length']]
    y_raw = df['genotype_class']
    
    # Encode Target Variable (Genotype_A, Genotype_B, Genotype_C -> 0, 1, 2)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    
    # Train Model (Using standard parameters)
    # The XGBoost model demonstrates competence in predictive analytics.
    model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='mlogloss',
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Training Complete ---")
    print(f"XGBoost Classifier Accuracy Score: {accuracy:.4f}")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")
    print("-----------------------------------")
    
    return accuracy

def main():
    """Main function to run the viral sequence analysis pipeline."""
    print("🧬 PRRS-Seq-Intro Pipeline Initiated...")
    
    # Step 1: Data Acquisition & Mock Data Loading
    print(f"[STEP 1/5] Loading mock sequences from {FASTA_FILE}...")
    try:
        # Biopython is essential for robust sequence handling
        sequence_records = list(SeqIO.parse(FASTA_FILE, "fasta"))
        print(f"Successfully loaded {len(sequence_records)} sequence records.")
        # Mocking the process of fetching public sequences (e.g., via NCBI's Entrez or local MSHMP/GISAID data)
        # For execution, we proceed with the local file.
    except FileNotFoundError:
        print(f"Error: {FASTA_FILE} not found. Ensure the file is in the current directory.")
        return
        
    # Step 2: Sequence Analysis & Alignment (Simulated)
    aligned_sequences = mock_alignment(sequence_records)
    
    # Step 3: Feature Engineering
    feature_df = feature_engineering(aligned_sequences)
    print("\n--- Extracted Features Sample ---")
    print(feature_df.head())
    print("---------------------------------")
    
    # Step 4: Machine Learning Classification
    accuracy = machine_learning_classification(feature_df)
    
    # Step 5: Output & Reproducibility
    print("\n✅ PRRS-Seq-Intro Pipeline Execution Complete!")
    print(f"Final Model Accuracy on Test Set: {accuracy:.4f}")
    print("The pipeline successfully demonstrated sequence handling, feature engineering, and ML classification.")
    print("This setup is fully reproducible with standard Python libraries on a Windows/VS Code environment.")
    
if __name__ == "__main__":
    main()