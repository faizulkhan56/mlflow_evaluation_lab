"""
Dataset Loader Helper
Handles downloading Wine Quality dataset from Kaggle or UCI ML Repository
"""
import os
import pandas as pd
import zipfile
import shutil

def load_wine_quality_dataset(data_file="data/winequality-red.csv"):
    """
    Load Wine Quality Red dataset.
    Tries in this order:
    1. Local file if exists
    2. Kaggle API (if credentials available)
    3. UCI ML Repository (fallback)
    
    Args:
        data_file: Path to save/load the dataset
        
    Returns:
        pandas.DataFrame: Wine quality dataset
    """
    # If file already exists, load it
    if os.path.exists(data_file):
        print(f"Loading dataset from {data_file}...")
        return pd.read_csv(data_file, sep=";")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Try Kaggle API first
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if kaggle_username and kaggle_key:
        try:
            print("Attempting to download from Kaggle...")
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Initialize Kaggle API
            api = KaggleApi()
            
            # Set Kaggle credentials
            os.environ["KAGGLE_USERNAME"] = kaggle_username
            os.environ["KAGGLE_KEY"] = kaggle_key
            
            # Authenticate
            api.authenticate()
            
            # Download dataset
            dataset = "uciml/red-wine-quality-cortez-et-al-2009"
            zip_file = "red-wine-quality-cortez-et-al-2009.zip"
            
            print(f"Downloading {dataset} from Kaggle...")
            api.dataset_download_files(
                dataset,
                path="data",
                unzip=True
            )
            
            # Find the CSV file (Kaggle might extract to different location)
            possible_paths = [
                "data/winequality-red.csv",
                "data/winequalityred.csv",
                "winequality-red.csv",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    # Move to standard location if needed
                    if path != data_file:
                        shutil.move(path, data_file)
                    print(f"Dataset downloaded from Kaggle and saved to {data_file}")
                    return pd.read_csv(data_file, sep=";")
            
            # If zip file exists, try to extract it
            zip_path = os.path.join("data", zip_file)
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("data")
                # Clean up zip file
                os.remove(zip_path)
                
                # Try to find CSV again
                for path in possible_paths:
                    if os.path.exists(path):
                        if path != data_file:
                            shutil.move(path, data_file)
                        print(f"Dataset extracted from Kaggle zip and saved to {data_file}")
                        return pd.read_csv(data_file, sep=";")
            
            print("Warning: Kaggle download completed but CSV file not found. Falling back to UCI...")
            
        except ImportError:
            print("Kaggle package not installed. Falling back to UCI ML Repository...")
        except Exception as e:
            print(f"Kaggle download failed: {str(e)}. Falling back to UCI ML Repository...")
    else:
        print("Kaggle credentials not found. Using UCI ML Repository...")
    
    # Fallback to UCI ML Repository
    print("Downloading from UCI ML Repository...")
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(data_url, sep=";")
    data.to_csv(data_file, index=False)
    print(f"Dataset downloaded from UCI and saved to {data_file}")
    
    return data

