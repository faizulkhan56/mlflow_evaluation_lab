# Kaggle Dataset Integration Setup

This guide explains how to set up Kaggle API credentials for downloading the Wine Quality dataset.

## Overview

The project supports downloading the Wine Quality dataset from:
1. **Kaggle API** (if credentials are provided) - Recommended
2. **UCI ML Repository** (fallback) - Automatic fallback

## Prerequisites

1. Kaggle account: [https://www.kaggle.com/](https://www.kaggle.com/)
2. Kaggle API token

## Getting Your Kaggle API Token

1. Log in to your Kaggle account
2. Go to your account settings: [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
3. Scroll down to the "API" section
4. Click "Create New Token"
5. This will download a file named `kaggle.json` containing:
   ```json
   {
     "username": "your-username",
     "key": "your-api-key"
   }
   ```

## Setup Methods

### Method 1: Environment Variables (Recommended for Docker)

#### Local Development

**Linux/Mac:**
```bash
export KAGGLE_USERNAME=your-username
export KAGGLE_KEY=your-api-key
```

**Windows (PowerShell):**
```powershell
$env:KAGGLE_USERNAME="your-username"
$env:KAGGLE_KEY="your-api-key"
```

**Windows (CMD):**
```cmd
set KAGGLE_USERNAME=your-username
set KAGGLE_KEY=your-api-key
```

#### Docker Compose

1. Create a `.env` file in the project root:
   ```bash
   KAGGLE_USERNAME=your-username
   KAGGLE_KEY=your-api-key
   ```

2. Update `docker-compose.yml` to use environment variables:
   ```yaml
   mlflow-app:
     environment:
       - KAGGLE_USERNAME=${KAGGLE_USERNAME}
       - KAGGLE_KEY=${KAGGLE_KEY}
   ```

3. Start services:
   ```bash
   docker-compose up -d
   ```

### Method 2: Kaggle Config File (Local Development)

1. Install Kaggle package:
   ```bash
   pip install kaggle
   ```

2. Create Kaggle config directory:
   ```bash
   mkdir -p ~/.kaggle  # Linux/Mac
   # or
   mkdir %USERPROFILE%\.kaggle  # Windows
   ```

3. Copy your `kaggle.json` to the config directory:
   ```bash
   cp kaggle.json ~/.kaggle/kaggle.json  # Linux/Mac
   # or
   copy kaggle.json %USERPROFILE%\.kaggle\kaggle.json  # Windows
   ```

4. Set proper permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Method 3: Direct Script Execution (One-time Setup)

Create a script `setup_kaggle.sh` (Linux/Mac) or `setup_kaggle.bat` (Windows):

**Linux/Mac (`setup_kaggle.sh`):**
```bash
#!/bin/bash
export KAGGLE_USERNAME=your-username
export KAGGLE_KEY=your-api-key
python lab6_wine_quality_autologging.py
```

**Windows (`setup_kaggle.bat`):**
```batch
@echo off
set KAGGLE_USERNAME=your-username
set KAGGLE_KEY=your-api-key
python lab6_wine_quality_autologging.py
```

## Usage

### Without Kaggle Credentials

The scripts will automatically fall back to UCI ML Repository:
```bash
docker-compose exec mlflow-app python lab6_wine_quality_autologging.py
```

### With Kaggle Credentials

If credentials are set, the scripts will use Kaggle API:
```bash
# Set environment variables first
export KAGGLE_USERNAME=your-username
export KAGGLE_KEY=your-api-key

# Then run the script
docker-compose exec mlflow-app python lab6_wine_quality_autologging.py
```

## Dataset Information

- **Kaggle Dataset**: `uciml/red-wine-quality-cortez-et-al-2009`
- **UCI ML Repository**: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **File**: `winequality-red.csv`
- **Location**: `data/winequality-red.csv` (after download)

## Security Notes

⚠️ **Important Security Considerations:**

1. **Never commit credentials to Git:**
   - Add `.env` to `.gitignore`
   - Never commit `kaggle.json` or files containing API keys

2. **Use environment variables:**
   - Prefer environment variables over hardcoded credentials
   - Use `.env` files for local development (and add to `.gitignore`)

3. **Docker secrets (Production):**
   - For production, use Docker secrets or secret management systems
   - Don't hardcode credentials in docker-compose.yml

4. **Rotate keys regularly:**
   - If credentials are exposed, regenerate them in Kaggle settings

## Troubleshooting

### Issue: "Kaggle package not installed"
**Solution:** Install kaggle package:
```bash
pip install kaggle
```

### Issue: "401 Unauthorized"
**Solution:** Check your credentials are correct and API token is valid

### Issue: "Dataset not found"
**Solution:** Verify the dataset name: `uciml/red-wine-quality-cortez-et-al-2009`

### Issue: "Permission denied" (Linux/Mac)
**Solution:** Set correct permissions on kaggle.json:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Script falls back to UCI
**Solution:** This is normal if:
- Kaggle credentials are not set
- Kaggle package is not installed
- Kaggle API call fails

The fallback to UCI ensures the script always works.

## Testing

Test Kaggle integration:
```bash
# Test with credentials
export KAGGLE_USERNAME=your-username
export KAGGLE_KEY=your-api-key
python -c "from dataset_loader import load_wine_quality_dataset; df = load_wine_quality_dataset(); print(f'Loaded {len(df)} rows')"
```

## Additional Resources

- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)

