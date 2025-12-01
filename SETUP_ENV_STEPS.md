# Step-by-Step: Setup Kaggle API with .env File (Option 3)

## Your Credentials
- **Username:** `your-username`
- **API Key:** `your-api-key`

---

## Step 1: Create .env File

Create a file named `.env` in the project root directory (same folder as `docker-compose.yml`).

### On Windows:
1. Open your project folder: `C:\faizul-personal\mlflow_evaluation_lab`
2. Right-click → New → Text Document
3. Name it `.env` (make sure it starts with a dot)
4. Open it with Notepad or any text editor

### On Linux/Mac:
```bash
cd /path/to/mlflow_evaluation_lab
nano .env
# or
vim .env
```

### Add this content to .env file:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_REGISTRY_URI=http://mlflow-server:5000

# Kaggle API Credentials
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

**Important:** 
- No spaces around the `=` sign
- No quotes needed
- Save the file

---

## Step 2: Verify docker-compose.yml

The `docker-compose.yml` should already have these lines in the `mlflow-app` service section (around line 64-66):

```yaml
environment:
  - MLFLOW_TRACKING_URI=http://mlflow-server:5000
  - MLFLOW_REGISTRY_URI=http://mlflow-server:5000
  # Kaggle API credentials for downloading datasets (from .env file)
  - KAGGLE_USERNAME=${KAGGLE_USERNAME}
  - KAGGLE_KEY=${KAGGLE_KEY}
```

If these lines are commented out (start with `#`), uncomment them.

---

## Step 3: Stop Existing Containers (if running)

```bash
docker-compose down
```

---

## Step 4: Rebuild Containers

This is important because:
1. It installs the `kaggle` package
2. It picks up the new environment variables from `.env`

```bash
docker-compose build
```

**Expected output:**
```
Building mlflow-app ...
...
Successfully built ...
```

---

## Step 5: Start Services

```bash
docker-compose up -d
```

**Expected output:**
```
Creating network ...
Creating mlflow-postgres ...
Creating mlflow-server ...
Creating mlflow-app ...
Creating mlflow-api ...
```

---

## Step 6: Verify Services are Running

```bash
docker-compose ps
```

**Expected output:**
```
NAME                STATUS
mlflow-postgres     Up (healthy)
mlflow-server       Up (healthy)
mlflow-app          Up
mlflow-api          Up (healthy)
```

---

## Step 7: Verify Kaggle Credentials in Container

Check that environment variables are loaded:

```bash
docker-compose exec mlflow-app env | grep KAGGLE
```

**Expected output:**
```
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

If you see this, credentials are loaded correctly! ✅

---

## Step 8: Test Kaggle Integration

Run Lab 6 to test the Kaggle download:

```bash
docker-compose exec mlflow-app python lab6_wine_quality_autologging.py
```

**Expected output:**
```
Attempting to download from Kaggle...
Downloading uciml/red-wine-quality-cortez-et-al-2009 from Kaggle...
Dataset downloaded from Kaggle and saved to data/winequality-red.csv
Loading dataset from data/winequality-red.csv...
...
Random Forest (Autolog) Accuracy: 0.XXX
...
✓ Lab 6 (Autologging) completed successfully!
```

If you see "Downloading from Kaggle..." and "Dataset downloaded from Kaggle", it's working! 🎉

---

## Step 9: Run Lab 7 (Optional)

```bash
docker-compose exec mlflow-app python lab7_wine_quality_manual.py
```

**Expected output:**
```
Loading dataset from data/winequality-red.csv...
...
Random Forest (Manual) Accuracy: 0.XXX
...
✓ Lab 7 (Manual Logging) completed successfully!
```

---

## Step 10: Verify Dataset File

Check that the dataset was downloaded:

```bash
# From host machine
ls -lh data/winequality-red.csv  # Linux/Mac
dir data\winequality-red.csv     # Windows

# Or from container
docker-compose exec mlflow-app ls -lh data/winequality-red.csv
```

**Expected output:**
```
-rw-r--r-- 1 user user 75K ... data/winequality-red.csv
```

---

## Troubleshooting

### Problem: "Kaggle credentials not found"

**Solution:**
1. Check `.env` file exists in project root
2. Verify file content (no extra spaces, correct format)
3. Restart containers: `docker-compose restart mlflow-app`

### Problem: "Kaggle package not installed"

**Solution:**
```bash
docker-compose down
docker-compose build --no-cache mlflow-app
docker-compose up -d
```

### Problem: "401 Unauthorized"

**Solution:**
1. Double-check credentials in `.env` file
2. Verify API key is still valid at https://www.kaggle.com/settings
3. Regenerate API key if needed

### Problem: Falls back to UCI instead of Kaggle

**Check:**
```bash
# Verify credentials are loaded
docker-compose exec mlflow-app env | grep KAGGLE

# Check kaggle package
docker-compose exec mlflow-app pip list | grep kaggle
```

---

## Quick Command Reference

```bash
# Create .env file (Windows PowerShell)
@"
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
"@ | Out-File -FilePath .env -Encoding utf8

# Create .env file (Linux/Mac)
cat > .env << EOF
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
EOF

# Rebuild and start
docker-compose down
docker-compose build
docker-compose up -d

# Test
docker-compose exec mlflow-app python lab6_wine_quality_autologging.py
```

---

## Success Checklist

- [ ] `.env` file created with correct credentials
- [ ] `docker-compose.yml` has KAGGLE environment variables
- [ ] Containers rebuilt successfully
- [ ] All services running and healthy
- [ ] Kaggle credentials visible in container
- [ ] Lab 6 runs and downloads from Kaggle
- [ ] Dataset file exists in `data/` directory

---

**You're all set!** 🚀

Your Kaggle API is now integrated with Docker using the `.env` file method.

