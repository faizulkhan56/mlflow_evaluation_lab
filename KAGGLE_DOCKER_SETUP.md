# Step-by-Step Guide: Using Kaggle API with Docker (.env Method)

This guide walks you through setting up Kaggle API credentials using a `.env` file with Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Kaggle account with API token
- Your credentials:
  - Username: ``
  - API Key: ``

## Step-by-Step Instructions

### Step 1: Verify .env File Exists

The `.env` file has been created in the project root with your credentials:

```bash
# Check if .env file exists
ls -la .env  # Linux/Mac
dir .env     # Windows
```

**Expected content:**
```
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

### Step 2: Verify docker-compose.yml is Updated

The `docker-compose.yml` file has been updated to read from the `.env` file. Verify these lines exist in the `mlflow-app` service:

```yaml
environment:
  - KAGGLE_USERNAME=${KAGGLE_USERNAME}
  - KAGGLE_KEY=${KAGGLE_KEY}
```

### Step 3: Rebuild Docker Containers (if already running)

If you have containers already running, you need to rebuild to include the new environment variables:

```bash
# Stop existing containers
docker-compose down

# Rebuild containers (this installs kaggle package)
docker-compose build

# Start services
docker-compose up -d
```

**Note:** The `kaggle` package will be installed during the build process.

### Step 4: Verify Services are Running

Check that all services are healthy:

```bash
docker-compose ps
```

You should see:
- `postgres` (healthy)
- `mlflow-server` (healthy)
- `mlflow-app` (running)
- `mlflow-api` (healthy)

### Step 5: Test Kaggle Integration

Test that the Kaggle credentials are working:

```bash
# Check environment variables in container
docker-compose exec mlflow-app env | grep KAGGLE
```

**Expected output:**
```
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

### Step 6: Run Lab 6 (Autologging)

Run the autologging lab script:

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
```

### Step 7: Run Lab 7 (Manual Logging)

Run the manual logging lab script:

```bash
docker-compose exec mlflow-app python lab7_wine_quality_manual.py
```

**Expected output:**
```
Loading dataset from data/winequality-red.csv...
...
Random Forest (Manual) Accuracy: 0.XXX
...
```

### Step 8: Verify Dataset Downloaded

Check that the dataset file exists:

```bash
# Check from host
ls -lh data/winequality-red.csv  # Linux/Mac
dir data\winequality-red.csv      # Windows

# Or check from container
docker-compose exec mlflow-app ls -lh data/winequality-red.csv
```

## Troubleshooting

### Issue: "Kaggle package not installed"

**Solution:** Rebuild the container:
```bash
docker-compose down
docker-compose build --no-cache mlflow-app
docker-compose up -d
```

### Issue: "401 Unauthorized" or "403 Forbidden"

**Solution:** 
1. Verify your credentials in `.env` file
2. Check that API key is valid at https://www.kaggle.com/settings
3. Regenerate API key if needed

### Issue: Environment variables not set in container

**Solution:**
1. Verify `.env` file exists in project root
2. Check `docker-compose.yml` has the environment variable lines
3. Restart containers: `docker-compose restart mlflow-app`

### Issue: Dataset downloads but script fails

**Solution:**
1. Check dataset file exists: `ls data/winequality-red.csv`
2. Check file permissions
3. Try deleting and re-downloading: `rm data/winequality-red.csv`

### Issue: Falls back to UCI instead of Kaggle

**Possible causes:**
1. Kaggle credentials not set correctly
2. Kaggle package not installed
3. Network issues with Kaggle API

**Check:**
```bash
# Verify credentials
docker-compose exec mlflow-app env | grep KAGGLE

# Check if kaggle package is installed
docker-compose exec mlflow-app pip list | grep kaggle
```

## Verification Checklist

- [ ] `.env` file exists with correct credentials
- [ ] `docker-compose.yml` includes KAGGLE environment variables
- [ ] Containers rebuilt after adding credentials
- [ ] All services are running and healthy
- [ ] Kaggle credentials visible in container environment
- [ ] Lab 6 script runs successfully
- [ ] Lab 7 script runs successfully
- [ ] Dataset file exists in `data/` directory

## Next Steps

After successful setup:

1. **View MLflow UI**: http://localhost:5000
   - Check experiments: `lab6-wine-quality-autologging` and `lab7-wine-quality-manual`

2. **Run other labs**:
   ```bash
   docker-compose exec mlflow-app python lab1_logistic_regression.py
   docker-compose exec mlflow-app python lab2_decision_tree_regressor.py
   # ... etc
   ```

3. **Test API endpoints**: http://localhost:8000/docs

## Security Reminders

⚠️ **Important:**
- `.env` file is in `.gitignore` (won't be committed to Git)
- Never share your API key publicly
- Regenerate API key if exposed
- Use different keys for different environments (dev/prod)

## Quick Reference Commands

```bash
# Start services
docker-compose up -d

# Rebuild after changes
docker-compose build
docker-compose up -d

# Run lab scripts
docker-compose exec mlflow-app python lab6_wine_quality_autologging.py
docker-compose exec mlflow-app python lab7_wine_quality_manual.py

# View logs
docker-compose logs -f mlflow-app

# Check environment variables
docker-compose exec mlflow-app env | grep KAGGLE

# Stop services
docker-compose down
```

---

**Your Setup is Complete!** 🎉

The `.env` file is configured with your credentials and `docker-compose.yml` is ready to use them.

