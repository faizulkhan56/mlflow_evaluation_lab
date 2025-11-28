# Troubleshooting Network Timeout Issues

## Problem
`ReadTimeoutError` or `HTTPSConnectionPool` timeout when building Docker images.

## Solutions

### Solution 1: Use Updated Dockerfiles (Already Applied)

The Dockerfiles have been updated with:
- Increased timeout: `--default-timeout=100`
- Retry logic: `--retries 5`
- Batch installation to avoid large package timeouts

**Try building again:**
```bash
docker-compose build --no-cache
```

### Solution 2: Use Alternative PyPI Mirror

If PyPI is slow, use a faster mirror. Create a `pip.conf` file:

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 100
retries = 5
```

Or use other mirrors:
- **Tsinghua (China)**: `https://pypi.tuna.tsinghua.edu.cn/simple`
- **Aliyun (China)**: `https://mirrors.aliyun.com/pypi/simple/`
- **Douban (China)**: `https://pypi.douban.com/simple/`

### Solution 3: Build with Network Mode

If behind a proxy or firewall:

```bash
# Build with host network (bypasses Docker network)
docker-compose build --network=host
```

### Solution 4: Install Packages Manually in Container

If build keeps failing, install packages at runtime:

1. **Start container without installing packages:**
   - Comment out `RUN pip install` lines in Dockerfile temporarily
   - Build and start container

2. **Install packages inside running container:**
```bash
docker-compose exec mlflow-app pip install --default-timeout=100 --retries 5 -r requirements.txt
```

3. **Commit the container as new image:**
```bash
docker commit mlflow-app mlflow-app:with-packages
```

### Solution 5: Use Pre-built Base Image

Create a base image with all packages pre-installed:

```dockerfile
# Dockerfile.base
FROM python:3.11-slim
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries 5 \
    mlflow scikit-learn xgboost shap pandas numpy psycopg2-binary \
    fastapi uvicorn pydantic requests
```

Build base image once:
```bash
docker build -f Dockerfile.base -t mlflow-base:latest .
```

Then use in other Dockerfiles:
```dockerfile
FROM mlflow-base:latest
```

### Solution 6: Increase Docker Build Timeout

If using Docker Desktop, increase timeout in settings.

### Solution 7: Build During Off-Peak Hours

PyPI can be slower during peak hours. Try building at different times.

## Quick Fix Commands

### Retry Build with More Verbose Output
```bash
docker-compose build --progress=plain --no-cache
```

### Build One Service at a Time
```bash
# Build mlflow-server first (smallest)
docker-compose build mlflow-server

# Then build others
docker-compose build mlflow-app
docker-compose build mlflow-api
```

### Check Network Connectivity
```bash
# Test PyPI connectivity from container
docker run --rm python:3.11-slim pip install --dry-run mlflow
```

## Recommended Approach

1. **First, try the updated Dockerfiles** (already done)
2. **If still failing, use Solution 2** (PyPI mirror)
3. **If behind firewall, use Solution 3** (host network)
4. **As last resort, use Solution 4** (manual installation)

## Current Status

✅ Dockerfiles updated with:
- `--default-timeout=100` (100 seconds timeout)
- `--retries 5` (retry 5 times)
- Batch installation (smaller packages first)
- pip upgrade before installation

**Next step**: Try building again with:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

