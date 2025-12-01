#!/bin/bash
# Setup script for Kaggle API credentials
# Usage: ./setup_kaggle.sh

echo "Kaggle API Credentials Setup"
echo "============================"
echo ""
echo "This script will help you set up Kaggle API credentials."
echo "You can get your API token from: https://www.kaggle.com/settings"
echo ""

# Check if credentials are already set
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "✓ Kaggle credentials are already set in environment variables"
    echo "  Username: $KAGGLE_USERNAME"
    echo "  Key: ${KAGGLE_KEY:0:10}..."
    echo ""
    read -p "Do you want to update them? (y/n): " update
    if [ "$update" != "y" ]; then
        echo "Keeping existing credentials."
        exit 0
    fi
fi

# Prompt for credentials
read -p "Enter your Kaggle username: " username
read -sp "Enter your Kaggle API key: " key
echo ""

# Validate inputs
if [ -z "$username" ] || [ -z "$key" ]; then
    echo "Error: Username and key cannot be empty"
    exit 1
fi

# Export credentials
export KAGGLE_USERNAME="$username"
export KAGGLE_KEY="$key"

echo ""
echo "✓ Credentials set successfully!"
echo ""
echo "To make these permanent, add them to your shell profile:"
echo "  export KAGGLE_USERNAME=\"$username\""
echo "  export KAGGLE_KEY=\"$key\""
echo ""
echo "Or create a .env file with:"
echo "  KAGGLE_USERNAME=$username"
echo "  KAGGLE_KEY=$key"
echo ""
echo "You can now run the lab scripts:"
echo "  python lab6_wine_quality_autologging.py"
echo "  python lab7_wine_quality_manual.py"
echo ""

