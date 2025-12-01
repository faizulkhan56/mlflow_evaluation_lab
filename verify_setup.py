"""
Quick verification script to check if .env file is configured correctly
Run this before building Docker containers
"""
import os
import sys

def verify_env_file():
    """Verify .env file exists and has correct format"""
    print("=" * 60)
    print("Kaggle Setup Verification")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print("❌ ERROR: .env file not found!")
        print(f"   Expected location: {os.path.abspath(env_file)}")
        print()
        print("   Create .env file with:")
        print("   KAGGLE_USERNAME=your-username")
        print("   KAGGLE_KEY=your-api-key")
        return False
    
    print(f"✅ .env file found: {os.path.abspath(env_file)}")
    print()
    
    # Read and verify content
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            
        # Check for required variables
        has_username = False
        has_key = False
        username_value = None
        key_value = None
        
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'KAGGLE_USERNAME':
                    has_username = True
                    username_value = value
                elif key == 'KAGGLE_KEY':
                    has_key = True
                    key_value = value
        
        # Verify both are present
        if not has_username:
            print("❌ ERROR: KAGGLE_USERNAME not found in .env file")
            return False
        
        if not has_key:
            print("❌ ERROR: KAGGLE_KEY not found in .env file")
            return False
        
        # Verify values are not empty
        if not username_value:
            print("❌ ERROR: KAGGLE_USERNAME is empty")
            return False
        
        if not key_value:
            print("❌ ERROR: KAGGLE_KEY is empty")
            return False
        
        # Verify expected values
        # Note: We don't check for specific username value to allow flexibility
        print(f"✅ KAGGLE_USERNAME: {username_value}")
        
        if len(key_value) < 10:
            print(f"⚠️  WARNING: KAGGLE_KEY seems too short (length: {len(key_value)})")
        else:
            print(f"✅ KAGGLE_KEY: {key_value[:10]}... (hidden)")
        
        print()
        print("=" * 60)
        print("✅ All checks passed! .env file is configured correctly.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ ERROR reading .env file: {str(e)}")
        return False

def verify_docker_compose():
    """Verify docker-compose.yml has Kaggle environment variables"""
    print()
    print("Checking docker-compose.yml...")
    
    compose_file = "docker-compose.yml"
    if not os.path.exists(compose_file):
        print(f"❌ ERROR: {compose_file} not found!")
        return False
    
    try:
        with open(compose_file, 'r') as f:
            content = f.read()
        
        if 'KAGGLE_USERNAME=${KAGGLE_USERNAME}' in content:
            print("✅ docker-compose.yml has KAGGLE_USERNAME configured")
        else:
            print("❌ ERROR: docker-compose.yml missing KAGGLE_USERNAME")
            return False
        
        if 'KAGGLE_KEY=${KAGGLE_KEY}' in content:
            print("✅ docker-compose.yml has KAGGLE_KEY configured")
        else:
            print("❌ ERROR: docker-compose.yml missing KAGGLE_KEY")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR reading docker-compose.yml: {str(e)}")
        return False

def verify_requirements():
    """Verify requirements.txt has kaggle package"""
    print()
    print("Checking requirements.txt...")
    
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"❌ ERROR: {req_file} not found!")
        return False
    
    try:
        with open(req_file, 'r') as f:
            content = f.read()
        
        if 'kaggle' in content.lower():
            print("✅ requirements.txt includes kaggle package")
            return True
        else:
            print("❌ ERROR: requirements.txt missing kaggle package")
            return False
        
    except Exception as e:
        print(f"❌ ERROR reading requirements.txt: {str(e)}")
        return False

if __name__ == "__main__":
    print()
    all_ok = True
    
    # Run all checks
    all_ok &= verify_env_file()
    all_ok &= verify_docker_compose()
    all_ok &= verify_requirements()
    
    print()
    if all_ok:
        print("🎉 Everything looks good! You can proceed with:")
        print()
        print("   docker-compose build")
        print("   docker-compose up -d")
        print()
        sys.exit(0)
    else:
        print("❌ Some issues found. Please fix them before proceeding.")
        print()
        sys.exit(1)

