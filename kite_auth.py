#!/usr/bin/env python3

import sys
import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv

def generate_kite_access_token():
    """Generate Kite access token using OAuth flow"""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    
    if not api_key or not api_secret:
        print("Kite API credentials not found in .env file")
        return None
    
    print("Kite API Credentials loaded:")
    print(f"   API Key: {api_key[:10]}...")
    print(f"   API Secret: {api_secret[:10]}...")
    
    # Initialize Kite Connect
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    login_url = kite.login_url()
    
    print("\nStep 1: Complete Authentication")
    print(f"   Visit this URL: {login_url}")
    print("   1. Login with your Kite credentials")
    print("   2. Authorize the application")
    print("   3. You'll be redirected to a callback URL")
    print("   4. Copy the 'request_token' from the URL")
    
    # Get request token from user
    request_token = input("\nEnter the request_token from callback URL: ").strip()
    
    if not request_token:
        print("No request token provided")
        return None
    
    try:
        # Generate access token
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("Access Token Generated:", access_token[:20] + "...")
        
        # Update .env file with access token and request token
        update_env_file(access_token, request_token)
        
        return access_token
        
    except Exception as e:
        print(f"Error generating access token: {e}")
        return None

def update_env_file(access_token, request_token):
    """Update .env file with access token and request token"""
    
    env_file = '.env'
    
    # Read current .env file
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update the lines
    updated_lines = []
    for line in lines:
        if line.startswith('KITE_ACCESS_TOKEN='):
            updated_lines.append(f'KITE_ACCESS_TOKEN={access_token}\n')
        elif line.startswith('KITE_REQUEST_TOKEN='):
            updated_lines.append(f'KITE_REQUEST_TOKEN={request_token}\n')
        else:
            updated_lines.append(line)
    
    # Write back to .env file
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    print(".env file updated with access token and request token")

def test_kite_connection():
    """Test Kite API connection with generated access token"""
    
    load_dotenv()
    
    api_key = os.getenv('KITE_API_KEY')
    access_token = os.getenv('KITE_ACCESS_TOKEN')
    
    if not access_token or access_token == 'your_kite_access_token_here':
        print("Access token not found. Please run authentication first.")
        return False
    
    try:
        kite = KiteConnect(api_key=api_key, access_token=access_token)
        
        # Test API call
        profile = kite.profile()
        print("Kite API Connection Successful!")
        print(f"   User: {profile['user_name']}")
        print(f"   User Type: {profile['user_type']}")
        print(f"   Email: {profile['email']}")
        
        # Get available instruments
        instruments = kite.instruments()
        print(f"   Total Instruments: {len(instruments)}")
        
        return True
        
    except Exception as e:
        print(f"Kite API Connection Failed: {e}")
        return False

if __name__ == "__main__":
    print("Kite API Authentication Tool")
    print("=" * 50)
    
    # Check if we need to authenticate
    load_dotenv()
    access_token = os.getenv('KITE_ACCESS_TOKEN')
    
    if access_token == 'your_kite_access_token_here' or not access_token:
        print("Starting Kite API Authentication...")
        token = generate_kite_access_token()
        
        if token:
            print("\nTesting Kite API Connection...")
            test_kite_connection()
        else:
            print("Authentication failed")
    else:
        print("Testing existing Kite API Connection...")
        test_kite_connection()
