#!/usr/bin/env python3

from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def authenticate_kite():
    """Complete Kite API authentication process"""
    
    # Your API credentials
    api_key = "nyj6rh8b0exlwh23"
    api_secret = "qx662nkun2xes6tpghv4segsamu7swg9"
    
    print("Kite API Authentication")
    print("=" * 50)
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    
    # Step 1: Get login URL
    print("\nStep 1: Login to Kite")
    print("Visit this URL to authorize:")
    print(kite.login_url())
    print("\nAfter login, copy the 'request_token' from the callback URL")
    
    # Step 2: Get request token from user
    request_token = input("\nEnter your request_token: ").strip()
    
    if not request_token:
        print("Request token is required!")
        return None
    
    # Step 3: Generate session
    print("\nStep 3: Generating access token...")
    
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        kite.set_access_token(access_token)
        
        print("Authentication successful!")
        print(f"Access Token: {access_token}")
        
        # Test connection
        print("\nTesting connection...")
        profile = kite.profile()
        print(f"User: {profile.get('user_name', 'N/A')}")
        print(f"User ID: {profile.get('user_id', 'N/A')}")
        print(f"Email: {profile.get('email', 'N/A')}")
        
        # Update .env file
        update_env_file(access_token)
        
        return kite
        
    except Exception as e:
        print(f"Authentication failed: {e}")
        return None

def update_env_file(access_token):
    """Update .env file with access token"""
    try:
        # Read current .env file
        env_lines = []
        with open('.env', 'r') as f:
            env_lines = f.readlines()
        
        # Update or add access token
        updated = False
        for i, line in enumerate(env_lines):
            if line.startswith('KITE_ACCESS_TOKEN='):
                env_lines[i] = f'KITE_ACCESS_TOKEN={access_token}\n'
                updated = True
                break
        
        if not updated:
            env_lines.append(f'KITE_ACCESS_TOKEN={access_token}\n')
        
        # Write back to .env
        with open('.env', 'w') as f:
            f.writelines(env_lines)
        
        print(".env file updated with access token")
        
    except Exception as e:
        print(f"Could not update .env file: {e}")
        print(f"Manually add this to your .env file:")
        print(f"KITE_ACCESS_TOKEN={access_token}")

def test_kite_connection(kite):
    """Test Kite API connection"""
    try:
        print("\nTesting API functions...")
        
        # Test profile
        profile = kite.profile()
        print(f"Profile: {profile.get('user_name', 'N/A')}")
        
        # Test margins
        margins = kite.margins()
        if 'equity' in margins:
            equity = margins['equity']
            print(f"Available Balance: {equity.get('net', 0):,.2f}")
        
        # Test holdings
        holdings = kite.holdings()
        print(f"Holdings: {len(holdings)} positions")
        
        print("\nKite API is fully functional!")
        return True
        
    except Exception as e:
        print(f"API test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Authenticate
        kite = authenticate_kite()
        
        if kite:
            # Test connection
            test_kite_connection(kite)
            
            print("\nYour Kite API is ready for use in the dashboard!")
            print("Your dashboard will now show real Kite data.")
        
    except KeyboardInterrupt:
        print("\nAuthentication cancelled")
    except Exception as e:
        print(f"Error: {e}")
