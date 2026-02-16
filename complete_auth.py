#!/usr/bin/env python3

from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def complete_authentication():
    """Complete Kite authentication with provided credentials"""
    
    # Your credentials
    api_key = "nyj6rh8b0exlwh23"
    api_secret = "qx662nkun2xes6tpghv4segsamu7swg9"
    request_token = "YOUR_NEW_REQUEST_TOKEN_HERE"
    
    print("Kite API Authentication")
    print("=" * 50)
    print(f"API Key: {api_key}")
    print(f"Request Token: {request_token}")
    
    print("\nGenerating access token...")
    
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("Authentication successful!")
        print(f"Access Token: {access_token}")
        
        # Update .env file
        update_env_file(access_token)
        
        # Test connection
        print("\nTesting API connection...")
        test_connection(kite)
        
        return access_token
        
    except Exception as e:
        print(f"Authentication failed: {e}")
        return None

def update_env_file(access_token):
    """Update .env file with access token"""
    try:
        # Read current .env file
        env_lines = []
        if os.path.exists('.env'):
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
        
        print(".env file updated successfully!")
        
    except Exception as e:
        print(f"Could not update .env file: {e}")
        print("Please manually update your .env file with:")
        print(f"KITE_ACCESS_TOKEN={access_token}")

def test_connection(kite):
    """Test Kite API connection"""
    try:
        print("Testing API functions...")
        
        # Test profile
        profile = kite.profile()
        print(f"User: {profile.get('user_name', 'N/A')}")
        print(f"Email: {profile.get('email', 'N/A')}")
        print(f"User ID: {profile.get('user_id', 'N/A')}")
        
        # Test margins
        margins = kite.margins()
        if 'equity' in margins:
            equity = margins['equity']
            print(f"Available Balance: {equity.get('net', 0):.2f}")
        
        # Test holdings
        holdings = kite.holdings()
        print(f"Holdings: {len(holdings)} positions")
        
        print("\nKite API is fully functional!")
        print("Your dashboard will now show real Kite data.")
        return True
        
    except Exception as e:
        print(f"API test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        complete_authentication()
    except KeyboardInterrupt:
        print("\nAuthentication cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
