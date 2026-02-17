#!/usr/bin/env python3

from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def simple_authenticate():
    """Simple authentication with new API key"""
    
    # Your new API key
    api_key = "qlqdpxy7z0p5ydsf"
    api_secret = "qx662nkun2xes6tpghv4segsamu7swg9"
    
    print("Kite API Quick Authentication")
    print("=" * 50)
    print(f"API Key: {api_key}")
    print(f"API Secret: {api_secret}")
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    
    # Show login URL
    print("\nStep 1: Login to Kite")
    print("Visit this URL to authorize:")
    print(kite.login_url())
    print("\nAfter login, copy the 'request_token' from the callback URL")
    
    # Get request token from user
    request_token = input("\nEnter your request_token: ").strip()
    
    if not request_token:
        print("Request token is required!")
        return None
    
    # Generate session
    print("\nStep 2: Generating access token...")
    
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("Authentication successful!")
        print(f"Access Token: {access_token}")
        
        # Update .env file
        update_env_file(access_token)
        
        # Test connection
        print("\nStep 3: Testing connection...")
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
            print(f"Available Balance: {equity.get('net', 0):,.2f}")
        
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
        simple_authenticate()
    except KeyboardInterrupt:
        print("\nAuthentication cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
