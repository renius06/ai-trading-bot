#!/usr/bin/env python3

from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def show_login_url():
    """Show Kite login URL"""
    api_key = "nyj6rh8b0exlwh23"
    kite = KiteConnect(api_key=api_key)
    
    print("Kite API Login URL:")
    print(kite.login_url())
    print("\nCopy this URL to your browser and login with your Kite account.")
    print("After login, you'll be redirected to a URL with 'request_token'")
    print("Copy that request_token value.")

def authenticate_with_token(request_token):
    """Authenticate with provided request token"""
    
    api_key = "nyj6rh8b0exlwh23"
    api_secret = "qx662nkun2xes6tpghv4segsamu7swg9"
    
    print("Generating access token...")
    
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("Authentication successful!")
        print(f"Access Token: {access_token}")
        
        # Update .env file
        update_env_file(access_token)
        
        # Test connection
        print("Testing API connection...")
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
        
        print("Kite API is fully functional!")
        print("Your dashboard will now show real Kite data.")
        return True
        
    except Exception as e:
        print(f"API test failed: {e}")
        return False

def main():
    """Main function"""
    print("Kite API Manual Authentication")
    print("=" * 50)
    
    # Show login URL
    show_login_url()
    
    # Get request token
    print("\nEnter your request_token from Kite login:")
    request_token = input("Request Token: ").strip()
    
    if not request_token:
        print("Request token cannot be empty.")
        return
    
    # Authenticate
    print("\nAuthenticating...")
    access_token = authenticate_with_token(request_token)
    
    if access_token:
        print("\nAuthentication completed successfully!")
        print("You can now restart your dashboard to see Kite data.")
    else:
        print("\nAuthentication failed. Please check your request token.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAuthentication cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
