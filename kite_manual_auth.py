#!/usr/bin/env python3

import sys
import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv

def show_auth_url():
    """Show the authentication URL for manual token generation"""
    
    load_dotenv()
    
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    
    if not api_key or not api_secret:
        print("Kite API credentials not found in .env file")
        return
    
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    
    print("Kite API Authentication URL:")
    print("=" * 50)
    print(login_url)
    print("=" * 50)
    
    print("\nInstructions:")
    print("1. Click on the URL above or copy it to your browser")
    print("2. Login with your Kite credentials")
    print("3. Authorize the application")
    print("4. You'll be redirected to a URL like: http://localhost:5000/?request_token=xxxxx&status=success")
    print("5. Copy the 'request_token' value from the URL")
    print("6. Run the token update script with your request token")
    
    print("\nExample callback URL:")
    print("http://localhost:5000/?request_token=abc123def456&status=success")
    print("In this example, the request_token would be: abc123def456")

def update_tokens(request_token):
    """Update access token using request token"""
    
    load_dotenv()
    
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("Access Token Generated:", access_token[:20] + "...")
        
        # Update .env file
        env_file = '.env'
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if line.startswith('KITE_ACCESS_TOKEN='):
                updated_lines.append(f'KITE_ACCESS_TOKEN={access_token}\n')
            elif line.startswith('KITE_REQUEST_TOKEN='):
                updated_lines.append(f'KITE_REQUEST_TOKEN={request_token}\n')
            else:
                updated_lines.append(line)
        
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        
        print("✓ .env file updated successfully")
        
        # Test connection
        kite = KiteConnect(api_key=api_key, access_token=access_token)
        profile = kite.profile()
        print("✓ Kite API Connection Successful!")
        print(f"  User: {profile['user_name']}")
        print(f"  User Type: {profile['user_type']}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Update tokens with provided request token
        request_token = sys.argv[1]
        print(f"Updating tokens with request_token: {request_token}")
        update_tokens(request_token)
    else:
        # Show authentication URL
        show_auth_url()
