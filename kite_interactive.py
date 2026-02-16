#!/usr/bin/env python3

from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_login_url():
    """Get Kite login URL"""
    api_key = "nyj6rh8b0exlwh23"
    kite = KiteConnect(api_key=api_key)
    return kite.login_url()

def manual_auth():
    """Manual Kite authentication process"""
    
    print("🔑 Kite API Manual Authentication")
    print("=" * 60)
    
    # Show login URL
    login_url = get_login_url()
    print(f"\n📱 Step 1: Visit this URL:")
    print(f"{login_url}")
    print("\nCopy this URL to your browser and login with your Kite account.")
    
    # Instructions for getting request token
    print("\n" + "=" * 60)
    print("📋 Step 2: Get Request Token")
    print("After login, you'll be redirected to a URL like:")
    print("https://kite.zerodha.com/connect/login?api_key=...&request_token=YOUR_TOKEN_HERE")
    print("\nCopy the 'request_token' value from that URL.")
    
    # Get request token from user
    print("\n" + "=" * 60)
    print("🔧 Step 3: Enter Request Token")
    
    while True:
        request_token = input("\nEnter your request_token (or 'skip' to exit): ").strip()
        
        if request_token.lower() == 'skip':
            print("👋 Authentication skipped.")
            return None
            
        if not request_token:
            print("❌ Request token cannot be empty. Please try again.")
            continue
        
        # Confirm token
        print(f"\n✅ You entered: {request_token}")
        confirm = input("Is this correct? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            return request_token
        elif confirm in ['n', 'no']:
            print("❌ Please try again.")
            continue
        else:
            print("❌ Please enter 'y' or 'n'.")

def complete_authentication(request_token):
    """Complete the authentication process"""
    
    api_key = "nyj6rh8b0exlwh23"
    api_secret = "qx662nkun2xes6tpghv4segsamu7swg9"
    
    print("\n🔄 Step 4: Generating Access Token...")
    
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print("✅ Authentication successful!")
        print(f"🔑 Access Token: {access_token}")
        
        # Update .env file
        update_env_file(access_token)
        
        # Test connection
        print("\n🧪 Testing API connection...")
        test_connection(kite)
        
        return access_token
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
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
        
        print("✅ .env file updated successfully!")
        
    except Exception as e:
        print(f"⚠️ Could not update .env file: {e}")
        print("Please manually update your .env file with:")
        print(f"KITE_ACCESS_TOKEN={access_token}")

def test_connection(kite):
    """Test Kite API connection"""
    try:
        print("\n📊 Testing API functions...")
        
        # Test profile
        profile = kite.profile()
        print(f"✅ User: {profile.get('user_name', 'N/A')}")
        print(f"✅ Email: {profile.get('email', 'N/A')}")
        print(f"✅ User ID: {profile.get('user_id', 'N/A')}")
        
        # Test margins
        margins = kite.margins()
        if 'equity' in margins:
            equity = margins['equity']
            print(f"✅ Available Balance: ₹{equity.get('net', 0):,.2f}")
        
        # Test holdings
        holdings = kite.holdings()
        print(f"✅ Holdings: {len(holdings)} positions")
        
        print("\n🎉 Kite API is fully functional!")
        print("📱 Your dashboard will now show real Kite data.")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def show_menu():
    """Show main menu"""
    print("\n" + "=" * 60)
    print("🤖 Kite API Authentication Menu")
    print("=" * 60)
    print("1. 🔑 Start Authentication Process")
    print("2. 📋 Show Login URL")
    print("3. 🧪 Test Current Connection")
    print("4. 👋 Exit")
    print("=" * 60)

def main():
    """Main function"""
    while True:
        show_menu()
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            request_token = manual_auth()
            if request_token:
                complete_authentication(request_token)
                
        elif choice == '2':
            login_url = get_login_url()
            print(f"\n📱 Login URL:")
            print(f"{login_url}")
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            # Test existing connection
            api_key = "nyj6rh8b0exlwh23"
            api_secret = "qx662nkun2xes6tpghv4segsamu7swg9"
            
            # Check if access token exists
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    content = f.read()
                    if 'KITE_ACCESS_TOKEN=' in content and 'your_kite_access_token_here' not in content:
                        access_token = content.split('KITE_ACCESS_TOKEN=')[1].strip()
                        kite = KiteConnect(api_key=api_key, access_token=access_token)
                        test_connection(kite)
                    else:
                        print("❌ No valid access token found. Please run option 1 first.")
            else:
                print("❌ No .env file found. Please run option 1 first.")
                
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Authentication cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
