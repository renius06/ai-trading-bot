#!/usr/bin/env python3

import requests
import json
import time

def check_streamlit_deployment():
    """Check Streamlit Cloud deployment status"""
    
    app_url = "https://ai-trading-bot-reni.streamlit.app"
    
    print("🚀 Auto-Deployment Monitor")
    print("=" * 50)
    print(f"App URL: {app_url}")
    print("\nChecking deployment status...")
    
    try:
        # Check if app is accessible
        response = requests.get(app_url, timeout=10)
        if response.status_code == 200:
            print("✅ App is accessible!")
            print("📊 Dashboard is live!")
            
            # Check for specific content
            if "AI Trading Bot" in response.text:
                print("✅ Main content loaded successfully!")
            else:
                print("⚠️ App accessible but content may be loading...")
                
        else:
            print(f"❌ App returned status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not access app: {e}")
    
    print("\n" + "=" * 50)

def show_deployment_status():
    """Show current deployment status"""
    
    print("📋 Deployment Status Checklist")
    print("=" * 50)
    
    status_items = [
        ("✅ Code pushed to GitHub", "Completed"),
        ("✅ All imports fixed", "Completed"),
        ("✅ File structure optimized", "Completed"),
        ("✅ Authentication scripts ready", "Completed"),
        ("✅ Local dashboard running", "Completed"),
        ("🔄 Streamlit Cloud deployment", "In Progress"),
        ("🔑 Kite API authentication", "Pending user action"),
        ("📱 App accessibility", "Testing..."),
    ]
    
    for item, status in status_items:
        print(f"{item}: {status}")
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Complete Kite API authentication")
    print("2. Verify Streamlit Cloud deployment")
    print("3. Test all dashboard features")
    print("4. Share your live app!")

def main():
    """Main function"""
    print("🤖 AI Trading Bot - Auto-Deployment Tool")
    print("=" * 60)
    
    while True:
        print("\n" + "=" * 60)
        print("Choose an option:")
        print("1. 🚀 Check deployment status")
        print("2. 📋 Show deployment checklist")
        print("3. 🔑 Kite authentication guide")
        print("4. 👋 Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            check_streamlit_deployment()
            
        elif choice == '2':
            show_deployment_status()
            
        elif choice == '3':
            print("\n🔑 Kite Authentication Guide:")
            print("=" * 40)
            print("1. Run: python kite_auth_manual.py")
            print("2. Visit: https://kite.zerodha.com/connect/login?api_key=nyj6rh8b0exlwh23&v=3")
            print("3. Copy request token from callback URL")
            print("4. Update script with your token")
            print("5. Run authentication")
            
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
        print("\n\n👋 Auto-deployment tool closed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
