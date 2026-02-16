#!/usr/bin/env python3

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Success: {result.stdout.strip()}")
            return True
        else:
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def setup_github_repo():
    """Setup GitHub repository for deployment"""
    
    print("AI Trading Bot - GitHub Setup")
    print("=" * 50)
    
    # Check if git is available
    if not run_command("git --version", "Checking Git installation"):
        print("\nGit is not installed!")
        print("Please install Git first:")
        print("Windows: Download from https://git-scm.com/download/win")
        print("After installation, restart your terminal and run this script again")
        print("Or follow manual instructions in README_DEPLOYMENT.md")
        return False
    
    print("\nStep 1: Initialize Git Repository")
    
    # Initialize git repo
    if not run_command("git init", "Initializing Git repository"):
        return False
    
    # Add all files
    if not run_command("git add .", "Adding all files to Git"):
        return False
    
    # Make initial commit
    if not run_command('git commit -m "Initial commit: AI Trading Bot with Dashboard"', 
                     "Making initial commit"):
        return False
    
    print("\nStep 2: GitHub Repository Setup")
    print("Please follow these steps:")
    print("1. Go to https://github.com")
    print("2. Sign in to your account")
    print("3. Click '+' -> 'New repository'")
    print("4. Repository name: ai-trading-bot")
    print("5. Description: AI-powered trading bot with real-time dashboard")
    print("6. Choose Public or Private")
    print("7. Don't initialize with README")
    print("8. Click 'Create repository'")
    
    # Get GitHub username
    username = input("\nEnter your GitHub username: ").strip()
    if not username:
        print("GitHub username is required")
        return False
    
    print(f"\nStep 3: Connect to GitHub")
    
    # Add remote
    remote_url = f"https://github.com/{username}/ai-trading-bot.git"
    if not run_command(f"git remote add origin {remote_url}", 
                     "Adding remote repository"):
        return False
    
    # Push to GitHub
    if not run_command("git branch -M main", "Setting main branch"):
        return False
    
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        print("\nPush failed! You may need to:")
        print("1. Create the repository on GitHub first")
        print("2. Check your GitHub credentials")
        print("3. Ensure you have push permissions")
        return False
    
    print("\nGitHub Setup Complete!")
    print(f"Repository URL: https://github.com/{username}/ai-trading-bot")
    
    print("\nStep 4: Streamlit Cloud Deployment")
    print("1. Go to https://share.streamlit.io")
    print("2. Click 'New app'")
    print("3. Select your repository")
    print("4. Main file path: src/ai_dashboard.py")
    print("5. Click 'Deploy'")
    
    print("\nStep 5: Configure Secrets")
    print("Add your environment variables to Streamlit Cloud secrets:")
    print("- KITE_API_KEY")
    print("- KITE_API_SECRET") 
    print("- SWIFT_API_KEY")
    print("- SWIFT_API_SECRET")
    print("- And other API keys...")
    
    print("\nFor detailed instructions, see: README_DEPLOYMENT.md")
    
    return True

if __name__ == "__main__":
    try:
        setup_github_repo()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)
