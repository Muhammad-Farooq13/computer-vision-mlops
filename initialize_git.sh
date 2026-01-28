#!/bin/bash
# Git Initialization Script for Linux/Mac
# Run this script to initialize your repository and prepare for GitHub upload

echo "🚀 Initializing Git Repository for Computer Vision Project"
echo "Author: Muhammad Farooq"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    echo "Ubuntu/Debian: sudo apt-get install git"
    echo "Mac: brew install git"
    exit 1
fi

echo "✅ Git is installed"

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Configure git user
echo ""
echo "⚙️  Configuring Git user..."
git config user.name "Muhammad Farooq"
git config user.email "mfarooqshafee333@gmail.com"
echo "✅ Git user configured"

# Add all files
echo ""
echo "📝 Adding files to Git..."
git add .
echo "✅ Files added"

# Create initial commit
echo ""
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Complete Computer Vision project setup

- Added comprehensive project structure
- Implemented data processing pipeline
- Added multiple model architectures (ResNet, EfficientNet, VGG)
- Created Flask REST API for deployment
- Added Docker support with docker-compose
- Implemented MLOps pipeline with MLflow tracking
- Added complete test suite with pytest
- Created Jupyter notebooks for exploration
- Added comprehensive documentation
- Configured CI/CD with GitHub Actions

Project ready for production deployment!"

if [ $? -eq 0 ]; then
    echo "✅ Initial commit created"
else
    echo "⚠️  Commit might have failed or nothing to commit"
fi

# Instructions for GitHub
echo ""
echo "================================================"
echo "📋 Next Steps to Upload to GitHub:"
echo "================================================"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: OPencv"
echo "   - Description: Production-ready Computer Vision Project with MLOps"
echo "   - Choose: Public or Private"
echo "   - DON'T initialize with README (we already have one)"
echo ""
echo "2. After creating the repository, run these commands:"
echo ""
echo "   git remote add origin https://github.com/Muhammad-Farooq-13/OPencv.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Optional: Add repository description and topics on GitHub"
echo "   Topics: computer-vision, deep-learning, pytorch, flask, mlops, docker"
echo ""
echo "================================================"
echo "✅ Git repository is ready for upload!"
echo "================================================"
echo ""
echo "📊 Repository Statistics:"
echo "Files tracked: $(git ls-files | wc -l)"
echo ""

echo "🎉 Setup complete! Happy coding!"
