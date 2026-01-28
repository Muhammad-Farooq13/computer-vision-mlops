# Git Initialization Script for Windows PowerShell
# Run this script to initialize your repository and prepare for GitHub upload

Write-Host "🚀 Initializing Git Repository for Computer Vision Project" -ForegroundColor Green
Write-Host "Author: Muhammad Farooq" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/downloads" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Git is installed" -ForegroundColor Green

# Initialize git repository if not already initialized
if (-not (Test-Path .git)) {
    Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git repository already exists" -ForegroundColor Green
}

# Configure git user
Write-Host ""
Write-Host "⚙️  Configuring Git user..." -ForegroundColor Yellow
git config user.name "Muhammad Farooq"
git config user.email "mfarooqshafee333@gmail.com"
Write-Host "✅ Git user configured" -ForegroundColor Green

# Add all files
Write-Host ""
Write-Host "📝 Adding files to Git..." -ForegroundColor Yellow
git add .
Write-Host "✅ Files added" -ForegroundColor Green

# Create initial commit
Write-Host ""
Write-Host "💾 Creating initial commit..." -ForegroundColor Yellow
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

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Initial commit created" -ForegroundColor Green
} else {
    Write-Host "⚠️  Commit might have failed or nothing to commit" -ForegroundColor Yellow
}

# Instructions for GitHub
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "📋 Next Steps to Upload to GitHub:" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor Yellow
Write-Host "   - Go to https://github.com/new" -ForegroundColor White
Write-Host "   - Repository name: OPencv" -ForegroundColor White
Write-Host "   - Description: Production-ready Computer Vision Project with MLOps" -ForegroundColor White
Write-Host "   - Choose: Public or Private" -ForegroundColor White
Write-Host "   - DON'T initialize with README (we already have one)" -ForegroundColor Red
Write-Host ""
Write-Host "2. After creating the repository, run these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "   git remote add origin https://github.com/Muhammad-Farooq-13/OPencv.git" -ForegroundColor White
Write-Host "   git branch -M main" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "3. Optional: Add repository description and topics on GitHub" -ForegroundColor Yellow
Write-Host "   Topics: computer-vision, deep-learning, pytorch, flask, mlops, docker" -ForegroundColor White
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✅ Git repository is ready for upload!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Repository Statistics:" -ForegroundColor Cyan
Write-Host "Files tracked: " -NoNewline
git ls-files | Measure-Object -Line | Select-Object -ExpandProperty Lines
Write-Host ""

Write-Host "🎉 Setup complete! Happy coding!" -ForegroundColor Green
