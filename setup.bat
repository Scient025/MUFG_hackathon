@echo off
echo 🚀 Setting up Superannuation AI Advisor
echo ======================================

echo 📁 Backend files are organized in the 'backend' folder
echo 🔧 Running backend setup...

cd backend
call setup.bat

echo.
echo 🎉 Setup complete!
echo.
echo To start the application:
echo 1. Edit backend/.env file and add your Hugging Face token
echo 2. Start the ML backend: cd backend && python api.py
echo 3. Start the frontend: npm run dev
echo.
echo The application will be available at:
echo - Frontend: http://localhost:8080
echo - Backend API: http://localhost:8000
pause
