@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting the application...
python app.py

pause
