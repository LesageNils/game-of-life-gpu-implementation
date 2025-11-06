@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
pip install --upgrade pip

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Setup complete! To activate again use:
echo venv\Scripts\activate
