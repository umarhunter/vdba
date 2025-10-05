# run.py
import os
from dotenv import load_dotenv
from app import create_app

# Load environment variables from secrets.env
load_dotenv('secrets.env')

app = create_app()

if __name__ == "__main__":
    #app.run(debug=True)
    app.run()
