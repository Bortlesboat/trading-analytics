# Configuration template
# Copy this file to config.py and update with your actual paths

DATA_DIR = "./data"  # Update to your data directory

# Trade history files - update with your brokerage export files
TRADE_FILES = [
    f"{DATA_DIR}/trade_history.csv",
]

# Current positions file (optional)
POSITIONS_FILE = f"{DATA_DIR}/positions.csv"

# For RAG/document indexing (optional)
DOCS_FOLDER = "./docs"
DB_FOLDER = "./faiss_db"
