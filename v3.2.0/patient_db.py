import sqlite3
import os
import re

# Define path to the database
DB_FILE = os.path.join(os.path.dirname(__file__), "patient_DB.sqlite")

def init_db():
    """Initializes the patient database with a patients table."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                meds TEXT DEFAULT ''
            )
        ''')
        conn.commit()

def add_patient(name):
    """Adds a new patient to the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO patients (name) VALUES (?)", (name,))
        conn.commit()

def get_patient_by_name(name):
    """Retrieves a patient's info by name."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, meds FROM patients WHERE name = ?", (name,))
        return cursor.fetchone()

def update_patient_meds(name, meds_list):
    """Updates the meds field for a patient."""
    meds_string = ', '.join(meds_list)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE patients SET meds = ? WHERE name = ?", (meds_string, name))
        conn.commit()

def get_patient_history(name):
    """Alias for get_patient_medications to match server expectations."""
    return get_patient_medications(name)

def update_patient_history(name, meds):
    """Alias for add_medications to match server expectations."""
    return add_medications(name, meds)

def add_medications(patient_name, new_meds):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Fetch existing medications
    cursor.execute("SELECT medications FROM patients WHERE name = ?", (patient_name,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False  # patient not found

    existing_meds = row[0].split(",") if row[0] else []
    updated_meds = list(set(existing_meds + new_meds))

    cursor.execute("UPDATE patients SET medications = ? WHERE name = ?",
                   (",".join(updated_meds), patient_name))
    conn.commit()
    conn.close()
    return True

def get_patient_medications(name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT medications FROM patients WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return row[0].split(",")
    return []

def extract_name_if_any(message):
    lowered = message.lower()

    # Match: "my name is ___"
    match = re.search(r"\bmy name is (\w+)", lowered)
    if match:
        name = match.group(1).capitalize()
        print(f"🔎 Detected name from 'my name is': {name}")
        return name

    # Match: "i'm ___" or "i am ___"
    match = re.search(r"\b(i'm|i am) (\w+)", lowered)
    if match:
        name = match.group(2).capitalize()
        print(f"🔎 Detected name from 'I'm/I am': {name}")
        return name

    print("❌ No name detected")
    return None

init_db()