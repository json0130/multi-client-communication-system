import sqlite3
import os
import re

# Define path to the database
DB_FILE = os.path.join(os.path.dirname(__file__), "patient_DB.sqlite")

def init_db():
    """Initializes the patient database with a patients table."""
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                medications TEXT DEFAULT ''
            )
        ''')
        conn.commit()

def add_patient(name):
    """Adds a new patient to the database."""
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO patients (name) VALUES (?)", (name,))
        conn.commit()

def get_patient_by_name(name):
    """Retrieves a patient's info by name."""
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, medications FROM patients WHERE name = ?", (name,))
        return cursor.fetchone()

def update_patient_meds(name, meds_list):
    """Updates the medications field for a patient."""
    meds_string = ', '.join(meds_list)
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE patients SET medications = ? WHERE name = ?", (meds_string, name))
        conn.commit()

def get_patient_history(name):
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT medications FROM patients WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row and row[0]:
            return [m.strip() for m in row[0].split(",") if m.strip()]
        return []

def update_patient_history(name, new_meds, replace=False):
    if replace:
        updated = new_meds
    else:
        old_meds = get_patient_history(name)
        updated = list(set(old_meds + new_meds))  # Merge and deduplicate
    meds_string = ", ".join(updated)
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE patients SET medications = ? WHERE name = ?", (meds_string, name))
        conn.commit()

def add_medications(patient_name, new_meds):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
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
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT medications FROM patients WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return row[0].split(",")
    return []

def extract_name_if_any(message):
    bad_names = {"Yes", "No", "Still", "Currently", "Again", "Medications", "Please", "Hi", "Hello"}

    patterns = [
        r"\b(?:my name is|i am|i'?m|this is)\s+([A-Z][a-z]+)",
    ]

    match = re.search(patterns[0], message)
    if match:
        name_candidate = match.group(1).capitalize()
        if name_candidate not in bad_names:
            return name_candidate
    return None

init_db()