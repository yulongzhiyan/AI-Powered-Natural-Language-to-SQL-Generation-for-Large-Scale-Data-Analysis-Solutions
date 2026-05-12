import sqlite3
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)
db_path = "data/school.sqlite"

# Connect to database (creates it if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the Students table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    department TEXT
)
''')

# Insert some dummy data for testing
cursor.executemany('''
INSERT INTO Students (name, age, department) VALUES (?, ?, ?)
''', [
    ('Alice', 19, 'Computer Science'),
    ('Bob', 22, 'Mechanical Engineering'),
    ('Charlie', 18, 'Computer Science')
])

conn.commit()
conn.close()

print(f"Database successfully initialized with tables at: {os.path.abspath(db_path)}")