# code/memory_insertion/insert_memory.py
# Insert user memory (text + embedding) into PostgreSQL with pgvector

import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv

# Import the shared embedding generator
from generator.llm_generator import generate_embedding

# Load environment variables
load_dotenv()

# Database connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "cognitive_memory")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
cursor = conn.cursor()

# Insert memory into database
def insert_memory(user_id, text_prompt):
    embedding = generate_embedding(text_prompt)
    if embedding is None:
        print("Failed to generate embedding. Memory not inserted.")
        return

    cursor.execute(
        """
        INSERT INTO memories (user_id, text_prompt, embedding)
        VALUES (%s, %s, %s)
        """,
        (user_id, text_prompt, embedding)
    )
    conn.commit()
    print("Memory inserted successfully.")

if __name__ == "__main__":
    user_id = 1
    text_prompt = "A peaceful cabin in the woods during winter"
    insert_memory(user_id, text_prompt)

conn.close()
