import pyodbc
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection parameters from environment variables
server = os.getenv('SERVER')
database = os.getenv('DATABASE')
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
driver = '{ODBC Driver 18 for SQL Server}'

# Connect to Azure SQL Server
def connect_to_db():
    try:
        connection = pyodbc.connect(
            f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
        )
        print("Connection successful!")
        return connection
    except Exception as e:
        print("Error connecting to database:", e)
        return None

# Create the table to store embeddings
def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            DROP TABLE IF EXISTS VectorStagingTable;
            CREATE TABLE VectorStagingTable(
                Id INT IDENTITY(1,1) PRIMARY KEY,
                DocumentText NVARCHAR(MAX),
                Embedding NVARCHAR(MAX)
            );
        """)
        connection.commit()
        print("Table created successfully!")
    except Exception as e:
        print("Error creating table:", e)

# Generate embeddings for each line of the file
def generate_embeddings_per_line(file_path, model):
    try:
        embeddings_data = []
        # Read the file line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Generate embeddings for each line
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                embedding = model.encode(line).tolist()
                embeddings_data.append((line, embedding))
        
        print("Embeddings generated for all lines!")
        return embeddings_data
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# Insert data into the table
def insert_data(connection, embeddings_data):
    try:
        cursor = connection.cursor()
        for document_text, embedding in embeddings_data:
            embedding_json = json.dumps(embedding)  # Convert embedding to JSON string
            cursor.execute("""
                INSERT INTO VectorStagingTable (DocumentText, Embedding)
                VALUES (?, ?)
            """, document_text, embedding_json)
        connection.commit()
        print("All data inserted successfully!")
    except Exception as e:
        print(f"Error inserting data: {e}")

# Main function
def main():
    # Connect to the database
    connection = connect_to_db()
    if connection is None:
        return
    
    # Create the VectorStagingTable table
    create_table(connection)
    
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings per line from the document
    embeddings_data = generate_embeddings_per_line('Documentation.txt', model)
    if not embeddings_data:
        print("No embeddings generated. Exiting.")
        connection.close()
        return
    
    # Insert embeddings into the database
    insert_data(connection, embeddings_data)

    # Close the connection
    connection.close()
    print("Connection closed!")

if __name__ == "__main__":
    main()
