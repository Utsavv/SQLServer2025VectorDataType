# Implementing Vector Database with Azure SQL Server

## Overview
Searching for relevant information in vast repositories of unstructured text can be a challenge. 
This article demonstrates how to implement a vector database using Azure SQL Server. Azure SQL Server recently introduced vector data type support which is still in preview mode ([Vector data type (preview)](https://learn.microsoft.com/en-us/sql/t-sql/data-types/vector-data-type?view=azuresqldb-current&tabs=csharp-sample)). Introduction to Vector data type can boost similarity search requirements.

This article showcases the integration of the `VECTOR` datatype with embeddings generated using the `SentenceTransformer` model using Python. The system enables efficient similarity search for text-based data, making it ideal for applications like document retrieval and semantic search.

In this guide, we will break down how to use Azure Sql Server Vector data type in combination with sentence transformers library to create a semantic search solution that can effectively locate related documents based on a user query. For example, this could be used in a customer support system to find the most relevant past tickets or knowledge base articles in response to a user's question.

For complete source code, please visit [My github repository](https://github.com/Utsavv/SQLServer2025VectorDataType)

## Introduction
### Embeddings and Vector DB
Embeddings are like special codes that turn words into numbers. Think of words as different puzzle pieces, and embeddings are like a map that shows where each piece fits best. When words mean almost the same thing, their embeddings are like pieces that fit together snugly. This helps computers understand not just what words say, but what they really mean when we use them in sentences.

For example, let's take the sentence 'The cat chased the mouse.' Each word in this sentence, like 'cat' and 'mouse,' gets transformed into a set of numbers that describe its meaning. These numbers help a computer quickly find sentences with similar meanings, like 'The dog chased the rat,' even if the words are different.

Vector databases store these numbers (embeddings) in an efficient way. For instance, in our example sentence 'The cat chased the mouse,' each word ('cat', 'chased', 'mouse') would have its meaning translated into numbers by a computer. These numbers are then organized in a special database that makes it easy for the computer to quickly find similar meanings, like in the sentence 'The dog chased the rat,' even if different words are used.

Please note I am not covering how to create Azure DB. You can refer to [this link](https://learn.microsoft.com/en-us/azure/azure-sql/database/single-database-create-quickstart?view=azuresql&tabs=azure-portal) for detailed steps on creating Azure SQL DB

### Implementation Objective

In production applications, documentation is often extensive and finding information related to a specific topic can be challenging due to scattered information across various documents. This article will demonstrate how a user's question is searched within a text file, and how the vector database retrieves the closest possible matches. Searching Vector DB is incredibly powerful for applications like Q&A systems, recommendations, or any context where finding relevant information quickly is important.

To mimic this scenario, I have created a documentation text file. This article will show you how to search for information within this file. Although a simple text file is used here, the same approach can be applied to PDFs as well or it could be existing text data in your SQL DB.

To make this example more realistic, I used the SAP rule engine documentation available at [SAP Help Portal](https://help.sap.com/docs/SAP_COMMERCE/9d346683b0084da2938be8a285c0c27a/ba076fa614e549309578fba7159fe628.html) and compiled it into a single documentation text file. The text file used in this demonstration is attached to the article and can also be found in the [GitHub repository](https://github.com/Utsavv/SQLServer2025VectorDataType).

## High-Level Workflow
1. Load text documents.
2. Generate embeddings for each line using `SentenceTransformer`.
3. Store embeddings in an Azure SQL Server table.
4. Query the database to retrieve similar text based on user input.


## Overview of the Components

Our solution is composed of following components:

1.	Sentence Transformers for Embeddings: We use a pre-trained model from the sentence-transformers library to convert textual documents into numerical representations (embeddings).

2.	Azure SQL Server Vector DB for Similarity Search


## Step-by-Step Implementation

### 1. Setup Environment
#### Dependencies
Ensure the following dependencies are installed:
```bash
pip install pyodbc
pip install sentence-transformers
pip install python-dotenv
```

- `pyodbc`: A Python library to connect to ODBC databases such as Azure SQL Server.
- `sentence-transformers`: A library for generating embeddings from text.
- `python-dotenv`: Used to load environment variables from a `.env` file for security.

---

### 2. Database Connection

Connect to Azure SQL Server using `pyodbc` for database operations. Store sensitive credentials in a `.env` file for enhanced security. The `.env` file contains sensitive information like server, database, username, and password. These variables are loaded securely into the Python script using the `dotenv` library.

#### Database Connection Code
```python
import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def connect_to_db():
    """
    Establish a connection to the Azure SQL Server using credentials from environment variables.

    Returns:
        connection: A pyodbc connection object if successful; None otherwise.
    """
    try:
        # Construct the connection string using environment variables
        connection = pyodbc.connect(
            f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={os.getenv('SERVER')};PORT=1433;DATABASE={os.getenv('DATABASE')};UID={os.getenv('USERNAME')};PWD={os.getenv('PASSWORD')}"
        )
        print("Connection successful!")
        return connection
    except Exception as e:
        # Print an error message if the connection fails
        print("Error connecting to database:", e)
        return None
```

---

### 3. Create Table for Vector Storage

Create a table in Azure SQL Server to store text and its corresponding embeddings. This table is essential for performing vector similarity searches. I have used `VECTOR(384)` as data type for storing vectors. Its important that length of this data type is matches with embeddings generated. Sentence_transformers model which I chose returns 384 dimensions thats why I chose 384 length. In case if you decide to chose different method to generate embeddings then you will need to adjust length of vector data type accordingly.

#### SQL Table Schema
```sql
DROP TABLE IF EXISTS VectorDB;
CREATE TABLE VectorDB (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    DocumentText NVARCHAR(MAX),
    Embedding VECTOR(384)
);
```


#### Python Function to Create Table
```python
def create_table(connection):
    """
    Create a table named `VectorDB` in the Azure SQL Server database to store document text and vector embeddings.

    Args:
        connection: A pyodbc connection object to the database.
    """
    try:
        # Obtain a cursor object to execute SQL commands
        cursor = connection.cursor()

        # SQL command to create the table, ensuring any existing table is dropped first
        cursor.execute("""
            DROP TABLE IF EXISTS VectorDB;
            CREATE TABLE VectorDB (
                Id INT IDENTITY(1,1) PRIMARY KEY,
                DocumentText NVARCHAR(MAX),
                Embedding VECTOR(384)
            );
        """)

        # Commit the changes to the database
        connection.commit()
        print("Table created successfully!")
    except Exception as e:
        # Print an error message if table creation fails
        print("Error creating table:", e)
```

---

### 4. Generate Embeddings

Generate embeddings for textual data using the `SentenceTransformer` library. Each line of text is transformed into a numerical representation (embedding). append_data argument was added to different requirements when inserting in table and generating embeddings for user quer query. I have used pre trained model `all-MiniLM-L6-v2` for generating embeddings.

#### Embedding Function
```python
from sentence_transformers import SentenceTransformer

# Generate embeddings for each line of the input (file or text)
def generate_embeddings(input_data, append_data=True):
    """
    Generate embeddings for each line of the input, which can be either a file path or a text string.
    
    Args:
        input_data (str): Path to a file or a string containing text.
        append_data (Bool): Whether original query needs to be embedded in returned result or not

    Returns:
        list of tuples: Each tuple contains a line and its corresponding embedding.
    """
    try:
        # Load the model, in prod implementation its better to load model once.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings_data = []
        lines = []
        
        # Check if input_data is a file path or a text string
        if "\n" in input_data or not input_data.strip().endswith(('.txt', '.csv', '.json')):
            # Treat as a single text string with potential multiline content
            lines = input_data.splitlines()
        else:
            # Treat as a file path
            with open(input_data, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        
        # Generate embeddings for each line
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                embedding = model.encode(line).tolist()
                if append_data == True:
                    embeddings_data.append((line, embedding))
                else: 
                    embeddings_data=embedding
        
        print("Embeddings generated for all lines!")
        return embeddings_data
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []
```


---

### 5. Insert Data into Vector Table

The `insert_data` function saves text data and their corresponding embeddings into a table called `VectorDB` in the database. It takes a database connection and a list of tuples (each containing text and its embedding) as input. For each tuple, it converts the embedding into a JSON string and executes an SQL query to store both the text and embedding in the database. After inserting all the data, the function commits the changes to ensure they are saved permanently. If any error occurs during this process, it prints an error message.

#### Insert Function
```python
import json

def insert_data(connection, embeddings_data):
    """
    Insert document text and embeddings into the `VectorDB` table.

    Args:
        connection: A pyodbc connection object to the database.
        embeddings_data: List of tuples containing document text and embeddings.
    """
    try:
        # Obtain a cursor object
        cursor = connection.cursor()

        for document_text, embedding in embeddings_data:
            # Convert the embedding to a JSON string for storage
            embedding_json = json.dumps(embedding)

            # SQL command to insert the document text and embedding
            cursor.execute("""
                INSERT INTO VectorDB (DocumentText, Embedding)
                VALUES (?, CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(384)))
            """, document_text, embedding_json)

        # Commit the changes to the database
        connection.commit()
        print("All data inserted successfully!")
    except Exception as e:
        # Print an error message if data insertion fails
        print(f"Error inserting data: {e}")
```

---

### 6. Perform Vector Search

The `vector_search_sql` function searches the `VectorDB` table to find text entries similar to a user's query. It first converts the query into an embedding—a numerical representation of the text. This embedding is then compared to those stored in the database using the `VECTOR_DISTANCE` function, which calculates the distance between two vectors. In this function, the 'cosine' distance metric is specified. Cosine distance measures the angle between two vectors, focusing on their direction rather than magnitude. This makes it particularly effective for comparing text data, where the context (direction) is more important than the length (magnitude) of the text. By ordering the results based on this cosine distance, the function retrieves the most contextually similar text entries to the user's query. 

For instance, if the query is "How to group conditions in a rule engine?" and the database contains a document stating "The Group condition allows combining multiple rules dynamically," this document would rank higher due to its semantic similarity.

In the provided SQL query, both `similarity_score` and `distance_score` are derived from the `vector_distance` function, which calculates the distance between two vectors using a specified metric—in this case, the cosine distance. 

The `distance_score` represents the cosine distance between the query vector and the document vector stored in the database. Cosine distance quantifies the angular difference between two vectors, with a range from 0 (indicating identical vectors) to 2 (indicating vectors pointing in completely opposite directions). 

The `similarity_score` is computed as `1 - distance_score`. This transformation converts the distance into a similarity measure, where a higher value indicates greater similarity. Specifically, a `similarity_score` closer to 1 suggests that the vectors are nearly identical, while a score approaching -1 indicates they are diametrically opposed.

In summary, while `distance_score` provides a direct measure of dissimilarity between vectors, `similarity_score` offers an intuitive gauge of similarity, with higher scores denoting more closely aligned vectors. 

#### Vector Search Function
```python

def vector_search_sql(query, conn, num_results=5):
    
    # Create a cursor object
    cursor = conn.cursor()

    # Generate the query embedding for the user's search query
    user_query_embedding = generate_embeddings(query,False)
    
    # SQL query for similarity search using the function vector_distance to calculate cosine similarity
    sql_similarity_search = f"""
    SELECT TOP(?) DocumentText,
           1-vector_distance('cosine', CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(384)), embedding) AS similarity_score,
           vector_distance('cosine', CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(384)), embedding) AS distance_score
    FROM dbo.VectorDB
    ORDER BY distance_score 
    """
    
    # Print the JSON results
    json_results = json.dumps(user_query_embedding)
    
    #Execute Query
    cursor.execute(sql_similarity_search, num_results, json_results, json_results)
    results = cursor.fetchall()

    return results
```


#### Bringing It All Together

The `main()` function orchestrates a comprehensive workflow for managing and querying a vector database. It begins by establishing a connection to the database using the `connect_to_db()` function. If the connection is unsuccessful, the function exits gracefully, ensuring no further operations are attempted without a valid connection.

Upon a successful connection, the function proceeds to create the `VectorStagingTable` in the database by invoking the `create_table()` function. This setup is crucial for storing the embeddings that will be generated. The function then reads the document `'Documentation.txt'` and generates embeddings for each line using the `generate_embeddings()` function. If no embeddings are generated, it notifies the user, closes the connection, and terminates the process to prevent any erroneous data handling.

Once embeddings are successfully generated, they are inserted into the database through the `insert_data()` function, populating the `VectorStagingTable` with the necessary data for similarity searches. The script defines a list of queries and processes each by printing the query text to provide context. It then executes the `vector_search_sql()` function to retrieve the top 3 relevant results from the database. The results are displayed in a clear format, showing the document text along with similarity and distance scores, offering insights based on the indexed documents. If no relevant information is found, the script informs the user accordingly.

For example, a query like "How can you manage and simplify multiple conditions together in the Rule Engine for complex scenarios like promotions?" might yield results highlighting the grouping capabilities of the Rule Engine, which allow modular management of logic. Similarly, a query about the "Container" condition could reveal its utility in packaging reusable rules for complex scenarios, making logic reusable across various promotions. Lastly, a query on the "Group" condition could emphasize its role in combining rules dynamically based on runtime parameters.

After processing all queries, the function ensures the database connection is closed properly and confirms this action to the user by printing "Connection closed!" This structured approach facilitates efficient data management and retrieval within the vector database, ensuring that each step is executed in a logical sequence and that resources are appropriately managed throughout the process. 

```python
def main():
    # Connect to the database
    connection = connect_to_db()
    if connection is None:
        return
    
    # Create the VectorStagingTable table
    create_table(connection)   
    
    # Generate embeddings per line from the document
    embeddings_data = generate_embeddings('Documentation.txt')
    
    if not embeddings_data:
        print("No embeddings generated. Exiting.")
        connection.close()
        return
    
    # Insert embeddings into the database
    insert_data(connection, embeddings_data)
    
    queries = [
        "How can you manage and simplify multiple conditions together in the Rule Engine for complex scenarios like promotions?",
        "When might you use the 'Container' condition in the Rule Engine, and what advantage does it provide?",
        "How does the 'Group' condition enhance flexibility when working with conditions in the Rule Engine?"
    ]  

    # Print results for each query   
    for query in queries:
        print(f"\n\n\nQuery: {query}")
        results=vector_search_sql(query, connection, num_results=3)
        # Print the results
        if results:
            print("Search Results:")
            for row in results:
                print(f"\n\nDocumentText: {row[0]}, SimilarityScore: {row[1]}, DistanceScore: {row[2]}")
        else:
            print("No results found.")

    # Close the connection
    if connection and not connection.closed:
        connection.close()
        
    print("Connection closed!")

if __name__ == "__main__":
    main()
```

### Output

```plaintext
Connection successful!
Table created successfully!
Embeddings generated for all lines!
All data inserted successfully!

Query: How can you manage and simplify multiple conditions together in the Rule Engine for complex scenarios like promotions?
Embeddings generated for all lines!
Search Results:

DocumentText: To create partner-product promotions in the Rule Engine, you can use the 'Container' condition to group multiple relevant conditions together. Actions can then be defined that apply to the entire group of conditions, making it easier to manage complex promotional rules., SimilarityScore: 0.72967424427127, DistanceScore: 0.27032575572872997

DocumentText: The 'Container' condition in the Rule Engine allows users to group multiple conditions together. Actions can then be created that reference the entire container, making it easier to manage complex rules. This is especially useful in scenarios like partner-product promotions where multiple conditions need to be evaluated together., SimilarityScore: 0.7019168589855438, DistanceScore: 0.2980831410144562

DocumentText: every rule has two parts: conditions and actions. In addition to a number of default conditions and actions, you can create custom conditions and actions., SimilarityScore: 0.6430638023014732, DistanceScore: 0.3569361976985268

Query: When might you use the 'Container' condition in the Rule Engine, and what advantage does it provide?
Embeddings generated for all lines!
Search Results:

DocumentText: The 'Container' condition in the Rule Engine allows users to group multiple conditions together. Actions can then be created that reference the entire container, making it easier to manage complex rules. This is especially useful in scenarios like partner-product promotions where multiple conditions need to be evaluated together., SimilarityScore: 0.7617857467894863, DistanceScore: 0.2382142532105137

DocumentText: The Rule Engine includes several 'Out of the Box' conditions by default, such as 'Rule executed,' 'Group,' and 'Container.' The 'Rule executed' condition allows for the creation of dependencies between rules. The 'Group' condition helps in changing logical operators between rules from AND to OR, and the 'Container' condition allows grouping other conditions to reference them collectively., SimilarityScore: 0.7058000661790532, DistanceScore: 0.2941999338209468

DocumentText: Examples of 'Out of the Box' conditions in the Rule Engine include 'Rule executed,' which manages dependencies between rules, 'Group,' which changes the logical operator between conditions, and 'Container,' which allows multiple conditions to be grouped together for ease of management., SimilarityScore: 0.6129632443577911, DistanceScore: 0.38703675564220885

Query: How does the 'Group' condition enhance flexibility when working with conditions in the Rule Engine?
Embeddings generated for all lines!
Search Results:

DocumentText: The 'Group' condition in the Rule Engine allows users to change the logical operator between conditions from the default AND to OR. This provides more flexibility in how conditions are evaluated within a rule, allowing users to create more sophisticated and varied decision-making criteria., SimilarityScore: 0.852246103392024, DistanceScore: 0.147753896607976

DocumentText: The Rule Engine includes several 'Out of the Box' conditions by default, such as 'Rule executed,' 'Group,' and 'Container.' The 'Rule executed' condition allows for the creation of dependencies between rules. The 'Group' condition helps in changing logical operators between rules from AND to OR, and the 'Container' condition allows grouping other conditions to reference them collectively., SimilarityScore: 0.7611163306001254, DistanceScore: 0.2388836693998746

DocumentText: The 'Container' condition in the Rule Engine allows users to group multiple conditions together. Actions can then be created that reference the entire container, making it easier to manage complex rules. This is especially useful in scenarios like partner-product promotions where multiple conditions need to be evaluated together., SimilarityScore: 0.6750005861379249, DistanceScore: 0.3249994138620751

Connection closed!
``` 

---

## Conclusion

This article provides a comprehensive guide to implementing a vector database using Azure SQL Server's new `VECTOR` datatype and integrating it with the `SentenceTransformer` Python library for semantic search. It demonstrates how to efficiently store and query textual embeddings for applications like document retrieval, Q&A systems, and semantic similarity searches. By following the step-by-step implementation, you can:

1. **Leverage Modern SQL Capabilities**: Utilize Azure SQL Server's preview `VECTOR` datatype to handle complex similarity searches natively in the database.

2. **Generate High-Quality Embeddings**: Transform textual data into numerical embeddings using pre-trained models from the `sentence-transformers` library, such as `all-MiniLM-L6-v2`.

3. **Perform Semantic Search**: Enable quick and accurate retrieval of relevant documents based on user queries using cosine similarity.

This guide is especially valuable for scenarios where information is scattered across large datasets or documents. By applying these techniques, you can significantly enhance search capabilities in systems like customer support platforms, knowledge management systems, and recommendation engines.

For further exploration, consider experimenting with different embedding models and optimizing database queries to suit your application's unique requirements.