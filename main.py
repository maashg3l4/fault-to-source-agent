import os
import time
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

def get_driver(max_retries=10, delay=3):
    """Establishes a connection with a retry loop for transient failures."""
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    for attempt in range(1, max_retries + 1):
        try:
            driver.verify_connectivity()
            print(f"✅ Successfully connected to Neo4j on attempt {attempt}.")
            return driver
        except ServiceUnavailable as e:
            print(f"⚠️  Attempt {attempt}/{max_retries} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise ConnectionError("❌ Could not connect to Neo4j after multiple retries.")

def test_connection():
    print("Connecting to Neo4j database...")
    try:
        driver = get_driver()
        with driver.session() as session:
            query = """
            MERGE (s:Supplier {name: 'Dhaka Textiles Ltd'})
            SET s.country = 'Bangladesh'
            RETURN s.name AS name
            """
            result = session.run(query)
            for record in result:
                print(f"✅ Created or found supplier node: {record['name']}")
    except Exception as e:
        print(f"❌ A critical error occurred: {e}")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    test_connection()