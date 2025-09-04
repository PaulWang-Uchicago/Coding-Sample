#!/bin/bash
import argparse
import json
import mysql.connector
import requests
from bs4 import BeautifulSoup

def parse_book_page(html):
    """Extract title, price, stock, rating, description from a book page."""
    soup = BeautifulSoup(html, "html.parser")
    # Title
    title = soup.h1.text.strip()
    # Price
    price = soup.select_one(".price_color").text.strip()
    # Stock
    stock = soup.select_one(".instock.availability").text.strip()
    # Rating lives in the class
    rating = next(c for c in soup.select_one(".star-rating")["class"] if c != "star-rating")
    # Description
    desc_tag = soup.find(id="product_description")
    if desc_tag:
        description = desc_tag.find_next_sibling("p").text.strip()
    else:
        description = ""
    return title, price, stock, rating, description

def main():
    parser = argparse.ArgumentParser(
        description="Scrape book data and insert into MySQL database"
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Path to JSON file listing URLs to scrape"
    )
    parser.add_argument(
        "--db_username", required=True,
        help="MySQL database username"
    )
    parser.add_argument(
        "--db_password", required=True,
        help="MySQL database password"
    )
    parser.add_argument(
        "--db_endpoint", required=True,
        help="MySQL endpoint (host)"
    )
    parser.add_argument(
        "--db_port", default="3306",
        help="MySQL port (default: 3306)"
    )
    parser.add_argument(
        "--db_name", required=True,
        help="Name of the MySQL database to use"
    )
    args = parser.parse_args()

    # Load URLs
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "urls" in data:
        urls = data["urls"]
    elif isinstance(data, list):
        urls = data
    else:
        raise ValueError("Input JSON must be a list or a dict with a 'urls' key")

    # Connect to MySQL
    conn = mysql.connector.connect(
        host=args.db_endpoint,
        port=int(args.db_port),
        user=args.db_username,
        password=args.db_password,
        database=args.db_name
    )
    cursor = conn.cursor()

    # Ensure the table exists
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title TEXT,
            price VARCHAR(64),
            stock VARCHAR(64),
            rating VARCHAR(32),
            description TEXT
        );
        """
    )
    conn.commit()

    # Scrape and insert each book
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            title, price, stock, rating, description = parse_book_page(resp.text)
            cursor.execute(
                "INSERT INTO books (title, price, stock, rating, description) "
                "VALUES (%s, %s, %s, %s, %s)",
                (title, price, stock, rating, description)
            )
            conn.commit()
            print(f"Inserted data for {url}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
