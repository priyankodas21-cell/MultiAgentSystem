import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import re
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from openai import OpenAI
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Set OPENAI_API_KEY in your environment")

openai_client = OpenAI(
    api_key=openai_api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
)


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
# Generated by GitHub Copilot - Oct-09-2026
def inspect_inventory(as_of_date: Union[str, datetime]) -> Dict[str, Dict[str, Union[int, bool]]]:
    """Return each catalog item's stock and whether it has reached its reorder threshold."""
    date_str = as_of_date.isoformat() if isinstance(as_of_date, datetime) else as_of_date
    stock_by_item = get_all_inventory(date_str)
    inventory_catalog = pd.read_sql(
        "SELECT item_name, min_stock_level FROM inventory",
        db_engine,
    )

    inventory_status = {}
    for item in inventory_catalog.itertuples(index=False):
        current_stock = int(stock_by_item.get(item.item_name, 0))
        min_stock_level = int(item.min_stock_level)
        inventory_status[item.item_name] = {
            "current_stock": current_stock,
            "min_stock_level": min_stock_level,
            "reorder_recommended": current_stock <= min_stock_level,
        }

    return inventory_status


def check_stock_for_request(
    item_name: str,
    requested_quantity: int,
    as_of_date: Union[str, datetime],
) -> Dict[str, Union[str, int, bool]]:
    """Validate an exact catalog item and report whether its requested quantity is in stock."""
    if not isinstance(requested_quantity, int) or isinstance(requested_quantity, bool) or requested_quantity <= 0:
        raise ValueError("Requested quantity must be a positive integer")

    inventory_item = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": item_name},
    )
    if inventory_item.empty:
        raise ValueError(f"Unknown inventory item: {item_name}")

    date_str = as_of_date.isoformat() if isinstance(as_of_date, datetime) else as_of_date
    stock_info = get_stock_level(item_name, date_str)
    current_stock = int(stock_info["current_stock"].iloc[0])
    min_stock_level = int(inventory_item["min_stock_level"].iloc[0])

    return {
        "item_name": item_name,
        "requested_quantity": requested_quantity,
        "current_stock": current_stock,
        "available": current_stock >= requested_quantity,
        "reorder_recommended": current_stock <= min_stock_level,
    }


# Tools for quoting agent
# Generated by GitHub Copilot - Oct-09-2026
def find_relevant_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """Return historical quotes matching non-empty search terms for pricing reference."""
    if not search_terms or any(not isinstance(term, str) or not term.strip() for term in search_terms):
        raise ValueError("Provide at least one non-empty historical quote search term")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("Limit must be a positive integer")

    return search_quote_history([term.strip() for term in search_terms], limit)


def calculate_quote(
    items: List[Dict[str, Union[str, int]]],
    discount_percent: float = 0.0,
) -> Dict:
    """Price exact catalog items and apply only the explicitly supplied discount percentage."""
    if not items:
        raise ValueError("A quote must contain at least one item")
    if (
        not isinstance(discount_percent, (int, float))
        or isinstance(discount_percent, bool)
        or not np.isfinite(discount_percent)
        or not 0 <= discount_percent <= 100
    ):
        raise ValueError("Discount percentage must be a finite number from 0 to 100")

    catalog_prices = {item["item_name"]: item["unit_price"] for item in paper_supplies}
    quote_lines = []
    subtotal = 0.0

    for quote_item in items:
        item_name = quote_item.get("item_name")
        quantity = quote_item.get("quantity")
        if not isinstance(item_name, str) or item_name not in catalog_prices:
            raise ValueError(f"Unknown catalog item: {item_name}")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            raise ValueError(f"Quantity for {item_name} must be a positive integer")

        unit_price = float(catalog_prices[item_name])
        line_total = unit_price * quantity
        subtotal += line_total
        quote_lines.append({
            "item_name": item_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": round(line_total, 2),
        })

    discount_amount = round(subtotal * float(discount_percent) / 100, 2)
    return {
        "items": quote_lines,
        "subtotal": round(subtotal, 2),
        "discount_percent": float(discount_percent),
        "discount_amount": discount_amount,
        "total": round(subtotal - discount_amount, 2),
    }


# Tools for ordering agent
# Generated by GitHub Copilot - Oct-10-2026
def get_ordering_financial_status(as_of_date: Union[str, datetime]) -> Dict:
    """Return the existing financial report for internal order-planning decisions."""
    return generate_financial_report(as_of_date)


def record_customer_sale(
    item_name: str,
    quantity: int,
    sale_total: float,
    sale_date: Union[str, datetime],
) -> int:
    """Record a sale only when the named inventory item has enough units in stock."""
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ValueError("Sale quantity must be a positive integer")
    if (
        not isinstance(sale_total, (int, float))
        or isinstance(sale_total, bool)
        or not np.isfinite(sale_total)
        or sale_total < 0
    ):
        raise ValueError("Sale total must be a finite, non-negative amount")

    sale_date_str = sale_date.isoformat() if isinstance(sale_date, datetime) else sale_date
    stock_check = check_stock_for_request(item_name, quantity, sale_date_str)
    if not stock_check["available"]:
        raise ValueError(
            f"Insufficient stock for {item_name}: "
            f"{stock_check['current_stock']} available, {quantity} requested"
        )

    return create_transaction(
        item_name,
        "sales",
        quantity,
        float(sale_total),
        sale_date_str,
    )


def place_restock_order(
    item_name: str,
    quantity: int,
    order_date: Union[str, datetime],
) -> Dict[str, Union[str, int, float]]:
    """Record a funded restock for an inventoried item, dated at its estimated arrival."""
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ValueError("Restock quantity must be a positive integer")

    inventory_catalog = pd.read_sql(
        "SELECT item_name, unit_price FROM inventory",
        db_engine,
    )
    requested_name = item_name.strip().casefold()
    exact_matches = inventory_catalog[
        inventory_catalog["item_name"].str.casefold() == requested_name
    ]
    if exact_matches.empty:
        base_name = item_name.partition("(")[0].strip().casefold()
        exact_matches = inventory_catalog[
            inventory_catalog["item_name"].map(
                lambda catalog_name: catalog_name.partition("(")[0].strip().casefold() == base_name
            )
        ]
    if len(exact_matches) != 1:
        raise ValueError(f"Item is not part of the current inventory: {item_name}")
    item_name = str(exact_matches.iloc[0]["item_name"])

    order_date_str = order_date.isoformat() if isinstance(order_date, datetime) else order_date
    try:
        parsed_order_date = datetime.fromisoformat(order_date_str)
    except (TypeError, ValueError) as exc:
        raise ValueError("Order date must be a valid ISO-formatted date") from exc

    unit_price = float(exact_matches["unit_price"].iloc[0])
    order_cost = round(unit_price * quantity, 2)
    available_cash = get_cash_balance(order_date_str)
    if order_cost > available_cash:
        raise ValueError(
            f"Insufficient funds for this restock: cost is {order_cost:.2f}"
        )

    delivery_date = get_supplier_delivery_date(order_date_str, quantity)
    transaction_id = create_transaction(
        item_name,
        "stock_orders",
        quantity,
        order_cost,
        delivery_date,
    )
    return {
        "transaction_id": transaction_id,
        "item_name": item_name,
        "quantity": quantity,
        "order_cost": order_cost,
        "estimated_delivery_date": delivery_date,
    }


# Generated by GitHub Copilot - Oct-11-2026
active_request_context: Dict[str, object] = {
    "request_id": None,
    "request_date": None,
    "request_text": None,
    "actions": [],
    "items": {},
}


# Generated by GitHub Copilot - Oct-13-2026
def reset_agent_transaction_audit() -> None:
    """Create and clear the application-owned mapping from request IDs to SQLite row IDs."""
    with db_engine.begin() as connection:
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS agent_transaction_audit (
                request_id INTEGER NOT NULL,
                transaction_rowid INTEGER NOT NULL,
                item_name TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                units INTEGER NOT NULL,
                price REAL NOT NULL,
                transaction_date TEXT NOT NULL,
                PRIMARY KEY (request_id, transaction_rowid)
            )
            """
        ))
        connection.execute(text("DELETE FROM agent_transaction_audit"))


def record_request_transaction(
    request_id: int,
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    transaction_date: str,
) -> int:
    """Use the starter transaction helper and reliably audit its inserted SQLite row."""
    with db_engine.connect() as connection:
        rowid_before = int(connection.execute(
            text("SELECT COALESCE(MAX(_rowid_), 0) FROM transactions")
        ).scalar_one())

    create_transaction(item_name, transaction_type, quantity, price, transaction_date)

    with db_engine.begin() as connection:
        rowids = connection.execute(
            text("SELECT _rowid_ FROM transactions WHERE _rowid_ > :rowid_before ORDER BY _rowid_"),
            {"rowid_before": rowid_before},
        ).scalars().all()
        if len(rowids) != 1:
            raise RuntimeError("Could not uniquely identify the inserted transaction")
        transaction_rowid = int(rowids[0])
        connection.execute(
            text(
                """
                INSERT INTO agent_transaction_audit
                    (request_id, transaction_rowid, item_name, transaction_type, units, price, transaction_date)
                VALUES
                    (:request_id, :transaction_rowid, :item_name, :transaction_type, :units, :price, :transaction_date)
                """
            ),
            {
                "request_id": request_id,
                "transaction_rowid": transaction_rowid,
                "item_name": item_name,
                "transaction_type": transaction_type,
                "units": quantity,
                "price": price,
                "transaction_date": transaction_date,
            },
        )
    return transaction_rowid


# Generated by GitHub Copilot - Oct-13-2026
def item_quantity_is_explicitly_requested(
    request_text: str,
    item_name: str,
    quantity: int,
) -> bool:
    """Require the exact catalog phrase and quantity to occur together in the source request."""
    request_lower = request_text.casefold()
    item_lower = item_name.casefold()
    quantity_prefix = re.compile(
        r"(?P<quantity>\d[\d,]*)\s*(?:sheets?|reams?|rolls?|packets?|"
        r"boxes?|packs?|pieces?|units?|cards?|cups?|plates?|napkins?)?\s*(?:of\s+)?$",
        re.IGNORECASE,
    )
    valid_suffix = re.compile(r"^(?:$|[,;.!?\r\n]|\s+(?:and|or|for|by|to)\b)", re.IGNORECASE)

    start_at = 0
    while True:
        item_start = request_lower.find(item_lower, start_at)
        if item_start < 0:
            return False

        item_end = item_start + len(item_lower)
        suffix = request_lower[item_end:]
        suffix_match = valid_suffix.match(suffix)
        prefix = request_lower[max(0, item_start - 100):item_start]
        quantity_match = quantity_prefix.search(prefix)
        if (
            suffix_match is not None
            and quantity_match is not None
            and int(quantity_match.group("quantity").replace(",", "")) == quantity
        ):
            return True
        start_at = item_end


# Set up your agents and create an orchestration agent that will manage them.
# Generated by GitHub Copilot - Oct-10-2026
def build_multi_agent_system(model_id: str = "gpt-4o-mini"):
    """Create inventory, quoting, and order agents with an orchestrator that delegates requests.

    Args:
        model_id: Model identifier accepted by the configured OpenAI-compatible endpoint.

    Returns:
        The orchestrator agent, configured to delegate to the three worker agents.
    """
    

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Provide the model ID supported by the configured API endpoint")

    model = OpenAIServerModel(
        model_id=model_id.strip(),
        api_base=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
        api_key=openai_api_key,
    )

    # Generated by GitHub Copilot - Oct-11-2026
    @tool
    def inventory_snapshot() -> str:
        """Get current stock and reorder flags for every catalog item.
        """
        request_date = active_request_context["request_date"]
        if not isinstance(request_date, str):
            return "Inventory is unavailable because no request date is active."
        return str(inspect_inventory(request_date))

    @tool
    def requested_item_stock(item_name: str, requested_quantity: int) -> str:
        """Check whether an exact catalog item and quantity are available.

        Args:
            item_name: Exact item name from the company catalog.
            requested_quantity: Positive whole number of units requested.
        """
        request_date = active_request_context["request_date"]
        if not isinstance(request_date, str):
            return "Inventory is unavailable because no request date is active."
        return str(check_stock_for_request(item_name, requested_quantity, request_date))

    @tool
    def historical_quote_search(search_terms: List[str], limit: int = 5) -> str:
        """Find historical quotes relevant to the supplied search terms.

        Args:
            search_terms: Non-empty terms describing products or the request context.
            limit: Maximum number of matching quote records to return.
        """
        return str(find_relevant_quote_history(search_terms, limit))

    @tool
    def catalog_quote(items: List[Dict[str, Union[str, int]]], discount_percent: float = 0.0) -> str:
        """Calculate an itemized quote from exact catalog names and quantities.

        Args:
            items: List of dictionaries with exact item_name and positive integer quantity fields.
            discount_percent: Explicit discount percentage; use zero unless an authorized discount applies.
        """
        return str(calculate_quote(items, discount_percent))

    # Generated by GitHub Copilot - Oct-11-2026
    @tool
    def fulfill_order_item(item_name: str, quantity: int) -> str:
        """Validate, price, and fulfill one explicitly requested catalog item.

        Stock, price, cash, and dates are checked by the tool. Customer orders are
        recorded only after the required stock is available or a funded restock is
        scheduled. Repeated calls for the same request item do not create duplicates.

        Args:
            item_name: Catalog item name requested by the customer.
            quantity: Positive whole number of units requested.
        """
        request_id = active_request_context["request_id"]
        request_date = active_request_context["request_date"]
        request_text = active_request_context["request_text"]
        request_actions = active_request_context["actions"]
        processed_items = active_request_context["items"]
        if (
            not isinstance(request_id, int)
            or not isinstance(request_date, str)
            or not isinstance(request_text, str)
        ):
            return json.dumps({"status": "declined", "reason": "No active dated request."})
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            return json.dumps({"status": "declined", "reason": "Quantity must be a positive whole number."})

        inventory_catalog = pd.read_sql(
            "SELECT item_name, unit_price FROM inventory",
            db_engine,
        )
        requested_name = item_name.strip().casefold()
        matches = inventory_catalog[inventory_catalog["item_name"].str.casefold() == requested_name]
        if matches.empty:
            base_name = item_name.partition("(")[0].strip().casefold()
            matches = inventory_catalog[
                inventory_catalog["item_name"].map(
                    lambda catalog_name: catalog_name.partition("(")[0].strip().casefold() == base_name
                )
            ]

        if len(matches) != 1:
            outcome = {
                "request_id": request_id,
                "item_name": item_name,
                "quantity": quantity,
                "status": "declined",
                "reason": "The item does not uniquely match an item in the seeded inventory.",
            }
            request_actions.append(outcome)
            return json.dumps(outcome)

        canonical_name = str(matches.iloc[0]["item_name"])
        if not item_quantity_is_explicitly_requested(request_text, canonical_name, quantity):
            outcome = {
                "request_id": request_id,
                "item_name": canonical_name,
                "quantity": quantity,
                "status": "declined",
                "reason": "The exact catalog item and quantity were not both stated in the customer request.",
            }
            request_actions.append(outcome)
            return json.dumps(outcome)

        previous_outcome = processed_items.get(canonical_name)
        if previous_outcome is not None:
            if previous_outcome["quantity"] == quantity:
                return json.dumps(previous_outcome)
            return json.dumps({
                "request_id": request_id,
                "item_name": canonical_name,
                "quantity": quantity,
                "status": "declined",
                "reason": "A different quantity for this item was already processed for this request.",
            })

        try:
            generate_financial_report(request_date)
            quote = calculate_quote([{"item_name": canonical_name, "quantity": quantity}])
            current_stock = int(
                get_stock_level(canonical_name, request_date)["current_stock"].iloc[0]
            )
            shortage = max(0, quantity - current_stock)
            outcome = {
                "request_id": request_id,
                "item_name": canonical_name,
                "quantity": quantity,
                "unit_price": quote["items"][0]["unit_price"],
                "sale_total": quote["total"],
                "stock_at_request": current_stock,
                "status": "fulfilled",
                "transaction_ids": [],
            }

            if shortage:
                restock_cost = round(float(matches.iloc[0]["unit_price"]) * shortage, 2)
                available_cash = get_cash_balance(request_date)
                outcome["status"] = "declined"
                outcome["reason"] = (
                    f"Only {current_stock} units are available on the request date; "
                    f"{quantity} were requested. No transaction was recorded."
                )
                outcome["restock_recommended"] = True
                outcome["restock_quantity"] = shortage
                outcome["estimated_delivery_date"] = get_supplier_delivery_date(
                    request_date,
                    shortage,
                )
                outcome["restock_fundable"] = restock_cost <= available_cash
            else:
                sale_id = record_request_transaction(
                    request_id,
                    canonical_name,
                    "sales",
                    quantity,
                    quote["total"],
                    request_date,
                )
                outcome["transaction_ids"].append(sale_id)

        except (ValueError, TypeError) as exc:
            outcome = {
                "request_id": request_id,
                "item_name": canonical_name,
                "quantity": quantity,
                "status": "declined",
                "reason": str(exc),
            }
        except Exception:
            outcome = {
                "request_id": request_id,
                "item_name": canonical_name,
                "quantity": quantity,
                "status": "declined",
                "reason": "The transaction could not be safely recorded.",
            }

        processed_items[canonical_name] = outcome
        request_actions.append(outcome)
        return json.dumps(outcome)

    inventory_agent = ToolCallingAgent(
        name="inventory_agent",
        description="Checks available inventory and identifies items at or below reorder levels.",
        tools=[inventory_snapshot, requested_item_stock],
        model=model,
        instructions="Handle inventory questions only. Use exact catalog item names and the request date for stock checks. Report quantities and availability; do not quote prices or record transactions.",
        max_steps=5,
    )
    quoting_agent = ToolCallingAgent(
        name="quoting_agent",
        description="Finds historical quote examples and calculates itemized prices from the catalog.",
        tools=[historical_quote_search, catalog_quote],
        model=model,
        instructions="Handle pricing only. Use catalog prices and provide an itemized rationale. Do not invent discount rules; use a discount only when explicitly authorized, otherwise use zero. Do not check or mutate inventory.",
        max_steps=5,
    )
    ordering_agent = ToolCallingAgent(
        name="ordering_agent",
        description="Validates itemized customer orders against seeded inventory and records only verified sales or funded restocks.",
        tools=[fulfill_order_item],
        model=model,
        instructions="Handle fulfillment only. Process each exact requested catalog item once using fulfill_order_item. Do not invent dates, prices, stock, cash balances, or completed transactions. Use the returned status and delivery date as the sole source of fulfillment claims. Never disclose internal cash balances or technical errors.",
        max_steps=6,
    )

    return ToolCallingAgent(
        name="orchestrator",
        description="Routes customer requests to the inventory, quoting, and ordering specialists and combines their findings.",
        tools=[],
        managed_agents=[inventory_agent, quoting_agent, ordering_agent],
        model=model,
        instructions="Coordinate the workers. For each customer order, ask the inventory agent to verify exact catalog items, ask the quoting agent for catalog prices, and ask the ordering agent to process each clearly identified catalog item exactly once with fulfill_order_item. Do not claim an order was placed or fulfilled unless the ordering tool returns that status. Do not invent substitute items, dates, prices, or stock. Explain unavailable or unrecognized items as not fulfilled. Never disclose internal cash or financial reports or internal tool errors.",
        max_steps=10,
    )


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    reset_agent_transaction_audit()
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%Y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    # Generated by GitHub Copilot - Oct-11-2026
    orchestrator = build_multi_agent_system()

    # Generated by GitHub Copilot - Oct-11-2026
    def call_your_multi_agent_system(
        request: str,
        request_id: int,
        request_date: str,
    ) -> Dict[str, Union[str, List[Dict]]]:
        """Run one request and return a customer response backed by recorded tool outcomes."""
        active_request_context.update({
            "request_id": request_id,
            "request_date": request_date,
            "request_text": request,
            "actions": [],
            "items": {},
        })
        try:
            orchestrator.run(request)
        except Exception as exc:
            print(f"Request {request_id} agent run failed: {type(exc).__name__}")
            actions = active_request_context["actions"]
            if not actions:
                return {
                    "response": "We could not verify or record this order. Please review the request and try again.",
                    "fulfillment_status": "system_error",
                    "fulfillment_reason": "The agent run failed before any validated order action was recorded.",
                    "fulfillment_actions": [],
                }

        actions = active_request_context["actions"]
        successful_actions = [
            action for action in actions
            if action.get("status") in {"fulfilled", "scheduled"}
        ]
        declined_actions = [action for action in actions if action.get("status") == "declined"]

        if not actions:
            fulfillment_status = "not_processed"
            fulfillment_reason = (
                "The agents did not submit an item and quantity that could be validated "
                "against the request and seeded catalog; no transaction was recorded."
            )
            response = (
                "We could not process this request because no item and quantity were "
                "validated against the catalog. No order was recorded."
            )
        elif successful_actions and declined_actions:
            fulfillment_status = "partially_fulfilled"
            fulfillment_reason = "Some requested items were fulfilled; other items could not be fulfilled."
        elif successful_actions and all(action["status"] == "scheduled" for action in successful_actions):
            fulfillment_status = "scheduled"
            fulfillment_reason = "All processed items are scheduled, not yet delivered."
        elif successful_actions:
            fulfillment_status = "fulfilled"
            fulfillment_reason = "The recorded fulfilled actions passed inventory and transaction checks."
        else:
            fulfillment_status = "declined"
            fulfillment_reason = "; ".join(
                dict.fromkeys(action.get("reason", "Request item was declined.") for action in declined_actions)
            )

        if actions:
            response_lines = []
            for action in actions:
                item_name = action["item_name"]
                quantity = action["quantity"]
                if action["status"] == "fulfilled":
                    response_lines.append(
                        f"Fulfilled {quantity} units of {item_name} for "
                        f"${action['sale_total']:.2f}."
                    )
                elif action["status"] == "scheduled":
                    response_lines.append(
                        f"Scheduled {quantity} units of {item_name} for "
                        f"${action['sale_total']:.2f}; estimated delivery "
                        f"{action['estimated_delivery_date']}."
                    )
                else:
                    restock_note = ""
                    if action.get("restock_recommended"):
                        funding_message = (
                            "estimated restock cost is within available funds"
                            if action.get("restock_fundable")
                            else "estimated restock cost exceeds available funds"
                        )
                        restock_note = (
                            f" A restock could be considered; estimated delivery is "
                            f"{action['estimated_delivery_date']} and the {funding_message}."
                        )
                    response_lines.append(
                        f"Could not fulfill {quantity} units of {item_name}: "
                        f"{action['reason']}{restock_note}"
                    )
            response = "\n".join(response_lines)

        return {
            "response": response,
            "fulfillment_status": fulfillment_status,
            "fulfillment_reason": fulfillment_reason,
            "fulfillment_actions": actions,
        }

    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"
        cash_before_request = get_cash_balance(request_date)

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        request_outcome = call_your_multi_agent_system(
            request_with_date,
            idx + 1,
            request_date,
        )
        response = request_outcome["response"]

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]
        cash_change = round(current_cash - cash_before_request, 2)

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance_before": cash_before_request,
                "cash_balance": current_cash,
                "cash_change": cash_change,
                "inventory_value": current_inventory,
                "response": response,
                "fulfillment_status": request_outcome["fulfillment_status"],
                "fulfillment_reason": request_outcome["fulfillment_reason"],
                "fulfillment_actions": json.dumps(request_outcome["fulfillment_actions"]),
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv("test_results.csv", index=False)
    except PermissionError:
        fallback_results_path = "test_results_updated.csv"
        results_df.to_csv(fallback_results_path, index=False)
        print(f"Could not write test_results.csv; saved this run to {fallback_results_path} instead.")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
