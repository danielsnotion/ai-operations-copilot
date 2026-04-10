import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_data(num_rows, filename):

    regions = ["North", "South", "East", "West"]
    products = ["Product_A", "Product_B", "Product_C"]
    categories = ["Electronics", "Clothing", "Home"]
    segments = ["Premium", "Standard"]

    start_date = datetime(2024, 1, 1)

    data = []

    for i in range(num_rows):

        date = start_date + timedelta(days=i % 90)

        region = random.choice(regions)
        product = random.choice(products)
        category = random.choice(categories)
        segment = random.choice(segments)

        base_orders = random.randint(50, 150)

        # 🔥 Inject anomaly
        if date.day in [10, 11] and product == "Product_A" and region == "North":
            orders = int(base_orders * 0.6)  # drop
            refunds = int(base_orders * 0.2)  # spike
        else:
            orders = base_orders
            refunds = random.randint(2, 8)

        price = random.randint(80, 120)
        revenue = orders * price - refunds * price

        data.append([
            date.strftime("%Y-%m-%d"),
            region,
            product,
            category,
            orders,
            revenue,
            refunds,
            price,
            segment
        ])

    df = pd.DataFrame(data, columns=[
        "date", "region", "product", "category",
        "orders", "revenue", "refunds", "price", "customer_segment"
    ])

    df.to_csv(filename, index=False)
    print(f"Generated {filename}")


if __name__ == "__main__":
    generate_data(300, "data/ai-operations-copilot-data/data_small.csv")
    generate_data(10000, "data/ai-operations-copilot-data/data_medium.csv")
    generate_data(1000000, "data/ai-operations-copilot-data/data_large.csv")
