import pandas as pd


class AnalyticsTools:
    def __init__(self, data_path="data/sales_data.csv"):
        self.df = pd.read_csv(data_path)

    def analyze_revenue_trend(self):
        recent = self.df.tail(5)
        earlier = self.df.head(5)

        recent_avg = recent["revenue"].mean()
        earlier_avg = earlier["revenue"].mean()

        trend = "decreasing" if recent_avg < earlier_avg else "increasing"

        return {
            "trend": trend,
            "recent_avg": recent_avg,
            "earlier_avg": earlier_avg
        }

    def compare_regions(self):
        grouped = self.df.groupby("region")["revenue"].sum()
        lowest = grouped.idxmin()
        highest = grouped.idxmax()

        return {
            "lowest_region": lowest,
            "highest_region": highest
        }

    def detect_anomalies(self):
        mean = self.df["revenue"].mean()
        std = self.df["revenue"].std()

        anomalies = self.df[
            (self.df["revenue"] < mean - 2 * std) |
            (self.df["revenue"] > mean + 2 * std)
        ]

        return anomalies.to_dict(orient="records")