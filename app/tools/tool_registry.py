from app.tools.analytics_tools import AnalyticsTools


class ToolRegistry:
    def __init__(self):
        self.tools = AnalyticsTools()

    def get_tool(self, tool_name):
        mapping = {
            "analyze_revenue_trend": self.tools.analyze_revenue_trend,
            "compare_regions": self.tools.compare_regions,
            "detect_anomalies": self.tools.detect_anomalies
        }
        return mapping.get(tool_name)