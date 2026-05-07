from blueprint.core.blueprint import Blueprint
from blueprint.core.feature import Feature
from blueprint.core.influence import Influence


def real_estate(n: int = 2000, seed: int = 42) -> Blueprint:
    bp = Blueprint(n=n, seed=seed)
    bp.add_feature(
        Feature("listing_id", dtype="id", style="prefixed", prefix="LST-", padding=6),
        Feature("neighborhood_tier", dtype="category", values=["A", "B", "C"], weights=[0.25, 0.5, 0.25]),
        Feature("sqft", dtype=int, base=1800, std=600, clip=(400, 8000)),
        Feature("bedrooms", dtype=int, base=3, std=1, clip=(1, 8)),
        Feature("bathrooms", dtype=float, base=2.0, std=0.8, clip=(1.0, 6.0)),
        Feature("has_pool", dtype=bool, p=0.2),
        Feature("crime_rate", dtype=float, base=50, std=20, clip=(0, 100)),
        Feature("price", dtype=float, base=300000, std=80000, clip=(50000, 5000000)),
    )
    return bp


def ecommerce(n: int = 5000, seed: int = 42, include_returns: bool = False) -> Blueprint:
    bp = Blueprint(n=n, seed=seed)
    bp.add_feature(
        Feature("order_id", dtype="id", style="prefixed", prefix="ORD-", padding=7),
        Feature("product_category", dtype="category",
                values=["Electronics", "Clothing", "Home", "Sports", "Books"],
                weights=[0.3, 0.25, 0.2, 0.15, 0.1]),
        Feature("unit_price", dtype=float, base=75, std=50, clip=(1, 2000)),
        Feature("quantity", dtype=int, base=2, std=1, clip=(1, 20)),
        Feature("discount_pct", dtype="percentage", base=0.1, std=0.1),
        Feature("revenue", dtype=float, base=150, std=80, clip=(1, 50000)),
    )
    return bp


def employee_survey(n: int = 500, seed: int = 42, departments: list = None) -> Blueprint:
    if departments is None:
        departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    bp = Blueprint(n=n, seed=seed)
    bp.add_feature(
        Feature("employee_id", dtype="id", style="sequential", start=1001),
        Feature("department", dtype="category", values=departments),
        Feature("tenure_years", dtype=float, base=5, std=3, clip=(0, 40)),
        Feature("satisfaction_score", dtype="percentage", base=0.7, std=0.2),
        Feature("performance_rating", dtype=int, base=3, std=1, clip=(1, 5)),
    )
    return bp


def web_events(n: int = 10000, seed: int = 42) -> Blueprint:
    bp = Blueprint(n=n, seed=seed)
    bp.add_feature(
        Feature("session_id", dtype="id", style="uuid4"),
        Feature("event_type", dtype="category",
                values=["pageview", "click", "scroll", "purchase", "signup"],
                weights=[0.5, 0.25, 0.15, 0.05, 0.05]),
        Feature("page_path", dtype="category",
                values=["/home", "/products", "/cart", "/checkout", "/about"]),
        Feature("duration_seconds", dtype=float, base=120, std=90, clip=(1, 1800)),
        Feature("is_mobile", dtype=bool, p=0.6),
        Feature("bounce", dtype=bool, p=0.35),
    )
    return bp
