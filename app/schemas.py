from pydantic import BaseModel

class PredictRequest(BaseModel):
    year: int
    month: int
    day_of_year: int
    day_of_week: int
    is_harvest_season: int
    is_lean_season: int
    arrivals: float
    num_markets: int
    price_lag_7: float
    price_lag_14: float
    price_lag_30: float
    arrivals_lag_7: float
    price_roll_7: float
    price_roll_30: float
    arrivals_roll_7: float

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2025,
                "month": 4,
                "day_of_year": 100,
                "day_of_week": 2,
                "is_harvest_season": 1,
                "is_lean_season": 0,
                "arrivals": 45.5,
                "num_markets": 12,
                "price_lag_7": 19500.0,
                "price_lag_14": 19200.0,
                "price_lag_30": 18800.0,
                "arrivals_lag_7": 42.0,
                "price_roll_7": 19350.0,
                "price_roll_30": 19000.0,
                "arrivals_roll_7": 44.0
            }
        }

class PredictResponse(BaseModel):
    predicted_price: float
    unit: str
    model_name: str
    model_version: str