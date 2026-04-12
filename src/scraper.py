import requests
import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd


BASE_URL = "https://api.agmarknet.gov.in/v1/prices-and-arrivals/market-report/specific"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://agmarknet.gov.in/",
    "Origin": "https://agmarknet.gov.in"
}

PARAMS = {
    "commodityGroupId": 7,
    "commodityId": 38,
    "includeExcel": "false"
}


def get_date_range(start_date: str, end_date: str):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return dates


def parse_response(data, date: str):
    records = []

    if not isinstance(data, dict):
        print(f"  Unexpected data type for {date}: {type(data)}")
        return records

    states = data.get("states", [])

    if not states:
        print(f"  No states data for {date}")
        return records

    for state_block in states:
        if not isinstance(state_block, dict):
            continue

        state_name = state_block.get("stateName", "")

        markets = state_block.get("markets", [])

        for market_block in markets:
            if not isinstance(market_block, dict):
                continue

            market_name = market_block.get("marketName", "")

            # actual price rows are inside "data" key of each market
            rows = market_block.get("data", [])

            for item in rows:
                if not isinstance(item, dict):
                    continue

                record = {
                    "date": date,
                    "state": state_name,
                    "market": market_name,
                    "variety": item.get("variety", ""),
                    "grade": item.get("grade", ""),
                    "arrivals": item.get("arrivals", 0),
                    "unit_of_arrivals": item.get("unitOfArrivals", ""),
                    "min_price": item.get("minimumPrice", 0),
                    "max_price": item.get("maximumPrice", 0),
                    "modal_price": item.get("modalPrice", 0),
                    "unit_of_price": item.get("unitOfPrice", "Rs./Quintal"),
                    "commodity": "Cumin (Jeera)"
                }

                records.append(record)

    return records


def save_data(records: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_path, index=False)


def fetch_data_for_date(date: str):
    params = PARAMS.copy()
    params["date"] = date

    try:
        response = requests.get(
            BASE_URL,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        if response.status_code == 404:
            print(f"  No data for {date} (holiday/weekend)")
            return []

        if response.status_code == 403:
            print(f"  Access denied for {date} - check headers/cookies")
            return []

        response.raise_for_status()

        data = response.json()
        records = parse_response(data, date)
        return records

    except json.JSONDecodeError:
        print(f"  Failed to parse JSON for {date}")
        return []

    except requests.exceptions.Timeout:
        print(f"  Request timed out for {date}")
        return []

    except requests.exceptions.RequestException as e:
        print(f"  Request error for {date}: {e}")
        return []


def run_scraper(start_date: str, end_date: str, output_path: str,
                delay_seconds: float = 2.0):
    print(f"Starting scraper: {start_date} to {end_date}")
    print(f"Output: {output_path}")
    print("-" * 50)

    dates = get_date_range(start_date, end_date)
    print(f"Total dates to fetch: {len(dates)}")

    all_records = []
    success_count = 0
    empty_count = 0

    for i, date in enumerate(dates):
        records = fetch_data_for_date(date)

        if records:
            all_records.extend(records)
            success_count += 1
            print(f"  Got {len(records)} records")
        else:
            empty_count += 1

        if (i + 1) % 10 == 0 and all_records:
            save_data(all_records, output_path)
            print(f"  [Checkpoint] Saved {len(all_records)} records so far")

        time.sleep(delay_seconds)

    if all_records:
        save_data(all_records, output_path)

    print("-" * 50)
    print(f"Scraping complete.")
    print(f"Successful dates : {success_count}")
    print(f"Empty dates      : {empty_count}")
    print(f"Total records    : {len(all_records)}")
    print(f"Saved to         : {output_path}")

    return all_records


if __name__ == "__main__":
    run_scraper(
        start_date="2020-01-01",
        end_date="2025-12-31",
        output_path="data/raw/cumin_all_states_raw.csv",
        delay_seconds=2.0
    )