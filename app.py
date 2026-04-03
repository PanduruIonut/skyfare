import asyncio
import csv
import io
import json
import logging
import os
import smtplib
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("alerts")

from fli.models import (
    Airport,
    Airline,
    DateSearchFilters,
    FlightLeg,
    FlightResult,
    FlightSearchFilters,
    FlightSegment,
    MaxStops,
    PassengerInfo,
    SeatType,
    SortBy,
    TimeRestrictions,
    TripType,
)
from fli.search import SearchDates, SearchFlights

app = FastAPI()

# Pre-compute airport and airline lists once
AIRPORTS = [{"code": a.name, "name": a.value} for a in Airport]
AIRLINES = [{"code": a.name, "name": a.value} for a in Airline]

# Build country → airport mapping from OpenFlights data
AIRPORT_CODES = {a.name for a in Airport}
COUNTRY_AIRPORTS: dict[str, list[str]] = {}
COUNTRIES: list[dict] = []


def _build_country_mapping():
    data_file = Path(__file__).parent / "countries.json"
    # Try loading cached data first
    if data_file.exists():
        try:
            mapping = json.loads(data_file.read_text())
            COUNTRY_AIRPORTS.update(mapping)
            COUNTRIES.extend(
                sorted([{"name": c} for c in mapping], key=lambda x: x["name"])
            )
            return
        except Exception:
            pass

    # Download from OpenFlights
    try:
        import httpx

        resp = httpx.get(
            "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
            timeout=15,
        )
        resp.raise_for_status()
        mapping: dict[str, list[str]] = defaultdict(list)
        reader = csv.reader(io.StringIO(resp.text))
        for row in reader:
            if len(row) >= 5 and row[4] and row[4] != "\\N":
                iata = row[4].strip()
                country = row[3].strip()
                if iata in AIRPORT_CODES and country:
                    if iata not in mapping[country]:
                        mapping[country].append(iata)
        COUNTRY_AIRPORTS.update(mapping)
        COUNTRIES.extend(
            sorted([{"name": c} for c in mapping], key=lambda x: x["name"])
        )
        # Cache for next startup
        data_file.write_text(json.dumps(dict(mapping), indent=2))
    except Exception as e:
        log.warning(f"Failed to load country data: {e}")


_build_country_mapping()

SEAT_TYPE_MAP = {
    "economy": SeatType.ECONOMY,
    "premium_economy": SeatType.PREMIUM_ECONOMY,
    "business": SeatType.BUSINESS,
    "first": SeatType.FIRST,
}

STOPS_MAP = {
    "any": MaxStops.ANY,
    "nonstop": MaxStops.NON_STOP,
    "one_stop": MaxStops.ONE_STOP_OR_FEWER,
    "two_stops": MaxStops.TWO_OR_FEWER_STOPS,
}

SORT_MAP = {
    "top_flights": SortBy.TOP_FLIGHTS,
    "cheapest": SortBy.CHEAPEST,
    "departure": SortBy.DEPARTURE_TIME,
    "arrival": SortBy.ARRIVAL_TIME,
    "duration": SortBy.DURATION,
}


def get_airport(code: str) -> Airport:
    airport = getattr(Airport, code.upper(), None)
    if airport is None:
        raise HTTPException(status_code=400, detail=f"Unknown airport code: {code}")
    return airport


def serialize_leg(leg: FlightLeg) -> dict:
    return {
        "airline_code": leg.airline.name,
        "airline_name": leg.airline.value,
        "flight_number": leg.flight_number,
        "departure_airport": leg.departure_airport.name,
        "arrival_airport": leg.arrival_airport.name,
        "departure_time": leg.departure_datetime.isoformat(),
        "arrival_time": leg.arrival_datetime.isoformat(),
        "duration": leg.duration,
    }


def serialize_flight(flight: FlightResult) -> dict:
    return {
        "price": flight.price,
        "duration": flight.duration,
        "stops": flight.stops,
        "legs": [serialize_leg(leg) for leg in flight.legs],
    }


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def index():
    html = Path(__file__).parent / "index.html"
    return HTMLResponse(html.read_text())


@app.get("/api/airports")
async def airports():
    return AIRPORTS


@app.get("/api/airlines")
async def airlines():
    return AIRLINES


@app.get("/api/countries")
async def countries():
    return COUNTRIES


class FlightSearchRequest(BaseModel):
    trip_type: str = "one_way"
    departure_airport: str
    arrival_airport: str
    departure_date: str
    return_date: str | None = None
    adults: int = 1
    seat_type: str = "economy"
    max_stops: str = "any"
    sort_by: str = "cheapest"
    earliest_departure: int | None = None
    latest_departure: int | None = None
    airlines: list[str] | None = None


@app.post("/api/search/flights")
async def search_flights(req: FlightSearchRequest):
    dep = get_airport(req.departure_airport)
    arr = get_airport(req.arrival_airport)

    is_round_trip = req.trip_type == "round_trip"
    if is_round_trip and not req.return_date:
        raise HTTPException(status_code=400, detail="Return date required for round-trip")

    time_res = None
    if req.earliest_departure is not None or req.latest_departure is not None:
        time_res = TimeRestrictions(
            earliest_departure=req.earliest_departure,
            latest_departure=req.latest_departure,
        )

    segments = [
        FlightSegment(
            departure_airport=[[dep, 0]],
            arrival_airport=[[arr, 0]],
            travel_date=req.departure_date,
            time_restrictions=time_res,
        )
    ]
    if is_round_trip:
        segments.append(
            FlightSegment(
                departure_airport=[[arr, 0]],
                arrival_airport=[[dep, 0]],
                travel_date=req.return_date,
                time_restrictions=time_res,
            )
        )

    airline_filter = None
    if req.airlines:
        airline_filter = [
            a for code in req.airlines
            if (a := getattr(Airline, code, None)) is not None
        ]
        airline_filter = airline_filter or None

    filters = FlightSearchFilters(
        trip_type=TripType.ROUND_TRIP if is_round_trip else TripType.ONE_WAY,
        passenger_info=PassengerInfo(adults=req.adults),
        flight_segments=segments,
        seat_type=SEAT_TYPE_MAP.get(req.seat_type, SeatType.ECONOMY),
        stops=STOPS_MAP.get(req.max_stops, MaxStops.ANY),
        sort_by=SORT_MAP.get(req.sort_by, SortBy.CHEAPEST),
        airlines=airline_filter,
    )

    try:
        search = SearchFlights()
        results = await asyncio.to_thread(search.search, filters, 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not results:
        return {"results": []}

    serialized = []
    for r in results:
        if isinstance(r, tuple):
            serialized.append({
                "outbound": serialize_flight(r[0]),
                "return": serialize_flight(r[1]),
            })
        else:
            serialized.append({
                "outbound": serialize_flight(r),
                "return": None,
            })

    return {"results": serialized}


class DateSearchRequest(BaseModel):
    departure_airport: str
    arrival_airport: str
    from_date: str
    to_date: str
    adults: int = 1
    seat_type: str = "economy"
    max_stops: str = "any"


@app.post("/api/search/dates")
async def search_dates(req: DateSearchRequest):
    dep = get_airport(req.departure_airport)
    arr = get_airport(req.arrival_airport)

    filters = DateSearchFilters(
        trip_type=TripType.ONE_WAY,
        passenger_info=PassengerInfo(adults=req.adults),
        flight_segments=[
            FlightSegment(
                departure_airport=[[dep, 0]],
                arrival_airport=[[arr, 0]],
                travel_date=req.from_date,
            )
        ],
        seat_type=SEAT_TYPE_MAP.get(req.seat_type, SeatType.ECONOMY),
        stops=STOPS_MAP.get(req.max_stops, MaxStops.ANY),
        from_date=req.from_date,
        to_date=req.to_date,
    )

    try:
        search = SearchDates()
        results = await asyncio.to_thread(search.search, filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not results:
        return {"results": []}

    serialized = []
    for r in results:
        dates = r.date
        serialized.append({
            "date": dates[0].strftime("%Y-%m-%d"),
            "return_date": dates[1].strftime("%Y-%m-%d") if len(dates) > 1 else None,
            "price": r.price,
        })

    serialized.sort(key=lambda x: x["price"])
    return {"results": serialized}


# --- Alerts ---

DEALS_FILE = Path(__file__).parent / "deals.json"
CITY_IMAGES_FILE = Path(__file__).parent / "city_images.json"
DISCOVER_DESTINATIONS = [
    {"code": "LHR", "city": "London"}, {"code": "CDG", "city": "Paris"},
    {"code": "BCN", "city": "Barcelona"}, {"code": "FCO", "city": "Rome"},
    {"code": "AMS", "city": "Amsterdam"}, {"code": "BER", "city": "Berlin"},
    {"code": "VIE", "city": "Vienna"}, {"code": "PRG", "city": "Prague"},
    {"code": "ATH", "city": "Athens"}, {"code": "IST", "city": "Istanbul"},
    {"code": "DUB", "city": "Dublin"}, {"code": "LIS", "city": "Lisbon"},
    {"code": "CPH", "city": "Copenhagen"}, {"code": "ZRH", "city": "Zurich"},
    {"code": "BUD", "city": "Budapest"}, {"code": "WAW", "city": "Warsaw"},
    {"code": "MAD", "city": "Madrid"}, {"code": "MXP", "city": "Milan"},
    {"code": "JFK", "city": "New York"}, {"code": "LAX", "city": "Los Angeles"},
    {"code": "BKK", "city": "Bangkok"}, {"code": "NRT", "city": "Tokyo"},
    {"code": "DXB", "city": "Dubai"}, {"code": "SIN", "city": "Singapore"},
]

CITY_IMAGES: dict[str, str] = {}


def _load_city_images():
    if CITY_IMAGES_FILE.exists():
        try:
            CITY_IMAGES.update(json.loads(CITY_IMAGES_FILE.read_text()))
            return
        except Exception:
            pass
    try:
        import httpx

        for dest in DISCOVER_DESTINATIONS:
            city = dest["city"]
            if city in CITY_IMAGES:
                continue
            try:
                resp = httpx.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{city}",
                    timeout=10,
                    follow_redirects=True,
                    headers={"User-Agent": "SkyFare/1.0 (flight search app)"},
                )
                data = resp.json()
                thumb = data.get("thumbnail", {}).get("source", "")
                if thumb:
                    # Request wider version
                    thumb = thumb.replace("/50px-", "/400px-")
                    CITY_IMAGES[city] = thumb
            except Exception:
                pass
        CITY_IMAGES_FILE.write_text(json.dumps(CITY_IMAGES, indent=2))
    except Exception as e:
        log.warning(f"Failed to load city images: {e}")


_load_city_images()

ALERTS_FILE = Path(__file__).parent / "alerts.json"
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", "") or GMAIL_USER


def load_alerts() -> list[dict]:
    if not ALERTS_FILE.exists():
        return []
    try:
        return json.loads(ALERTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_alerts(alerts: list[dict]):
    ALERTS_FILE.write_text(json.dumps(alerts, indent=2))


class AlertRequest(BaseModel):
    departure_airports: list[str]
    arrival_airports: list[str]
    target_price: float
    seat_type: str = "economy"
    max_stops: str = "any"


def _migrate_alert(alert: dict) -> dict:
    """Migrate old single-airport alerts to new format."""
    if "departure_airport" in alert and "departure_airports" not in alert:
        alert["departure_airports"] = [alert.pop("departure_airport")]
        alert["arrival_country"] = "Unknown"
        alert["arrival_airports"] = [alert.pop("arrival_airport")]
    return alert


@app.get("/api/alerts")
async def get_alerts():
    alerts = load_alerts()
    return [_migrate_alert(a) for a in alerts]


@app.post("/api/alerts")
async def create_alert(req: AlertRequest):
    # Validate departure airports
    dep_codes = []
    for code in req.departure_airports:
        c = code.upper()
        if getattr(Airport, c, None) is None:
            raise HTTPException(status_code=400, detail=f"Unknown departure airport: {c}")
        dep_codes.append(c)
    if not dep_codes:
        raise HTTPException(status_code=400, detail="At least one departure airport required")

    # Validate arrival airports
    arr_codes = []
    for code in req.arrival_airports:
        c = code.upper()
        if getattr(Airport, c, None) is None:
            raise HTTPException(status_code=400, detail=f"Unknown arrival airport: {c}")
        arr_codes.append(c)
    if not arr_codes:
        raise HTTPException(status_code=400, detail="At least one arrival airport required")

    alert = {
        "id": uuid.uuid4().hex[:8],
        "departure_airports": dep_codes,
        "arrival_airports": arr_codes,
        "target_price": req.target_price,
        "seat_type": req.seat_type,
        "max_stops": req.max_stops,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "last_checked": None,
        "current_price": None,
        "triggered": False,
    }
    alerts = load_alerts()
    alerts.append(alert)
    save_alerts(alerts)
    return alert


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    alerts = load_alerts()
    alerts = [a for a in alerts if a["id"] != alert_id]
    save_alerts(alerts)
    return {"ok": True}


@app.post("/api/alerts/{alert_id}/reactivate")
async def reactivate_alert(alert_id: str):
    alerts = load_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            a["triggered"] = False
            break
    save_alerts(alerts)
    return {"ok": True}


def send_price_alert_email(alert: dict, cheapest_price: float, cheapest_date: str):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        log.warning("Gmail credentials not configured, skipping email")
        return

    alert = _migrate_alert(alert)
    dep_str = ", ".join(alert["departure_airports"])
    route = f"{dep_str} → {alert.get('arrival_country', 'Unknown')}"
    subject = f"Price Alert: {route} dropped to ${cheapest_price:.0f}!"

    html = f"""<div style="font-family:sans-serif;max-width:480px">
    <h2 style="color:#16a34a">Price dropped!</h2>
    <p style="font-size:16px"><strong>{route}</strong></p>
    <p>Cheapest price found: <strong style="font-size:20px;color:#16a34a">${cheapest_price:.0f}</strong></p>
    <p>Your target was: ${alert['target_price']:.0f}</p>
    <p>Date: {cheapest_date}</p>
    <p style="color:#666;font-size:13px;margin-top:24px">— SkyFare Price Alerts</p>
    </div>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = ALERT_RECIPIENT
    msg.attach(MIMEText(f"{route}: ${cheapest_price:.0f} on {cheapest_date} (target: ${alert['target_price']:.0f})", "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.send_message(msg)
        log.info(f"Alert email sent for {route}")
    except Exception as e:
        log.error(f"Failed to send alert email: {e}")


async def check_alerts():
    alerts = load_alerts()
    changed = False
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")

    for alert in alerts:
        _migrate_alert(alert)
        if alert.get("triggered"):
            continue
        try:
            dep_list = [
                [getattr(Airport, c), 0]
                for c in alert["departure_airports"]
                if getattr(Airport, c, None) is not None
            ]
            arr_list = [
                [getattr(Airport, c), 0]
                for c in alert["arrival_airports"]
                if getattr(Airport, c, None) is not None
            ]
            if not dep_list or not arr_list:
                log.warning(f"Alert {alert['id']}: no valid airports, skipping")
                continue

            filters = DateSearchFilters(
                trip_type=TripType.ONE_WAY,
                passenger_info=PassengerInfo(adults=1),
                flight_segments=[
                    FlightSegment(
                        departure_airport=dep_list,
                        arrival_airport=arr_list,
                        travel_date=tomorrow,
                    )
                ],
                seat_type=SEAT_TYPE_MAP.get(alert.get("seat_type", "economy"), SeatType.ECONOMY),
                stops=STOPS_MAP.get(alert.get("max_stops", "any"), MaxStops.ANY),
                from_date=tomorrow,
                to_date=end_date,
            )

            search = SearchDates()
            results = await asyncio.to_thread(search.search, filters)

            alert["last_checked"] = datetime.now().isoformat(timespec="seconds")
            changed = True

            if results:
                results.sort(key=lambda r: r.price)
                cheapest = results[0]
                alert["current_price"] = cheapest.price
                cheapest_date = cheapest.date[0].strftime("%Y-%m-%d")

                if cheapest.price <= alert["target_price"]:
                    alert["triggered"] = True
                    send_price_alert_email(alert, cheapest.price, cheapest_date)
                    dep_str = ",".join(alert["departure_airports"])
                    log.info(f"Alert triggered: {dep_str}->{alert.get('arrival_country')} ${cheapest.price:.0f}")
            else:
                log.info(f"No results for alert {alert['id']}")

        except Exception as e:
            log.error(f"Error checking alert {alert['id']}: {e}")

    if changed:
        save_alerts(alerts)


async def alert_loop():
    await asyncio.sleep(10)  # initial delay to let server start
    while True:
        try:
            await check_alerts()
        except Exception as e:
            log.error(f"Alert loop error: {e}")
        await asyncio.sleep(7200)  # check every 2 hours


# --- Discover deals ---


def load_deals() -> dict:
    if not DEALS_FILE.exists():
        return {}
    try:
        return json.loads(DEALS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_deals(deals: dict):
    DEALS_FILE.write_text(json.dumps(deals, indent=2))


async def refresh_deals(dep_code: str):
    dep = getattr(Airport, dep_code.upper(), None)
    if dep is None:
        return
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
    results_list = []

    for dest in DISCOVER_DESTINATIONS:
        if dest["code"] == dep_code.upper():
            continue
        arr = getattr(Airport, dest["code"], None)
        if arr is None:
            continue
        try:
            filters = DateSearchFilters(
                trip_type=TripType.ONE_WAY,
                passenger_info=PassengerInfo(adults=1),
                flight_segments=[
                    FlightSegment(
                        departure_airport=[[dep, 0]],
                        arrival_airport=[[arr, 0]],
                        travel_date=tomorrow,
                    )
                ],
                seat_type=SeatType.ECONOMY,
                stops=MaxStops.ANY,
                from_date=tomorrow,
                to_date=end_date,
            )
            search = SearchDates()
            res = await asyncio.to_thread(search.search, filters)
            if res:
                res.sort(key=lambda r: r.price)
                cheapest = res[0]
                results_list.append({
                    "code": dest["code"],
                    "city": dest["city"],
                    "price": cheapest.price,
                    "date": cheapest.date[0].strftime("%Y-%m-%d"),
                    "image": CITY_IMAGES.get(dest["city"], ""),
                })
                log.info(f"Discover {dep_code}->{dest['code']}: ${cheapest.price:.0f}")
        except Exception as e:
            log.error(f"Discover error {dep_code}->{dest['code']}: {e}")

    results_list.sort(key=lambda x: x["price"])
    deals = load_deals()
    deals[dep_code.upper()] = {
        "results": results_list,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_deals(deals)
    log.info(f"Discover refresh done for {dep_code}: {len(results_list)} destinations")


@app.get("/api/discover")
async def get_discover(dep: str):
    dep = dep.upper()
    if getattr(Airport, dep, None) is None:
        raise HTTPException(status_code=400, detail=f"Unknown airport: {dep}")
    deals = load_deals()
    entry = deals.get(dep)
    if entry:
        age = (datetime.now() - datetime.fromisoformat(entry["updated_at"])).total_seconds()
        if age > 43200:  # stale after 12 hours
            asyncio.create_task(refresh_deals(dep))
        return entry
    # No cache — trigger refresh in background
    asyncio.create_task(refresh_deals(dep))
    return {"results": [], "updated_at": None}


@app.post("/api/discover/refresh")
async def force_refresh_discover(dep: str):
    dep = dep.upper()
    if getattr(Airport, dep, None) is None:
        raise HTTPException(status_code=400, detail=f"Unknown airport: {dep}")
    await refresh_deals(dep)
    return load_deals().get(dep, {"results": [], "updated_at": None})


async def discover_loop():
    await asyncio.sleep(30)
    while True:
        deals = load_deals()
        for dep_code in list(deals.keys()):
            try:
                await refresh_deals(dep_code)
            except Exception as e:
                log.error(f"Discover loop error for {dep_code}: {e}")
        await asyncio.sleep(43200)  # every 12 hours


@app.on_event("startup")
async def start_background_loops():
    asyncio.create_task(alert_loop())
    asyncio.create_task(discover_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
