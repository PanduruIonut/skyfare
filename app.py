import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import secrets
import smtplib
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import jwt
from fastapi import FastAPI, HTTPException, Request
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

# --- Database ---

DB_PATH = Path(__file__).parent / "skyfare.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            home_airports TEXT DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dep_code TEXT NOT NULL,
            arr_code TEXT NOT NULL,
            price REAL NOT NULL,
            checked_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ph_route ON price_history(dep_code, arr_code);
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            departure_airports TEXT NOT NULL,
            arrival_airports TEXT NOT NULL,
            target_price REAL NOT NULL,
            seat_type TEXT DEFAULT 'economy',
            max_stops TEXT DEFAULT 'any',
            created_at TEXT NOT NULL,
            last_checked TEXT,
            current_price REAL,
            triggered INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()


init_db()

# --- Auth helpers ---

JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRY_DAYS = 30


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, h = stored.split(":")
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex() == h


def create_token(user_id: int, email: str) -> str:
    return jwt.encode(
        {"uid": user_id, "email": email, "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS)},
        JWT_SECRET,
        algorithm="HS256",
    )


def get_current_user(request: Request) -> dict:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return {"id": payload["uid"], "email": payload["email"]}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


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


# --- Auth endpoints ---


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UpdateProfileRequest(BaseModel):
    home_airports: list[str] = []


@app.post("/api/register")
async def register(req: RegisterRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, hash_password(req.password), datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        user_id = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()["id"]
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        conn.close()
    return {"token": create_token(user_id, email), "user": {"id": user_id, "email": email, "home_airports": []}}


@app.post("/api/login")
async def login(req: LoginRequest):
    email = req.email.strip().lower()
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {
        "token": create_token(row["id"], email),
        "user": {"id": row["id"], "email": email, "home_airports": json.loads(row["home_airports"])},
    }


@app.get("/api/me")
async def get_me(request: Request):
    user = get_current_user(request)
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": row["id"], "email": row["email"], "home_airports": json.loads(row["home_airports"])}


@app.put("/api/me")
async def update_me(req: UpdateProfileRequest, request: Request):
    user = get_current_user(request)
    conn = get_db()
    conn.execute("UPDATE users SET home_airports = ? WHERE id = ?", (json.dumps(req.home_airports), user["id"]))
    conn.commit()
    conn.close()
    return {"ok": True}


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

GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")


def _row_to_alert(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "departure_airports": json.loads(row["departure_airports"]),
        "arrival_airports": json.loads(row["arrival_airports"]),
        "target_price": row["target_price"],
        "seat_type": row["seat_type"],
        "max_stops": row["max_stops"],
        "created_at": row["created_at"],
        "last_checked": row["last_checked"],
        "current_price": row["current_price"],
        "triggered": bool(row["triggered"]),
    }


class AlertRequest(BaseModel):
    departure_airports: list[str]
    arrival_airports: list[str]
    target_price: float
    seat_type: str = "economy"
    max_stops: str = "any"


@app.get("/api/alerts")
async def get_alerts(request: Request):
    user = get_current_user(request)
    conn = get_db()
    rows = conn.execute("SELECT * FROM alerts WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)).fetchall()
    conn.close()
    return [_row_to_alert(r) for r in rows]


@app.post("/api/alerts")
async def create_alert(req: AlertRequest, request: Request):
    user = get_current_user(request)
    dep_codes = [c.upper() for c in req.departure_airports if getattr(Airport, c.upper(), None)]
    arr_codes = [c.upper() for c in req.arrival_airports if getattr(Airport, c.upper(), None)]
    if not dep_codes:
        raise HTTPException(status_code=400, detail="At least one valid departure airport required")
    if not arr_codes:
        raise HTTPException(status_code=400, detail="At least one valid arrival airport required")

    alert_id = uuid.uuid4().hex[:8]
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db()
    conn.execute(
        "INSERT INTO alerts (id, user_id, departure_airports, arrival_airports, target_price, seat_type, max_stops, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (alert_id, user["id"], json.dumps(dep_codes), json.dumps(arr_codes), req.target_price, req.seat_type, req.max_stops, now),
    )
    conn.commit()
    conn.close()
    return {"id": alert_id, "departure_airports": dep_codes, "arrival_airports": arr_codes, "target_price": req.target_price, "seat_type": req.seat_type, "max_stops": req.max_stops, "created_at": now, "last_checked": None, "current_price": None, "triggered": False}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str, request: Request):
    user = get_current_user(request)
    conn = get_db()
    conn.execute("DELETE FROM alerts WHERE id = ? AND user_id = ?", (alert_id, user["id"]))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/alerts/{alert_id}/reactivate")
async def reactivate_alert(alert_id: str, request: Request):
    user = get_current_user(request)
    conn = get_db()
    conn.execute("UPDATE alerts SET triggered = 0 WHERE id = ? AND user_id = ?", (alert_id, user["id"]))
    conn.commit()
    conn.close()
    return {"ok": True}


def send_price_alert_email(email: str, dep_airports: list, arr_airports: list, target_price: float, cheapest_price: float, cheapest_date: str):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        log.warning("Gmail credentials not configured, skipping email")
        return

    dep_str = ", ".join(dep_airports)
    arr_str = ", ".join(arr_airports[:5])
    route = f"{dep_str} → {arr_str}"
    subject = f"SkyFare: {dep_str} flights dropped to ${cheapest_price:.0f}!"

    html = f"""<div style="font-family:sans-serif;max-width:480px">
    <h2 style="color:#16a34a">Price dropped!</h2>
    <p style="font-size:16px"><strong>{route}</strong></p>
    <p>Cheapest price found: <strong style="font-size:20px;color:#16a34a">${cheapest_price:.0f}</strong></p>
    <p>Your target was: ${target_price:.0f}</p>
    <p>Date: {cheapest_date}</p>
    <p style="color:#666;font-size:13px;margin-top:24px">— SkyFare Price Alerts</p>
    </div>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = email
    msg.attach(MIMEText(f"{route}: ${cheapest_price:.0f} on {cheapest_date} (target: ${target_price:.0f})", "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.send_message(msg)
        log.info(f"Alert email sent to {email} for {route}")
    except Exception as e:
        log.error(f"Failed to send alert email: {e}")


async def check_alerts():
    conn = get_db()
    rows = conn.execute(
        "SELECT a.*, u.email FROM alerts a JOIN users u ON a.user_id = u.id WHERE a.triggered = 0"
    ).fetchall()
    conn.close()

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")

    for row in rows:
        try:
            dep_codes = json.loads(row["departure_airports"])
            arr_codes = json.loads(row["arrival_airports"])
            dep_list = [[getattr(Airport, c), 0] for c in dep_codes if getattr(Airport, c, None)]
            arr_list = [[getattr(Airport, c), 0] for c in arr_codes if getattr(Airport, c, None)]
            if not dep_list or not arr_list:
                continue

            filters = DateSearchFilters(
                trip_type=TripType.ONE_WAY,
                passenger_info=PassengerInfo(adults=1),
                flight_segments=[FlightSegment(departure_airport=dep_list, arrival_airport=arr_list, travel_date=tomorrow)],
                seat_type=SEAT_TYPE_MAP.get(row["seat_type"], SeatType.ECONOMY),
                stops=STOPS_MAP.get(row["max_stops"], MaxStops.ANY),
                from_date=tomorrow,
                to_date=end_date,
            )

            search = SearchDates()
            results = await asyncio.to_thread(search.search, filters)

            conn = get_db()
            if results:
                results.sort(key=lambda r: r.price)
                cheapest = results[0]
                cheapest_date = cheapest.date[0].strftime("%Y-%m-%d")

                if cheapest.price <= row["target_price"]:
                    conn.execute("UPDATE alerts SET last_checked=?, current_price=?, triggered=1 WHERE id=?",
                                 (datetime.now().isoformat(timespec="seconds"), cheapest.price, row["id"]))
                    send_price_alert_email(row["email"], dep_codes, arr_codes, row["target_price"], cheapest.price, cheapest_date)
                    log.info(f"Alert triggered: {row['id']} ${cheapest.price:.0f}")
                else:
                    conn.execute("UPDATE alerts SET last_checked=?, current_price=? WHERE id=?",
                                 (datetime.now().isoformat(timespec="seconds"), cheapest.price, row["id"]))
            else:
                conn.execute("UPDATE alerts SET last_checked=? WHERE id=?",
                             (datetime.now().isoformat(timespec="seconds"), row["id"]))
            conn.commit()
            conn.close()

        except Exception as e:
            log.error(f"Error checking alert {row['id']}: {e}")


async def alert_loop():
    await asyncio.sleep(10)
    while True:
        try:
            await check_alerts()
        except Exception as e:
            log.error(f"Alert loop error: {e}")
        await asyncio.sleep(7200)


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

    # Save to price history and compute averages
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db()
    for r in results_list:
        conn.execute(
            "INSERT INTO price_history (dep_code, arr_code, price, checked_at) VALUES (?,?,?,?)",
            (dep_code.upper(), r["code"], r["price"], now),
        )
    conn.commit()

    # Compute avg from last 30 days of history
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    for r in results_list:
        row = conn.execute(
            "SELECT AVG(price) as avg_price, COUNT(*) as cnt FROM price_history WHERE dep_code=? AND arr_code=? AND checked_at>?",
            (dep_code.upper(), r["code"], cutoff),
        ).fetchone()
        avg = row["avg_price"] if row and row["avg_price"] else r["price"]
        r["avg_price"] = round(avg, 2)
        r["pct_diff"] = round((r["price"] - avg) / avg * 100) if avg > 0 else 0
    conn.close()

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
