import asyncio
import json
import logging
import os
import smtplib
import uuid
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

ALERTS_FILE = Path(__file__).parent / "alerts.json"
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")


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
    departure_airport: str
    arrival_airport: str
    target_price: float
    seat_type: str = "economy"
    max_stops: str = "any"


@app.get("/api/alerts")
async def get_alerts():
    return load_alerts()


@app.post("/api/alerts")
async def create_alert(req: AlertRequest):
    get_airport(req.departure_airport)
    get_airport(req.arrival_airport)
    alert = {
        "id": uuid.uuid4().hex[:8],
        "departure_airport": req.departure_airport.upper(),
        "arrival_airport": req.arrival_airport.upper(),
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

    route = f"{alert['departure_airport']} → {alert['arrival_airport']}"
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
    msg["To"] = GMAIL_USER
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
        if alert.get("triggered"):
            continue
        try:
            dep = getattr(Airport, alert["departure_airport"])
            arr = getattr(Airport, alert["arrival_airport"])

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
                    log.info(f"Alert triggered: {alert['departure_airport']}->{alert['arrival_airport']} ${cheapest.price:.0f}")
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


@app.on_event("startup")
async def start_alert_loop():
    asyncio.create_task(alert_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
