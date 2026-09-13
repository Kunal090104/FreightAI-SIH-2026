import os
import pandas as pd
import searoute as sr


WPI_PATH = "data/raw/UpdatedPub150.csv"
OUTPUT_PATH = "data/raw/routes.csv"
MISSING_PATH = "data/raw/routes_missing_ports.csv"


ORIGIN_PORTS = [
    ("Australia", "Port Hedland"),
    ("Australia", "Newcastle"),
    ("Australia", "Hay Point"),
    ("Australia", "Gladstone"),

    ("Indonesia", "Balikpapan"),
    ("Indonesia", "Samarinda"),
    ("Indonesia", "Bontang"),

    ("Mozambique", "Beira"),
    ("Mozambique", "Nacala"),

    ("Russia", "Vostochny"),
    ("Russia", "Vanino"),

    ("USA", "Norfolk"),
    ("USA", "Baltimore"),
    ("USA", "Longview"),
]


DESTINATION_PORTS = [
    ("India", "Paradip"),
    ("India", "Visakhapatnam"),
    ("India", "Gangavaram"),
    ("India", "Gopalpur"),
    ("India", "Dhamra"),
    ("India", "Haldia"),
]


ALIASES = {
    "Visakhapatnam": [
        "Visakhapatnam",
        "Vishakhapatnam",
        "Vizagapatam",
    ],

    "Vostochny": [
        "Vostochny",
        "Vostochnyy",
    ],

    "Vanino": [
        "Vanino",
        "Bukhta Vanino",
    ],

    "Norfolk": [
        "Norfolk",
    ],
}


def normalize(text):
    if pd.isna(text):
        return ""

    return (
        str(text)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("/", " ")
    )


def find_port(wpi, country, requested_name):
    names_to_try = ALIASES.get(
        requested_name,
        [requested_name]
    )

    wpi_country = (
        wpi["Country Code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    wpi_main = (
        wpi["Main Port Name"]
        .fillna("")
        .map(normalize)
    )

    wpi_alt = (
        wpi["Alternate Port Name"]
        .fillna("")
        .map(normalize)
    )

    country_norm = country.strip().lower()

    for name in names_to_try:

        name_norm = normalize(name)

        mask = (
            (wpi_country == country_norm)
            &
            (wpi_main == name_norm)
        )

        matches = wpi.loc[mask]

        if not matches.empty:
            return matches.iloc[0]

        mask = (
            (wpi_country == country_norm)
            &
            (wpi_alt == name_norm)
        )

        matches = wpi.loc[mask]

        if not matches.empty:
            return matches.iloc[0]

    return None


def get_coordinates(row):
    lat = pd.to_numeric(row["Latitude"], errors="coerce")
    lon = pd.to_numeric(row["Longitude"], errors="coerce")

    if pd.isna(lat) or pd.isna(lon):
        return None

    return float(lon), float(lat)


print("Loading WPI...")

wpi = pd.read_csv(
    WPI_PATH,
    low_memory=False
)

print(f"WPI loaded: {len(wpi)} ports")


resolved_origins = []
resolved_destinations = []
missing = []


print("\nResolving origin ports...")

for country, name in ORIGIN_PORTS:

    row = find_port(
        wpi,
        country,
        name
    )

    if row is None:
        print(f"❌ Missing: {country} - {name}")

        missing.append({
            "type": "origin",
            "country": country,
            "port": name
        })

    else:

        coords = get_coordinates(row)

        if coords is None:
            print(f"❌ Missing coordinates: {country} - {name}")

            missing.append({
                "type": "origin",
                "country": country,
                "port": name
            })

        else:

            lon, lat = coords

            resolved_origins.append({
                "country": country,
                "name": name,
                "lon": lon,
                "lat": lat
            })

            print(
                f"✅ {country}: {name} "
                f"({lat:.4f}, {lon:.4f})"
            )


print("\nResolving destination ports...")

for country, name in DESTINATION_PORTS:

    row = find_port(
        wpi,
        country,
        name
    )

    if row is None:
        print(f"❌ Missing: {country} - {name}")

        missing.append({
            "type": "destination",
            "country": country,
            "port": name
        })

    else:

        coords = get_coordinates(row)

        if coords is None:
            print(f"❌ Missing coordinates: {country} - {name}")

            missing.append({
                "type": "destination",
                "country": country,
                "port": name
            })

        else:

            lon, lat = coords

            resolved_destinations.append({
                "country": country,
                "name": name,
                "lon": lon,
                "lat": lat
            })

            print(
                f"✅ {country}: {name} "
                f"({lat:.4f}, {lon:.4f})"
            )


routes = []

print("\nCalculating maritime routes...\n")

for origin in resolved_origins:

    for destination in resolved_destinations:

        try:

            result = sr.searoute(
                [origin["lon"], origin["lat"]],
                [destination["lon"], destination["lat"]],
                units="naut"
            )

            distance_nm = result["properties"]["length"]

            routes.append({
                "origin_country": origin["country"],
                "origin": origin["name"],
                "destination": destination["name"],
                "distance_nm": round(float(distance_nm), 2),
                "source": "NGA WPI + SeaRoute",
                "distance_basis": "maritime route"
            })

            print(
                f"✅ {origin['name']} → "
                f"{destination['name']}: "
                f"{distance_nm:.2f} NM"
            )

        except Exception as e:

            print(
                f"❌ Route failed: "
                f"{origin['name']} → "
                f"{destination['name']} | {e}"
            )


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

routes_df = pd.DataFrame(routes)

routes_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("ROUTE GENERATION COMPLETE")
print("========================================")

print(f"Routes generated: {len(routes_df)}")
print(f"Saved to: {OUTPUT_PATH}")


if missing:

    missing_df = pd.DataFrame(missing)

    missing_df.to_csv(
        MISSING_PATH,
        index=False
    )

    print(
        f"\nMissing ports saved to: "
        f"{MISSING_PATH}"
    )

else:

    print("\nAll requested ports were found.")


print("\nFirst 10 routes:")
print(routes_df.head(10).to_string(index=False))