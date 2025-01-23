import pandas as pd
import re


def parse_cc(cc):
    if isinstance(cc, str):
        cc = cc.lower()
        if "nan" in cc or cc == "-":
            return None
        parts = re.findall(r"(\d+\.?\d*)\s*(?:cc|kwh)?", cc)
        if parts:
            values = [float(part) for part in parts]
            if values:
                return sum(values) / len(values)
        return None
    return cc


def parse_horsepower(hp):
    if isinstance(hp, str):
        hp = hp.lower()
        parts = re.findall(r"(\d+)\s*(?:hp|cc)?", hp)
        if parts:
            values = [int(part) for part in parts]
            if values:
                return sum(values) / len(values)
        return None
    return hp


def parse_seats(seats):
    if pd.isna(seats):
        return None
    if isinstance(seats, str):
        seats = seats.replace("+", "")
        match = re.search(r"(\d+)", seats)
        if match:
            return int(match.group(1))
        return None
    return seats


def parse_speed(speed):
    if isinstance(speed, str):
        speed = speed.lower()
        match = re.search(r"(\d+)", speed)
        if match:
            return int(match.group(1))
        return None
    return speed


def parse_performance(perf):
    if pd.isna(perf):
        return None
    if isinstance(perf, str):
        perf = perf.lower()
        parts = re.findall(r"(\d+\.?\d*)", perf)
        if parts:
            values = [float(part) for part in parts]
            if values:
                return sum(values) / len(values)
        return None
    return perf


def parse_price(price):
    if pd.isna(price):
        return None
    if isinstance(price, str):
        price = price.lower()
        if "n/a" in price:
            return None
        parts = re.findall(r"(\d+\.?\d*)", price)
        if parts:
            values = [float(part) for part in parts]
            if values:
                return sum(values) / len(values)
        return None
    return price


def parse_torque(torque):
    if pd.isna(torque):
        return None
    if isinstance(torque, str):
        torque = torque.lower()
        parts = re.findall(r"(\d+)\s*nm", torque)
        if parts:
            values = [int(part) for part in parts]
            if values:
                return sum(values) / len(values)
        return None
    return torque


def extract_turbo(engine):
    if pd.isna(engine):
        return None
    engine = engine.lower()
    if "turbo" in engine:
        return "Yes"
    else:
        return "No"


# Load the dataset

df = pd.read_csv("The Ultimate Cars Dataset 2024.csv", encoding="ISO-8859-1")
df["CC/Battery Capacity"] = df["CC/Battery Capacity"].apply(parse_cc)
df["HorsePower"] = df["HorsePower"].apply(parse_horsepower)
df["Total Speed"] = df["Total Speed"].apply(parse_speed)
df["Performance(0 - 100 )KM/H"] = df["Performance(0 - 100 )KM/H"].apply(
    parse_performance
)
df["Cars Prices"] = df["Cars Prices"].apply(parse_price)
df["Torque"] = df["Torque"].apply(parse_torque)
df["Seats"] = df["Seats"].apply(parse_seats)
df["Turbo"] = df["Engines"].apply(extract_turbo)

df.to_csv("parsed_cars_data.csv", index=False, na_rep="")
print("Parsed data saved to parsed_cars_data.csv")

# Assess the parser by comparing null values
df_original = pd.read_csv("The Ultimate Cars Dataset 2024.csv", encoding="ISO-8859-1")
null_counts_original = df_original.isnull().sum()
null_counts_parsed = df.isnull().sum()

print("\nNull value counts in original dataset:")
print(null_counts_original)
print("\nNull value counts in parsed dataset:")
print(null_counts_parsed)
