from datetime import datetime, timedelta
from io import BytesIO
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# 1) Collect data
now = datetime.utcnow()
ten_hours_ago = now - timedelta(hours=10)
min_date_utc = int(ten_hours_ago.timestamp())
max_date_utc = int(now.timestamp())

TOKEN = "dd246f5afde3eceba4aa392777df19de4f1b2e71339a2450c24e70e83a54ad3dbe1cff1a226b4f1b4a5954cab691d4e5eb4b74ab699520770bfba723041e9dca"

line_IDs = [34, 38, 64, 80, 54, 64, 71, 36, 60, 95, 59]
poin_IDs = [1162, 1233, 1278, 1280, 1654, 1706, 1712, 1713, 1714, 1715,
    1728, 1729, 1733, 1734, 1762, 1764, 1769, 1776, 1777, 1803,
    1804, 1824, 1870, 1871, 1880, 1881, 1883, 1884, 1906, 1909,
    1910, 1980, 1981, 1983, 2351, 2928, 2963, 2964, 2967, 3120,
    3121, 3155, 3156, 3160, 3372, 3506, 3510, 3514, 3517, 3518,
    3525, 3558, 3559, 3562, 3570, 3572, 3912, 3921, 4303, 4304,
    4305, 4306, 4307, 4362, 4363, 4364, 4365, 4366, 4367, 5407,
    5611, 5612, 6112, 6432, 6433]

def auth_request(url):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.get(url, headers=headers)
    print("Status Code:", response.status_code)
    response.raise_for_status()
    return response

response = auth_request(
    f"https://api.mobilitytwin.brussels/parquetized?start_timestamp={min_date_utc}&end_timestamp={max_date_utc}&component=stib_vehicle_distance_parquetize"
).json()

arrow_table = None
for url in response["results"]:
    data = BytesIO(requests.get(url).content)
    if arrow_table is None:
        arrow_table = pq.read_table(data)
    else:
        arrow_table = pa.concat_tables([arrow_table, pq.read_table(data)])

df = arrow_table.to_pandas()
df.to_parquet("combined_data.parquet")

line_ids_str = ",".join(f"'{i}'" for i in line_IDs)
poin_ids_str = ",".join(f"'{i}'" for i in poin_IDs)

con = duckdb.connect()
results_df = con.execute(f"""WITH entries AS (   
        SELECT lineId,pointId,directionId,distanceFromPoint, (date AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Brussels')::timestamp as local_date
        FROM read_parquet('combined_data.parquet')
        WHERE lineId IN ({line_ids_str}) 
            AND pointId IN ({poin_ids_str})
    ), filtered_entries AS (
        SELECT 
            *,
            count(*) OVER (PARTITION BY directionId, pointId, local_date, lineId) as row_count
        FROM entries
    ), deltaTable as (
    SELECT 
        local_date,
        lineId as lineId,
        directionId as directionId,
        pointId as pointId,
        distanceFromPoint as distanceFromPoint,
        distanceFromPoint - lag(distanceFromPoint) OVER (PARTITION BY pointId, directionId,lineId ORDER BY local_date) AS distance_delta,
        (local_date - lag(local_date) OVER (PARTITION BY pointId, directionId, lineId ORDER BY local_date)) as time_delta
    FROM filtered_entries
    WHERE row_count = 1
    ), speedTable as (
    SELECT 
       local_date,
        lineId,
        directionId,
        pointId,
        distanceFromPoint,
        (distance_delta / epoch(time_delta)) as speed
        FROM deltaTable
        WHERe epoch(time_delta) < 30 AND distance_delta < 600
    )
    SELECT  lineId, directionId, pointId, avg(speed) * 3.6, count(*) as count, time_bucket(interval '15 minutes', local_date) as agg
    FROM speedTable
    WHERE speed > 0
    GROUP BY lineId, directionId, pointId, agg
"""
).df()

results_df['agg'] = pd.to_datetime(results_df['agg'])

# Select the last 10 time intervals (not just the latest)
last_n = 10
last_timestamps = results_df['agg'].drop_duplicates().sort_values().iloc[-last_n:]
results_df = results_df[results_df['agg'].isin(last_timestamps)]

results_df = results_df.sort_values(['agg', 'pointId', 'lineId'])
results_df = results_df.rename(columns={'agg': 'local_time'})

# Produce both detailed and matrix outputs
last_csv = "Last_timestamp_STIB_buses_speeds.csv"
results_df.to_csv(last_csv, index=False)


# --- aggregate
segments = pd.read_csv("Etterbeek_STIB_segments.csv", sep=";")

segments["bus lines"] = (
    segments["bus lines"]
    .astype(str)
    .str.split(",")
)
segments = segments.explode("bus lines")
segments["bus lines"] = segments["bus lines"].astype(int)

segments = segments.rename(columns={
    "ID-start": "pointId",
    "bus lines": "lineId"
})

speeds = results_df.copy()
segments = segments.dropna(subset=["pointId", "lineId"])
speeds = speeds.dropna(subset=["pointId", "lineId"])

segments["pointId"] = segments["pointId"].astype(int)
segments["lineId"] = segments["lineId"].astype(int)
speeds["pointId"] = speeds["pointId"].astype(int)
speeds["lineId"] = speeds["lineId"].astype(int)


df_merge = pd.merge(
    speeds,
    segments[["pointId", "lineId", "ID_graph_edge"]],
    on=["pointId", "lineId"],
    how="right"
)

df_merge = df_merge.rename(columns={"(avg(speed) * 3.6)": "speed"})

agg = (
    df_merge
    .groupby(["local_time", "ID_graph_edge"])["speed"]
    .mean()
    .reset_index()
)

# In this context, there's only one local_time (= latest timestamp), but we keep full logic
times = pd.to_datetime(last_timestamps)
edges = segments["ID_graph_edge"].unique()
full_index = pd.MultiIndex.from_product(
    [times, edges],
    names=["local_time", "ID_graph_edge"]
)

agg_full = (
    agg
    .set_index(["local_time", "ID_graph_edge"])
    .reindex(full_index)
    .reset_index()
)

speed_matrix = (
    agg_full
    .pivot(index="local_time", columns="ID_graph_edge", values="speed")
)

# matrix_file = "STIB_speeds.csv"
# speed_matrix.to_csv(matrix_file)


speed_long = speed_matrix.reset_index().melt(
    id_vars="local_time",
    var_name="SegmentID",
    value_name="Speed"
)
speed_long = speed_long.rename(columns={"local_time": "Time"})
speed_long.to_csv("STIB_speeds.csv", index=False)



