import os
from deeplake import Client

TABLE_NAME = "lightwheel_bevorg_frames"

client = Client(
    token=os.environ.get("DEEPLAKE_API_KEY"),
    workspace_id=os.environ.get("DEEPLAKE_WORKSPACE"),
)

print("Trying fluent query...")
rows = client.table(TABLE_NAME).limit(3)()
print("num rows fetched:", len(rows))
for i, row in enumerate(rows):
    print(f"\nROW {i}")
    if isinstance(row, dict):
        print("keys:", sorted(row.keys()))
        for k, v in row.items():
            t = type(v).__name__
            shape = getattr(v, "shape", None)
            print(" ", k, "type=", t, "shape=", shape)
    else:
        print(row)
