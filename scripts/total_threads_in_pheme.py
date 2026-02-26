import os

# =====================================================
# FULL PHEME DATASET ROOT (UPDATED PATH)
# =====================================================
ROOT = r"C:\Users\KIIT0001\rumor_detection_sls\phemernrdataset\pheme-rnr-dataset"

print("=" * 85)
print("FULL PHEME DATASET — THREAD STATISTICS (5802 EXPECTED)")
print("=" * 85)

grand_total = 0
grand_rumours = 0
grand_nonrumours = 0


def count_threads(folder_path):
    """Each tweet-id folder = one thread"""
    if not os.path.exists(folder_path):
        return 0

    return sum(
        os.path.isdir(os.path.join(folder_path, name))
        for name in os.listdir(folder_path)
    )


# iterate through 5 events
for event in sorted(os.listdir(ROOT)):

    event_path = os.path.join(ROOT, event)

    if not os.path.isdir(event_path):
        continue

    rumours_path = os.path.join(event_path, "rumours")
    nonrumours_path = os.path.join(event_path, "non-rumours")

    rumour_count = count_threads(rumours_path)
    nonrumour_count = count_threads(nonrumours_path)

    total = rumour_count + nonrumour_count

    grand_total += total
    grand_rumours += rumour_count
    grand_nonrumours += nonrumour_count

    print(
        f"{event:<22} | Rumours: {rumour_count:4d} | "
        f"Non-rumours: {nonrumour_count:4d} | Total: {total:4d}"
    )

print("-" * 85)
print(
    f"{'TOTAL':<22} | Rumours: {grand_rumours:4d} | "
    f"Non-rumours: {grand_nonrumours:4d} | Total: {grand_total:4d}"
)
print("=" * 85)