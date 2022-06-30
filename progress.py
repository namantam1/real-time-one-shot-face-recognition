
PROGRESS_BAR_LENGHT = 40

def update_progress(percent, text=""):
    chars = int(percent * PROGRESS_BAR_LENGHT / 100)

    print(
        f"\r{text}: "
        f"[{'#'*chars}{' '*(PROGRESS_BAR_LENGHT-chars)}] {percent:.2f}% DONE",
        end="", flush=True)
    if int(percent) == 100:
        print()


if __name__ == "__main__":
    import time

    for i in range(100):
        time.sleep(.01)
        update_progress((i+1))
