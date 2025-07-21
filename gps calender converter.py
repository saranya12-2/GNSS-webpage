from datetime import datetime, timedelta

# GPS Epoch (Start of GPS time)
GPS_EPOCH = datetime(1980, 1, 6)

# Convert GPS Week + Day Number to Gregorian Date
def gps_week_to_gregorian(gps_week, day_of_week):
    return GPS_EPOCH + timedelta(weeks=gps_week, days=day_of_week)

# Convert Gregorian Date to GPS Week and Day Number (Sunday = 0)
def gregorian_to_gps_week(year, month, day):
    date = datetime(year, month, day)
    delta = date - GPS_EPOCH
    gps_week = delta.days // 7
    # Convert Python weekday (Monday=0) to (Sunday=0)
    day_number = (date.weekday() + 1) % 7
    return gps_week, day_number

# Main loop
while True:
    print("\nSelect input type:")
    print("1. GPS Calendar (GPS Week + Day of Week number)")
    print("2. Gregorian Calendar (Year + Month + Day)")

    choice = input("Enter 1 or 2: ").strip()

    try:
        if choice == "1":
            gps_week = int(input("Enter GPS Week Number: "))
            day_of_week = int(input("Enter Day of Week (Sunday=0 to Saturday=6): "))
            if not (0 <= day_of_week <= 6):
                raise ValueError("Day of week must be between 0 and 6.")
            date = gps_week_to_gregorian(gps_week, day_of_week)
            print(f"Gregorian Date: {date.strftime('%Y-%m-%d')}")
            print(f"Day {day_of_week}")

        elif choice == "2":
            year = int(input("Enter Year: "))
            month = int(input("Enter Month (1–12): "))
            day = int(input("Enter Day: "))
            date = datetime(year, month, day)
            gps_week, day_number = gregorian_to_gps_week(year, month, day)
            print(f"Gregorian Date: {date.strftime('%Y-%m-%d')}")
            print(f"GPS Week: {gps_week}")
            print(f"Day {day_number}")

        else:
            print("Invalid option. Please enter 1 or 2.")

    except Exception as e:
        print("Error:", e)

    again = input("\nDo you want to try another conversion? (yes/no): ").strip().lower()
    if again not in ['yes', 'y']:
        print("Thank you for using the GPS ↔ Gregorian Calendar converter. Goodbye!")
        break