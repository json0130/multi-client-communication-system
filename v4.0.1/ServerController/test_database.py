# test_database.py

import os
from database import Database

def main():
    db = Database()

    # 1) Create a new test user
    user_id = 100
    print(f"Creating user with user_id={user_id}…")
    user_id = db.create_user(
        user_id=user_id,
        name="Test User",
        interests=["coding", "robotics"],
        health_conditions=["asthma"]
    )
    print(" → Created user_id:", user_id)

    # 2) Fetch by user_id
    user = db.get_user_by_user_id(user_id)
    print("Fetched user:", user)

    # 3) Update name and arrays
    updated = db.update_user(
        user_id=user_id,
        name="Test User Jr.",
        interests=user["interests"] + ["AI"],
        health_conditions=user["health_conditions"] + ["allergy"]
    )
    print("After update:", updated)

    # 4) Add a single interest (no-op if duplicate)
    after_interest = db.add_interest(user_id, "robotics")
    print("After add_interest('robotics'):", after_interest["interests"])
    after_interest = db.add_interest(user_id, "machine learning")
    print("After add_interest('machine learning'):", after_interest["interests"])

    # 5) Add a single health_condition
    after_hc = db.add_health_condition(user_id, "allergy")
    print("After add_health_condition('allergy'):", after_hc["health_conditions"])
    after_hc = db.add_health_condition(user_id, "diabetes")
    print("After add_health_condition('diabetes'):", after_hc["health_conditions"])

    # # 6) Clean up: delete the test user
    # print("Cleaning up test user…")
    # db.client.table("users").delete().eq("user_id", user_id).execute()
    # print("Done.")

if __name__ == "__main__":
    main()
