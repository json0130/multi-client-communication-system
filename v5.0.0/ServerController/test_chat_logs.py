# test_chat_logs.py
from database import Database

def main():
    db = Database()
    
    print("Creating test user...")
    user_id = db.create_user(
        name="Chat Logger Test",
        interests=["testing"],
        health_conditions=[]
    )
    print(f"Created user with ID: {user_id}")
    
    print("\nTesting chat log insertion...")
    messages = [
        ("Hello, how are you?", "I'm doing well, thanks for asking!"),
        ("What's the weather today?", "It's sunny and 75°F"),
        ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!")
    ]
    
    for msg, resp in messages:
        log_id = db.insert_chat_log(user_id, msg, resp)
        print(f"Inserted log {log_id}: {msg[:20]}... → {resp[:20]}...")
    
    print("\nTesting chat log retrieval...")
    user_data = db.get_user_by_user_id(user_id)
    print(f"User: {user_data['name']}")
    print(f"Interests: {', '.join(user_data['interests'])}")
    
    # Note: To retrieve actual chat logs, you'd need to query the chat_logs table directly
    # This is just showing the user exists

if __name__ == "__main__":
    main()