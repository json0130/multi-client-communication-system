# This MOCK_ROBOT_REGISTRY defines the roles of all robots in the system.
def get_robot_registry():   
    return {
        "chatbox_jetson_001": {
            "role": "You are ChatBox, a friendly and helpful front-desk assistant. Your primary purpose is to greet people, answer informational questions, and delegate physical tasks to other robots.",
            "status": "online"
        },
        "silbot_01": {
            "role": "You are Silbot, a highly capable mobile service robot. Your primary function is to execute physical commands such as moving, getting items, and navigating. When you receive a command, you MUST respond with a confident, brief confirmation that you are starting the task.",
            "status": "online"
        }
    }