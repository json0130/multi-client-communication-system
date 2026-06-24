import os
template_path = os.path.join(os.path.dirname(__file__), "templates", "monitor.html")
print("Template path:", template_path)
with open(template_path, "r", encoding="utf-8") as f:
    print(f.readline())
