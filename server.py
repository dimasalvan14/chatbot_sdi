from flask import Flask, request, jsonify
import os
import requests

# Import chatbot logic
from chatbot import detect_intent, generate_response

app = Flask(__name__)

# Set default verify and access tokens if not from environment
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "chatbotsmpislamarrohman123")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "EAAUHgvUZAMoABO7qgvFYZAYGZB0pkJOqZC91Y4BLC0buOnYo0pfoMldODEEp30w53L51Q6hkNhZCQK5vxmnps4sAwJGAHMplbRwN0V0mu9zwW9AZCjqjzbkMU6EenXuAHuhvn4HmaM4oGquj9ghQZAvttnoPLnZCpgtWLjTTyWFG37i03JYb1Ama0OFXXi36xwZDZD")  # Replace for safety

@app.route("/", methods=["GET"])
def index():
    return "✅ Chatbot SMP Islam Arrohman is running!"

# Webhook verification from Meta
@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("[INFO] Webhook verified successfully.")
        return challenge, 200
    else:
        print("[ERROR] Webhook verification failed.")
        return "Verification token mismatch", 403

# Webhook message handler from WhatsApp
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    print("[WEBHOOK RECEIVED]", data)  # 🔥 ADD THIS
    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]["value"]
        message = changes["messages"][0]
        phone_number_id = changes["metadata"]["phone_number_id"]
        from_number = message["from"]
        user_text = message["text"]["body"]

        # Bot logic
        intent = detect_intent(user_text)
        response_text = generate_response(user_text, intent)

        # Send response back to user via WhatsApp
        url = f"https://graph.facebook.com/v23.0/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": from_number,
            "text": {"body": response_text}
        }

        r = requests.post(url, headers=headers, json=payload)
        print("[INFO] Message sent:", r.status_code, r.text)

    except Exception as e:
        print("[ERROR] Failed to process webhook:", str(e))
        return "Internal Server Error", 500

    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)