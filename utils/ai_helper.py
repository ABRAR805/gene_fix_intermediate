import requests
import json

def get_ai_simulation_message(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",  # Replace with your actual key
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourdomain.com",         # Optional
        "X-Title": "YourSiteName"                         # Optional
    }
    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Failed to get AI response: {str(e)}"