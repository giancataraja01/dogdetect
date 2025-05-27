import firebase_admin
from firebase_admin import credentials, db

SERVICE_ACCOUNT_PATH = 'replace_this.json'
DATABASE_URL = 'https://project8-b295f-default-rtdb.asia-southeast1.firebasedatabase.app'

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {
            'databaseURL': DATABASE_URL
        })
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit(1)

try:
    ref = db.reference('detectstat')
    new_value = True
    ref.set(new_value)
    print(f"'detectstat' updated to: {new_value}")
except Exception as e:
    print(f"Database update error: {e}")
