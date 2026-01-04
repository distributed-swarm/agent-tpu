# ops/trigger_oracle.py
import requests
import os

# Oracle Configuration
ORACLE_HOST = os.environ.get("ORACLE_HOST", "https://eg-dev.fa.us2.oraclecloud.com")
ORACLE_AUTH = (os.environ.get("ORA_USER"), os.environ.get("ORA_PASS"))

def run(payload):
    """
    Expects payload: {"event": "InventoryUpdate", "item": "A544", "qty": -1}
    """
    try:
        # Example: Adjusting Inventory via Oracle SCM Cloud REST API
        endpoint = f"{ORACLE_HOST}/fscmRestApi/resources/11.13.18.05/inventoryTransactions"
        
        oracle_payload = {
            "TransactionType": "Material Issue",
            "ItemNumber": payload.get("item"),
            "TransactionQuantity": payload.get("qty"),
            "TransactionDate": "2026-01-04T12:00:00Z"
        }
        
        headers = {"Content-Type": "application/vnd.oracle.adf.resourceitem+json"}

        # Fire the signal
        response = requests.post(endpoint, json=oracle_payload, headers=headers, auth=ORACLE_AUTH)

        if response.status_code == 201:
            return {"status": "success", "oracle_tx_id": response.json()['TransactionId']}
        else:
            return {"error": f"Oracle Rejected: {response.text}"}

    except Exception as e:
        return {"error": str(e)}
