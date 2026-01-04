# ops/trigger_sap.py
import requests
import os

# SAP Configuration (In production, load these from secure Env Vars)
SAP_HOST = os.environ.get("SAP_HOST", "https://my-sap-instance.com")
SAP_AUTH = (os.environ.get("SAP_USER"), os.environ.get("SAP_PASS"))

def run(payload):
    """
    Expects payload: {"event_type": "QualityIssue", "material": "PART-123", "text": "Crack detected"}
    """
    try:
        # Example: Triggering a Quality Notification in S/4HANA via OData
        endpoint = f"{SAP_HOST}/sap/opu/odata/sap/API_QUALNOTIFICATION_SRV/A_QualityNotification"
        
        sap_payload = {
            "NotificationType": "Q1",  # Customer Complaint / Defect
            "Material": payload.get("material"),
            "NotificationText": payload.get("text"),
            "Priority": "1"  # High Priority (It's an excitatory event!)
        }

        # Fire the signal
        response = requests.post(endpoint, json=sap_payload, auth=SAP_AUTH)
        
        if response.status_code == 201:
            return {"status": "success", "sap_id": response.json()['d']['Notification']}
        else:
            return {"error": f"SAP Rejected: {response.text}"}

    except Exception as e:
        return {"error": str(e)}
