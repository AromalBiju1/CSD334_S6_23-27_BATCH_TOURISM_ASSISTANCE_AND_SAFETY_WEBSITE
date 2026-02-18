# India Emergency Contacts Seed Data
# Source: Official government helplines and Wikipedia
# Use this to seed your EmergencyContact table
NATIONAL_HELPLINES = [
    {"name": "Emergency (All Services)", "number": "112", "type": "emergency", "category": "national"},
    {"name": "Police", "number": "100", "type": "police", "category": "national"},
    {"name": "Fire", "number": "101", "type": "fire", "category": "national"},
    {"name": "Ambulance", "number": "102", "type": "ambulance", "category": "national"},
    {"name": "Ambulance (EMRI)", "number": "108", "type": "ambulance", "category": "national"},
    {"name": "Women Helpline", "number": "181", "type": "women", "category": "national"},
    {"name": "Women Helpline (NCW)", "number": "1091", "type": "women", "category": "national"},
    {"name": "Child Helpline", "number": "1098", "type": "child", "category": "national"},
    {"name": "Tourist Helpline", "number": "1363", "type": "tourist", "category": "national"},
    {"name": "Senior Citizen Helpline", "number": "14567", "type": "senior", "category": "national"},
    {"name": "Road Accident Emergency", "number": "1073", "type": "accident", "category": "national"},
    {"name": "Railway Protection", "number": "182", "type": "railway", "category": "national"},
    {"name": "Anti-Corruption", "number": "1064", "type": "other", "category": "national"},
    {"name": "Cyber Crime", "number": "1930", "type": "cyber", "category": "national"},
    {"name": "Disaster Management", "number": "1078", "type": "disaster", "category": "national"},
]

# City-specific police control rooms (state/city, control room number)
CITY_CONTROL_ROOMS = [
    # Delhi
    {"city": "Delhi", "state": "Delhi", "name": "Delhi Police Control Room", "number": "011-23469200", "type": "police"},
    {"city": "Delhi", "state": "Delhi", "name": "Traffic Police", "number": "011-25844444", "type": "traffic"},
    {"city": "Delhi", "state": "Delhi", "name": "Women Helpdesk", "number": "011-23317054", "type": "women"},
    
    # Mumbai
    {"city": "Mumbai", "state": "Maharashtra", "name": "Mumbai Police Control Room", "number": "022-22621855", "type": "police"},
    {"city": "Mumbai", "state": "Maharashtra", "name": "Traffic Police", "number": "022-24937747", "type": "traffic"},
    {"city": "Mumbai", "state": "Maharashtra", "name": "Coastal Security", "number": "1093", "type": "police"},
    
    # Bangalore
    {"city": "Bangalore", "state": "Karnataka", "name": "Bangalore Police Control Room", "number": "080-22942222", "type": "police"},
    {"city": "Bangalore", "state": "Karnataka", "name": "Traffic Police", "number": "080-22868444", "type": "traffic"},
    
    # Chennai
    {"city": "Chennai", "state": "Tamil Nadu", "name": "Chennai Police Control Room", "number": "044-23452365", "type": "police"},
    {"city": "Chennai", "state": "Tamil Nadu", "name": "Traffic Police", "number": "044-23452359", "type": "traffic"},
    
    # Kolkata
    {"city": "Kolkata", "state": "West Bengal", "name": "Kolkata Police Control Room", "number": "033-22145060", "type": "police"},
    {"city": "Kolkata", "state": "West Bengal", "name": "Traffic Police", "number": "033-22145001", "type": "traffic"},
    
    # Hyderabad
    {"city": "Hyderabad", "state": "Telangana", "name": "Hyderabad Police Control Room", "number": "040-27852482", "type": "police"},
    {"city": "Hyderabad", "state": "Telangana", "name": "Traffic Police", "number": "040-23243333", "type": "traffic"},
    
    # Jaipur
    {"city": "Jaipur", "state": "Rajasthan", "name": "Jaipur Police Control Room", "number": "0141-2565555", "type": "police"},
    {"city": "Jaipur", "state": "Rajasthan", "name": "Traffic Police", "number": "0141-5105100", "type": "traffic"},
    
    # Ahmedabad
    {"city": "Ahmedabad", "state": "Gujarat", "name": "Ahmedabad Police Control Room", "number": "079-25393939", "type": "police"},
    
    # Pune
    {"city": "Pune", "state": "Maharashtra", "name": "Pune Police Control Room", "number": "020-26122880", "type": "police"},
    {"city": "Pune", "state": "Maharashtra", "name": "Traffic Police", "number": "020-26050550", "type": "traffic"},
    
    # Goa
    {"city": "Panaji", "state": "Goa", "name": "Goa Police Control Room", "number": "0832-2420444", "type": "police"},
    {"city": "Panaji", "state": "Goa", "name": "Tourist Police", "number": "1800-233-4444", "type": "tourist"},
    
    # Lucknow
    {"city": "Lucknow", "state": "Uttar Pradesh", "name": "Lucknow Police Control Room", "number": "0522-2621666", "type": "police"},
    
    # Varanasi
    {"city": "Varanasi", "state": "Uttar Pradesh", "name": "Varanasi Police Control Room", "number": "0542-2505505", "type": "police"},
    
    # Agra
    {"city": "Agra", "state": "Uttar Pradesh", "name": "Agra Police Control Room", "number": "0562-2266606", "type": "police"},
    {"city": "Agra", "state": "Uttar Pradesh", "name": "Tourist Police", "number": "0562-2421204", "type": "tourist"},
    
    # Amritsar
    {"city": "Amritsar", "state": "Punjab", "name": "Amritsar Police Control Room", "number": "0183-2563101", "type": "police"},
    
    # Kochi
    {"city": "Kochi", "state": "Kerala", "name": "Kochi Police Control Room", "number": "0484-2394500", "type": "police"},
    
    # Udaipur
    {"city": "Udaipur", "state": "Rajasthan", "name": "Udaipur Police Control Room", "number": "0294-2414600", "type": "police"},
]

# Hospitals (major cities)
HOSPITALS = [
    # Delhi
    {"city": "Delhi", "name": "AIIMS Emergency", "number": "011-26588500", "type": "hospital"},
    {"city": "Delhi", "name": "Safdarjung Hospital", "number": "011-26707447", "type": "hospital"},
    
    # Mumbai
    {"city": "Mumbai", "name": "JJ Hospital Emergency", "number": "022-23735555", "type": "hospital"},
    {"city": "Mumbai", "name": "KEM Hospital", "number": "022-24136051", "type": "hospital"},
    
    # Bangalore
    {"city": "Bangalore", "name": "Victoria Hospital", "number": "080-26701150", "type": "hospital"},
    
    # Chennai
    {"city": "Chennai", "name": "Government General Hospital", "number": "044-25305000", "type": "hospital"},
]


def seed_emergency_contacts(db_session):
    """
    Seed emergency contacts into database
    Usage: 
        from database.database import SessionLocal
        db = SessionLocal()
        seed_emergency_contacts(db)
    """
    from models import EmergencyContact, City
    
    # Add national helplines
    for contact in NATIONAL_HELPLINES:
        existing = db_session.query(EmergencyContact).filter_by(number=contact["number"]).first()
        if not existing:
            db_session.add(EmergencyContact(
                name=contact["name"],
                number=contact["number"],
                service_type=contact["type"],
                is_national=True
            ))
    
    # Add city-specific contacts
    for contact in CITY_CONTROL_ROOMS + HOSPITALS:
        city = db_session.query(City).filter_by(name=contact["city"]).first()
        if city:
            existing = db_session.query(EmergencyContact).filter_by(number=contact["number"]).first()
            if not existing:
                db_session.add(EmergencyContact(
                    city_id=city.id,
                    name=contact["name"],
                    number=contact["number"],
                    service_type=contact["type"],
                    is_national=False
                ))
    
    db_session.commit()
    print("Emergency contacts seeded successfully!")


if __name__ == "__main__":
    from database.database import sessionLocal
    db = sessionLocal()
    seed_emergency_contacts(db)
