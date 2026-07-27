TAXONOMY = {
    "Housing": [
        "Mortgage",
        "Rent",
        "Body Corporate",
        "Property Rates",
        "Cleaning",
        "Maintenance",
        "Renovations",
    ],
    "Utilities": ["Electricity", "Gas", "Water", "Internet", "Mobile Phone"],
    "Groceries": ["Groceries"],
    "Dining": ["Cafe & Coffee", "Restaurants & Takeaway", "Alcohol & Bars"],
    "Car": [
        "Fuel",
        "Parking",
        "Tolls",
        "Registration",
        "Servicing",
        "Car Wash",
        "Roadside Assistance",
        "Car Repayments",
    ],
    "Childcare": ["Daycare", "Other Childcare"],
    "Transport": ["Taxis & Rideshare", "Public Transport"],
    "Health": ["Doctor", "Dentist", "Medicine", "Supplements", "Optometry", "Imaging"],
    "Fitness": ["Gym", "Martial Arts", "Yoga"],
    "Insurance": [
        "Health Insurance",
        "Car Insurance",
        "Contents Insurance",
        "Ambulance Cover",
    ],
    "Personal Care": ["Haircuts", "Grooming & Cosmetics"],
    "Pets": ["Pet Food", "Vet"],
    "Shopping": [
        "Clothing",
        "Electronics",
        "Homeware",
        "Department & General",
        "Online Marketplace",
        "Office Supplies",
    ],
    "Subscriptions": ["Streaming", "Music", "Audiobooks", "Software & Apps"],
    "Entertainment": ["Events", "Attractions", "Hobbies", "Media"],
    "Gifts": ["Presents", "Florist"],
    "Travel": ["Accommodation", "Flights & Fares"],
    "Government": ["Traffic Fines", "Government Services"],
    "Fees": ["Bank Fees", "Foreign Transaction Fees"],
    "Services": ["Professional Services", "Postal & Delivery"],
    "Donations": ["Donations"],
    "Cash": ["Cash Withdrawal"],
    "Transfers": ["Internal Transfers", "External Transfers"],
}


def get_taxonomy_children() -> dict[str, str]:
    return {
        child: parent for parent, children in TAXONOMY.items() for child in children
    }


BANK_ADAPTORS = {
    "nab": {
        "columns": {
            "Date": "date",
            "Amount": "amount",
            "Account Number": "account",
            "Transaction Type": "type",
            "Transaction Details": "description",
            "Category": "category",
            "Merchant Name": "merchant",
        },
        "date_format": "%d %b %y",
    }
}
