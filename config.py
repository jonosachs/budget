# Manual-only category: park a transaction here to keep it out of every total.
# Deliberately NOT in TAXONOMY - that dict is the LLM's option list, and nothing
# should be auto-excluded. It is its own parent so parent-level views treat it
# like any other top-level bucket.
EXCLUDED = "Excluded"

# Where a transaction lands when the LLM returns no category. Unlike EXCLUDED it
# IS in TAXONOMY, so the model can also choose it outright, and it counts towards
# totals like any other category — an unlabelled expense is still an expense.
UNCATEGORISED = "Uncategorised"

# Below this a classification is flagged for review, not discarded. Set high on
# purpose: the model returns exactly 1.0 for ~82% of records, so a low bar caught
# almost nothing, and flagging costs nothing now that flagged rows still count.
CONFIDENCE_THRESHOLD = 0.85

TAXONOMY = {
    "Housing": [
        "Mortgage",
        "Rent",
        "Body Corporate",
        "Property Rates",
        "Cleaning",
        "Maintenance",
        "Renovations",
        "Gardening & Lawn",
        "Pest Control",
        "Removals & Storage",
        "Home Security",
        "Building Insurance",
        "Contents Insurance",
        "Landlord Insurance",
    ],
    "Utilities": [
        "Electricity",
        "Gas",
        "Water",
        "Internet",
        "Mobile Phone",
        "Home Phone",
        "Waste & Recycling",
    ],
    "Groceries": [
        "Supermarket",
        "Fruit & Veg",
        "Bakery",
        "Butcher & Seafood",
        "Deli & Specialty",
        "Convenience Store",
        "Liquor Store",
        "Groceries",  # catch-all: retire once records are re-categorised
    ],
    "Dining": [
        "Cafe & Coffee",
        "Restaurants & Takeaway",
        "Food Delivery",
        # Consumed on the spot: vending machines, servo drinks, a milk bar run
        # for a Coke. Groceries covers the same shops when you are stocking up —
        # the split is provisioning vs eating now, not where you bought it.
        "Snacks & Drinks",
        "Alcohol & Bars",
    ],
    "Car": [
        "Fuel",
        "EV Charging",
        "Parking",
        "Tolls",
        "Registration",
        "Servicing",
        "Repairs",
        "Car Wash",
        "Car Accessories",
        "Roadside Assistance",
        "Car Purchase",
        "Car Repayments",
        "Car Insurance",
    ],
    "Childcare": [
        "Daycare",
        "Toys",
        "Baby Food",
        "Baby Clothing & Gear",
        "Nappies & Supplies",
        "Babysitting",
        "Kids Activities",
        "Other Childcare",
    ],
    "Education": [
        "School Fees",
        "Tuition & Course Fees",
        "Textbooks & Supplies",
        "Uniforms",
        "Excursions & Camps",
        "Training & Certification",
        "Student Loan Repayments",
    ],
    "Transport": [
        "Taxis & Rideshare",
        "Bike & Scooter Hire",
        "Car Share",
        "Public Transport",
    ],
    "Health": [
        "Doctor",
        "Specialist",
        "Hospital",
        "Physiotherapy",
        "Chiropractic & Osteo",
        "Psychology",
        "Dentist",
        "Medicine",
        "Pathology",
        "Supplements",
        "Optometry",
        "Imaging",
    ],
    "Fitness": [
        "Gym",
        "Martial Arts",
        "Yoga",
        "Personal Training",
        "Sports Club Membership",
    ],
    # Insurance of a person. Anything insuring an asset lives under that asset, so
    # a parent row answers "what does this cost me?" - see the note on Debt.
    "Insurance": [
        "Health Insurance",
        "Life Insurance",
        "Income Protection",
        "Ambulance Cover",
    ],
    "Personal Care": [
        "Haircuts",
        "Grooming & Cosmetics",
        "Massage & Spa",
        "Waxing & Nails",
    ],
    "Pets": [
        "Pet Food",
        "Pet Supplies",
        "Vet",
        "Pet Grooming",
        "Boarding & Kennels",
        "Pet Training",
        "Pet Insurance",
    ],
    "Shopping": [
        "Clothing",
        "Footwear",
        "Sporting Goods",
        "Jewelry",
        "Electronics",
        "Homeware",
        "Furniture",
        "Hardware & DIY",
        "Garden & Outdoor",
        "Department & General",
        "Online Marketplace",
        "Office Supplies",
    ],
    "Subscriptions": [
        "Streaming",
        "Music",
        "Audiobooks",
        "Software & Apps",
        "Cloud & Hosting",
        "AI Tools",
    ],
    "Entertainment": [
        "Events",
        "Cinema",
        "Attractions",
        "Hobbies",
        "Books",
        "Games",
        "News & Magazines",
        "Media",  # catch-all: retire once records are re-categorised
    ],
    "Gifts": ["Presents", "Florist"],
    # One-off occasions. Kept top-level rather than under Gifts or Entertainment so
    # the lumpy spend can be excluded from monthly trends instead of skewing them.
    "Special Events": [
        "Wedding",
        "Birthday & Party",
        "Christmas & Holidays",
        "Funeral & Memorial",
        "Catering & Venue Hire",
    ],
    "Travel": [
        "Accommodation",
        "Flights & Fares",
        "Car Hire",
        "Tours & Activities",
        "Airport Parking & Transfers",
        "Visas & Passports",
        "Foreign Currency",
        "Travel Insurance",
    ],
    "Government": ["Traffic Fines", "Licences & Permits", "Government Services"],
    "Fees": [
        "Bank Fees",
        "Account Keeping Fees",
        "ATM Fees",
        "Foreign Transaction Fees",
        "Interest Charges",
        "Late Payment Fees",
    ],
    "Services": [
        "Professional Services",
        "Legal Services",
        "Accounting & Tax",
        "Postal & Delivery",
        "Equipment Hire",
    ],
    "Donations": ["Donations", "Charity", "Religious Giving", "Sponsorship"],
    "Work & Business": [
        "Work Expenses",
        "Business Supplies",
        "Professional Memberships",
        "Union Fees",
        "Coworking & Office",
    ],
    # Debt servicing that isn't already under its asset (Housing/Mortgage,
    # Car/Car Repayments, Education/Student Loan Repayments).
    "Debt": [
        "Credit Card Payment",
        "Personal Loan",
        "Buy Now Pay Later",
        "Loan Interest",
        "Debt Collection",
    ],
    "Taxes": ["Income Tax", "Land Tax", "Capital Gains Tax", "Other Tax"],
    # Money moving into an asset rather than being consumed. Like Transfers, these
    # are not expenses — exclude them before totalling spend.
    "Investments": [
        "Shares & ETFs",
        "Managed Funds",
        "Superannuation",
        "Cryptocurrency",
        "Property Investment",
        "Investment Fees",
    ],
    # Credits, not debits. 'spend' is -amount, so these land as negative spend and
    # will net against expenses unless the dashboard filters them out.
    "Income": [
        "Salary & Wages",
        "Bonus & Commission",
        "Self-Employment Income",
        "Interest Received",
        "Dividends",
        "Rental Income",
        "Government Benefits",
        "Tax Refund",
        "Refunds & Reimbursements",
        "Other Income",
    ],
    "Cash": ["Cash Withdrawal"],
    "Transfers": ["Internal Transfers", "External Transfers", "Savings Transfer"],
    # Explicit escape hatch: better an honest 'unknown' than a forced wrong label.
    "Uncategorised": ["Uncategorised"],
}


def get_taxonomy_children() -> dict[str, str]:
    return {
        child: parent for parent, children in TAXONOMY.items() for child in children
    }


BANK_ADAPTORS = {
    "NAB": {
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
