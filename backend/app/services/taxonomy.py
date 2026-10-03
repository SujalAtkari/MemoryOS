from __future__ import annotations

from typing import Mapping

CATEGORY_SUBCATEGORIES: dict[str, tuple[str, ...]] = {
    'Government & Identity': ('Aadhaar', 'PAN', 'Passport', 'Driving Licence', 'Voter ID', 'Government Document', 'Identity Document', 'Other Identity'),
    'Education': ('Certificate', 'Marksheet', 'Academic Transcript', 'Notes', 'Assignment', 'Examination', 'Result', 'College Document', 'Other Education'),
    'Medical & Health': ('Medical Report', 'Prescription', 'Lab Report', 'Medical Bill', 'Health Record', 'Appointment', 'Other Medical'),
    'Finance': ('Bank Statement', 'Passbook', 'Credit Card', 'Investment', 'Mutual Fund', 'Demat', 'Insurance', 'Loan', 'Tax', 'Salary / Income', 'Financial Certificate', 'Other Finance'),
    'Bills & Receipts': ('Shopping', 'Electricity', 'Internet', 'Mobile', 'Restaurant', 'Grocery', 'Online Order', 'Invoice', 'Payment Receipt', 'Other Bill'),
    'Work & Professional': ('Work Report', 'Project Document', 'Meeting Notes', 'Presentation', 'Resume / CV', 'Offer Letter', 'Internship Document', 'Professional Certificate', 'Client Document', 'Other Work'),
    'Travel': ('Flight Ticket', 'Train Ticket', 'Bus Ticket', 'Boarding Pass', 'Hotel', 'Travel Booking', 'Travel Document', 'Itinerary', 'Map / Directions', 'Destination Photo', 'Travel Activity', 'Travel Receipt', 'Other Travel'),
    'Events & Celebrations': ('Birthday', 'Wedding', 'Festival', 'College Event', 'Party', 'Ceremony', 'Invitation', 'Other Event'),
    'People & Family': ('Family', 'Friends', 'Portrait', 'Group Photo', 'Personal Memory', 'Other People'),
    'Nature & Places': ('Landscape', 'Beach', 'Mountain', 'City', 'Building', 'Monument', 'Nature', 'Sunset', 'Other Place'),
    'Animals & Pets': ('Dog', 'Cat', 'Bird', 'Pet', 'Wildlife', 'Other Animal'),
    'Food & Drinks': ('Food', 'Restaurant', 'Dessert', 'Beverage', 'Recipe', 'Menu', 'Other Food'),
    'Screenshots': ('App Screenshot', 'Website Screenshot', 'Chat Screenshot', 'Social Media Screenshot', 'Error Screenshot', 'Code Screenshot', 'UI / Design Screenshot', 'Other Screenshot'),
    'Notes & Documents': ('Notes', 'Text Document', 'Form', 'Letter', 'General Document', 'Reference', 'Other Document'),
    'Others': ('Miscellaneous', 'Unknown', 'Unclassified'),
}

SUBCATEGORY_HINTS: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    'Government & Identity': (
        ('Aadhaar', ('aadhaar', 'aadhar', 'uidai')),
        ('PAN', ('pan card', 'permanent account number')),
        ('Passport', ('passport',)),
        ('Driving Licence', ('driving licence', 'driving license', 'driver license')),
        ('Voter ID', ('voter id', 'voter identity')),
    ),
    'Education': (
        ('Certificate', ('certificate',)),
        ('Marksheet', ('marksheet', 'mark sheet', 'mark-sheet')),
        ('Academic Transcript', ('academic transcript', 'transcript')),
        ('Assignment', ('assignment', 'coursework')),
        ('Examination', ('exam paper', 'examination', 'question paper')),
        ('Result', ('exam result', 'result sheet', 'results')),
        ('College Document', ('university', 'college', 'semester', 'bachelor', 'master of')),
        ('Notes', ('lecture notes', 'class notes', 'study notes')),
    ),
    'Medical & Health': (
        ('Prescription', ('prescription', 'prescribed')),
        ('Lab Report', ('lab report', 'laboratory report', 'blood test')),
        ('Medical Report', ('medical report', 'diagnosis', 'clinical report')),
        ('Medical Bill', ('hospital bill', 'medical bill', 'pharmacy bill')),
        ('Appointment', ('appointment', 'doctor visit')),
    ),
    'Finance': (
        ('Bank Statement', ('bank statement', 'account statement')),
        ('Passbook', ('passbook',)),
        ('Credit Card', ('credit card',)),
        ('Investment', ('investment', 'portfolio')),
        ('Insurance', ('insurance policy', 'insurance')),
        ('Loan', ('loan account', 'loan agreement')),
        ('Tax', ('income tax', 'tax return', 'gst')),
        ('Salary / Income', ('salary slip', 'payslip', 'pay slip')),
    ),
    'Bills & Receipts': (
        ('Electricity', ('electricity bill', 'power bill')),
        ('Internet', ('internet bill', 'broadband bill')),
        ('Mobile', ('mobile bill', 'phone bill', 'telecom bill')),
        ('Restaurant', ('restaurant bill', 'restaurant receipt')),
        ('Grocery', ('grocery bill', 'grocery receipt')),
        ('Online Order', ('order confirmation', 'online order', 'order total')),
        ('Invoice', ('invoice', 'tax invoice')),
        ('Payment Receipt', ('payment receipt', 'receipt', 'paid')),
        ('Shopping', ('shopping receipt', 'purchase receipt')),
    ),
    'Work & Professional': (
        ('Work Report', ('work report', 'performance report', 'monthly report')),
        ('Project Document', ('project report', 'project document')),
        ('Meeting Notes', ('meeting notes', 'meeting agenda', 'minutes of meeting')),
        ('Presentation', ('presentation', 'slide deck', 'powerpoint')),
        ('Resume / CV', ('resume', 'curriculum vitae', 'cv')),
        ('Offer Letter', ('offer letter', 'employment offer')),
        ('Professional Certificate', ('professional certificate', 'training certificate')),
    ),
    'Travel': (
        ('Flight Ticket', ('flight ticket', 'airline ticket')),
        ('Train Ticket', ('train ticket', 'railway ticket')),
        ('Bus Ticket', ('bus ticket',)),
        ('Boarding Pass', ('boarding pass', 'passenger', 'departure', 'arrival', 'seat')),
        ('Hotel', ('hotel booking', 'hotel reservation')),
        ('Travel Booking', ('travel booking', 'reservation')),
        ('Itinerary', ('itinerary',)),
        ('Map / Directions', ('map', 'directions', 'route')),
        ('Travel Receipt', ('travel receipt', 'airline receipt')),
    ),
    'Events & Celebrations': (
        ('Birthday', ('birthday',)),
        ('Wedding', ('wedding', 'bride', 'groom')),
        ('Festival', ('festival', 'diwali', 'holi', 'christmas')),
        ('College Event', ('college event', 'campus event')),
        ('Party', ('party',)),
        ('Ceremony', ('ceremony',)),
        ('Invitation', ('invitation',)),
    ),
    'People & Family': (
        ('Family', ('family', 'mother', 'father', 'sister', 'brother')),
        ('Friends', ('friends',)),
        ('Portrait', ('portrait', 'selfie')),
        ('Group Photo', ('group photo', 'people together')),
    ),
    'Nature & Places': (
        ('Beach', ('beach',)),
        ('Mountain', ('mountain',)),
        ('City', ('cityscape', 'city view')),
        ('Building', ('building', 'architecture')),
        ('Monument', ('monument', 'landmark')),
        ('Sunset', ('sunset',)),
        ('Landscape', ('landscape', 'scenery')),
    ),
    'Animals & Pets': (
        ('Dog', ('dog', 'puppy')),
        ('Cat', ('cat', 'kitten')),
        ('Bird', ('bird', 'parrot')),
        ('Pet', ('pet',)),
        ('Wildlife', ('wildlife', 'wild animal')),
    ),
    'Food & Drinks': (
        ('Restaurant', ('restaurant',)),
        ('Dessert', ('dessert', 'cake', 'ice cream')),
        ('Beverage', ('beverage', 'coffee', 'tea', 'drink')),
        ('Recipe', ('recipe',)),
        ('Menu', ('menu',)),
        ('Food', ('food', 'meal', 'dish')),
    ),
    'Screenshots': (
        ('App Screenshot', ('app screen', 'application interface')),
        ('Website Screenshot', ('website screenshot', 'browser screenshot')),
        ('Chat Screenshot', ('chat screenshot', 'conversation screenshot')),
        ('Social Media Screenshot', ('instagram', 'facebook', 'social media')),
        ('Error Screenshot', ('error message', 'exception', 'stack trace')),
        ('Code Screenshot', ('source code', 'code screenshot', 'programming')),
        ('UI / Design Screenshot', ('user interface', 'ui design', 'settings screen')),
    ),
    'Notes & Documents': (
        ('Notes', ('handwritten notes', 'lecture notes', 'notebook')),
        ('Text Document', ('text document', 'typed document')),
        ('Form', ('form', 'application form')),
        ('Letter', ('letter', 'correspondence')),
        ('Reference', ('reference', 'manual', 'guide')),
    ),
}


def valid_subcategory(category: str, subcategory: str | None) -> bool:
    return isinstance(subcategory, str) and subcategory in CATEGORY_SUBCATEGORIES.get(category, ())


def infer_subcategory(category: str, text: str) -> str | None:
    normalized = f' {" ".join(str(text or "").lower().split())} '
    for subcategory, hints in SUBCATEGORY_HINTS.get(category, ()):
        if any(f' {hint} ' in normalized for hint in hints):
            return subcategory
    return None