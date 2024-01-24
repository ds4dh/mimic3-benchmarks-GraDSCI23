rules_normal = { # this dictionary says which index of the rules dictionary (below) is considered Healthy for this fetures
    'Oxygen saturation': 1,
    'Heart Rate': 1,
    'BMI': 2,
    'Glucose': 2,
    'Capillary refill rate': 1,
    'Systolic blood pressure': 1,
    'Diastolic blood pressure': 1,
    'Mean blood pressure': 1,
    'Glascow coma scale eye opening': 1,
    'Glascow coma scale verbal response': 1,
    'Glascow coma scale motor response': 1,
    'Glascow coma scale total': 1,
    'Respiratory rate': 1,
    'Temperature': 1,
    'pH': 1
}


rules = { # These healthy/unhealthy intervals are based on the Perl stats https://www.ncbi.nlm.nih.gov/books/NBK430685/
    'Oxygen saturation': [
        lambda x : x < 95, 
        lambda x : x >= 95],
    'Heart Rate': [
        lambda x : x < 60, 
        lambda x : (x>=60 and x <= 100),  
        lambda x : x > 100 
    ],
    'BMI': [
        lambda x : x < 16.5, 
        lambda x : (x >= 16.5 and x < 25), 
        lambda x : (x >= 25 and x < 30), 
        lambda x : (x >= 30 and x < 35), 
        lambda x : (x >= 35 and x < 40),
        lambda x : x >= 40
    ],
    'Glucose': [
        lambda x : x < 72, 
        lambda x : (x >= 72 and x <= 108), 
        lambda x : (x > 108 and x <= 125), 
        lambda x : x > 125, 
    ],
    'Capillary refill rate': [
        lambda x: x < 3,
        lambda x : (x >= 3), 
    ],
    'Systolic blood pressure': [
        lambda x: x < 90,
        lambda x : (x >= 90 and x <= 120),
        lambda x : (x >= 120 and x <= 129),
        lambda x : (x >= 129 and x <= 139),
        lambda x : x > 139,
    ],
    'Diastolic blood pressure': [
        lambda x : x < 60,
        lambda x : (x >= 60 and x <= 80),
        lambda x : (x > 80 and x <= 89),
        lambda x : (x > 89),
    ],
    'Mean blood pressure': [
        lambda x : x < 60,
        lambda x : (x >= 60 and x < 110),
        lambda x : (x >= 110 and x < 160),
        lambda x : (x >= 160),
        
    ],
    'Glascow coma scale eye opening': [
        lambda x: x < 4,
        lambda x: x >= 4,
    ],
    'Glascow coma scale verbal response': [
        lambda x: x < 5,
        lambda x: x >= 5,
    ],
    'Glascow coma scale motor response': [
        lambda x: x < 6,
        lambda x: x >= 6,    
    ],
    'Glascow coma scale total': [
        lambda x: x < 15,
        lambda x: x >= 15
    ],
    'Respiratory rate': [
        lambda x: x < 12,
        lambda x: (x >= 12 and x <= 20),
        lambda x: x > 20
    ],
    'Temperature': [
        lambda x: x < 35,
        lambda x: (x >= 35 and x <= 36.5),
        lambda x: (x > 36.5 and x <= 37.5),
        lambda x: (x > 37.5 and x <= 38.3),
        lambda x: (x > 38.3 and x <= 40),
        lambda x: (x > 40),
    ],
    'pH': [
        lambda x: x < 7.35,
        lambda x: (x >= 7.35 and x <= 7.45),
        lambda x: x > 7.45        
    ]

}


mapping = {
    'heart_related': [
        'Heart Rate', 
        'Systolic blood pressure', 
        'Diastolic blood pressure', 
        'Mean blood pressure'
    ],
    'glascow_coma_scale': [
        'Glascow coma scale eye opening',
        'Glascow coma scale verbal response',
        'Glascow coma scale motor response',
        'Glascow coma scale total'
    ],
    'respiratory_related': [
        'Oxygen saturation',
        'Respiratory rate'
    ],
    'pH': [],
    'Temperature': [],
    'Glucose': [],
    'BMI': [],
    'Capillary refill rate': []
}
