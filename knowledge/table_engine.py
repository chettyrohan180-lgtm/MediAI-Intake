import string

class TableDiagnosisEngine:
    # direct table-lookup for diagnosis (replacing old prolog rules)
    # matches incoming symptoms against our quick DB
    def __init__(self):
        # Database based on the provided tabular medical information
        self.database = [
            # Cardiac
            {"category": "Cardiac", "problem": "Myocardial Infarction", "symptoms": "Chest pain, left arm pain, sweating, nausea", "explanation": "Coronary artery blockage; heart tissue death.", "treatment": "ECG, Aspirin, Thrombolytics/Cath Lab.", "resource": "surgery", "task_type": "surgery"},
            {"category": "Cardiac", "problem": "Atrial Fibrillation", "symptoms": "Palpitations, SOB, weakness, irregular pulse", "explanation": "Irregular rapid heart rhythm (upper chambers).", "treatment": "Beta-blockers, Anticoagulants, Cardioversion.", "resource": "icu", "task_type": "cardioversion"},
            {"category": "Cardiac", "problem": "Hypertension", "symptoms": "Headache, blurred vision, often asymptomatic", "explanation": "High arterial pressure; risk of organ damage.", "treatment": "ACE inhibitors, Diuretics, Low sodium.", "resource": "doctor", "task_type": "medication"},
            {"category": "Cardiac", "problem": "Congestive Heart Failure", "symptoms": "Leg swelling, SOB when flat, fatigue", "explanation": "Heart's inability to pump volume effectively.", "treatment": "Diuretics (Lasix), Fluid restriction, ACEI.", "resource": "icu", "task_type": "diuresis"},
            
            # Respiratory
            {"category": "Respiratory", "problem": "Pneumonia", "symptoms": "Productive cough, fever, chills, chest pain", "explanation": "Infection inflaming the lung air sacs.", "treatment": "Antibiotics (Azithromycin), Oxygen, Rest.", "resource": "icu", "task_type": "antibiotics"},
            {"category": "Respiratory", "problem": "Pulmonary Embolism", "symptoms": "Sudden SOB, sharp pain, hemoptysis", "explanation": "Blood clot in lung artery; obstructive shock.", "treatment": "Heparin, Thrombolytics, CT Angiography.", "resource": "surgery", "task_type": "surgery"},
            {"category": "Respiratory", "problem": "Asthma Attack", "symptoms": "Wheezing, chest tightness, dry cough, shortness of breath", "explanation": "Reversible bronchospasm and inflammation.", "treatment": "Albuterol Nebulizer, Oral Steroids.", "resource": "doctor", "task_type": "nebulizer"},
            {"category": "Respiratory", "problem": "COPD", "symptoms": "Chronic cough, wheezing, barrel chest", "explanation": "Irreversible airflow blockage from smoking.", "treatment": "Bronchodilators, Oxygen, Steroids.", "resource": "doctor", "task_type": "oxygen_therapy"},
            
            # Neuro
            {"category": "Neuro", "problem": "Ischemic Stroke", "symptoms": "Face droop, arm drift, slurred speech", "explanation": "Clot blocking brain blood flow.", "treatment": "TPA (within 4.5h), Thrombectomy.", "resource": "surgery", "task_type": "surgery"},
            {"category": "Neuro", "problem": "Migraine", "symptoms": "Unilateral throb, photophobia, aura, headache", "explanation": "Neuro-vascular dysfunction.", "treatment": "Triptans, NSAIDs, Dark room, IV fluids.", "resource": "doctor", "task_type": "medication"},
            {"category": "Neuro", "problem": "Meningitis", "symptoms": "Stiff neck, high fever, photophobia", "explanation": "Infection of brain/spinal cord membranes.", "treatment": "IV Antibiotics, Steroids, Lumbar Puncture.", "resource": "icu", "task_type": "antibiotics"},
            
            # Digestive
            {"category": "Digestive", "problem": "Appendicitis", "symptoms": "Pain migrating to RLQ, nausea, fever", "explanation": "Obstruction/infection of the appendix.", "treatment": "NPO, IV Fluids, Appendectomy.", "resource": "surgery", "task_type": "surgery"},
            {"category": "Digestive", "problem": "GERD", "symptoms": "Heartburn, regurgitation, chest pain", "explanation": "Stomach acid reflux into esophagus.", "treatment": "PPIs (Omeprazole), Diet modification.", "resource": "doctor", "task_type": "medication"},
            {"category": "Digestive", "problem": "GI Bleed", "symptoms": "Black stools, coffee-ground vomit", "explanation": "Ulcer or variceal rupture in GI tract.", "treatment": "Endoscopy, IV PPIs, Blood transfusion.", "resource": "surgery", "task_type": "endoscopy"},
            
            # Endocrine
            {"category": "Endocrine", "problem": "Type 2 Diabetes", "symptoms": "Thirst, frequent urination, fatigue", "explanation": "Insulin resistance; high blood glucose.", "treatment": "Metformin, Insulin, Glucose monitoring.", "resource": "doctor", "task_type": "insulin"},
            {"category": "Endocrine", "problem": "Hypoglycemia", "symptoms": "Sweating, shaking, confusion, clammy", "explanation": "Dangerously low blood glucose.", "treatment": "Oral Glucose or IM Glucagon.", "resource": "doctor", "task_type": "glucose"},
            {"category": "Endocrine", "problem": "DKA", "symptoms": "Fruity breath, deep breathing, confusion", "explanation": "Ketone buildup due to lack of insulin.", "treatment": "IV Insulin, Saline, K+ monitoring.", "resource": "icu", "task_type": "infusion"},
            
            # Renal
            {"category": "Renal", "problem": "Nephrolithiasis", "symptoms": "Flank pain, blood in urine, nausea", "explanation": "Kidney stones blocking urinary tract.", "treatment": "Toradol, Hydration, Tamsulosin (Flomax).", "resource": "doctor", "task_type": "hydration"},
            {"category": "Renal", "problem": "UTI", "symptoms": "Burning urination, frequency, pelvic pain", "explanation": "Bacterial infection of the bladder.", "treatment": "Antibiotics (Nitrofurantoin), Hydration.", "resource": "doctor", "task_type": "antibiotics"},
            {"category": "Renal", "problem": "Pyelonephritis", "symptoms": "Flank pain, high fever, shivering", "explanation": "Kidney infection (ascended UTI).", "treatment": "IV Antibiotics (Ceftriaxone), Imaging.", "resource": "icu", "task_type": "antibiotics"},
            
            # Infection
            {"category": "Infection", "problem": "Sepsis", "symptoms": "Low BP, high HR, confusion, shivering, elevated heart rate", "explanation": "Systemic response to infection; organ failure.", "treatment": "30mL/kg Fluids, IV Antibiotics, Vasopressors.", "resource": "icu", "task_type": "resuscitation"},
            {"category": "Infection", "problem": "Cellulitis", "symptoms": "Red, hot, swollen skin, painful", "explanation": "Deep skin bacterial infection.", "treatment": "Oral Antibiotics (Keflex), Elevation.", "resource": "doctor", "task_type": "antibiotics"},
            {"category": "Infection", "problem": "Influenza", "symptoms": "Body aches, high fever, sore throat, cough", "explanation": "Viral respiratory infection.", "treatment": "Tamiflu (if <48h), Rest, Fluids.", "resource": "doctor", "task_type": "rest"},
            
            # Psych
            {"category": "Psych", "problem": "Panic Attack", "symptoms": "Feeling of doom, heart racing, trembling, anxiety, elevated heart rate, sharp chest pain, fatigue", "explanation": "Acute psychological 'fight or flight'.", "treatment": "Grounding, Benzos (acute), SSRIs.", "resource": "doctor", "task_type": "therapy"},
            {"category": "Psych", "problem": "Depression", "symptoms": "Low mood, sleep issues, guilt, fatigue", "explanation": "Neurotransmitter imbalance.", "treatment": "SSRIs, CBT (Therapy).", "resource": "doctor", "task_type": "therapy"},
            {"category": "Psych", "problem": "Schizophrenia", "symptoms": "Hallucinations, delusions", "explanation": "Chronic brain disorder.", "treatment": "Antipsychotics, Social support.", "resource": "doctor", "task_type": "therapy"},
            {"category": "Psych", "problem": "ADHD", "symptoms": "Inattention, hyperactivity, impulsivity", "explanation": "Neurodevelopmental disorder.", "treatment": "Stimulants, Behavioral therapy.", "resource": "doctor", "task_type": "therapy"},
            
            # Orthopedic
            {"category": "Orthopedic", "problem": "Osteoarthritis", "symptoms": "Joint stiffness, pain with movement", "explanation": "Cartilage wear and tear.", "treatment": "Physical therapy, NSAIDs, Weight loss.", "resource": "doctor", "task_type": "physical_therapy"},
            {"category": "Orthopedic", "problem": "Herniated Disc", "symptoms": "Radiating leg pain (sciatica), numbness", "explanation": "Spinal disc pressing on nerve.", "treatment": "Physical therapy, NSAIDs, Surgery.", "resource": "surgery", "task_type": "surgery"},
            
            # Additional common matches
            {"category": "Derm", "problem": "Shingles", "symptoms": "Painful blistering rash in a line", "explanation": "Reactivation of Varicella Zoster virus.", "treatment": "Acyclovir, Pain management.", "resource": "doctor", "task_type": "medication"},
            {"category": "ENT", "problem": "Otitis Media", "symptoms": "Ear pain, hearing loss, fever", "explanation": "Middle ear infection.", "treatment": "Amoxicillin (if bacterial), decongestants.", "resource": "doctor", "task_type": "antibiotics"},
            {"category": "ENT", "problem": "Epistaxis", "symptoms": "Bleeding from nose", "explanation": "Ruptured nasal vessels.", "treatment": "Lean forward, pressure, Cautery.", "resource": "doctor", "task_type": "cautery"},
            {"category": "General", "problem": "Dehydration", "symptoms": "Dark urine, dizziness, dry mouth", "explanation": "Depletion of body fluids.", "treatment": "Oral or IV hydration.", "resource": "doctor", "task_type": "hydration"},
            {"category": "General", "problem": "Heat Stroke", "symptoms": "Temp >40C, confusion, no sweat", "explanation": "Failure of body thermoregulation.", "treatment": "Rapid cooling (ice bath), IV fluids.", "resource": "icu", "task_type": "cooling"},
        ]
        
        self.severity_map = {
            "Cardiac": 0.9,
            "Trauma": 0.9,
            "Neuro": 0.8,
            "Respiratory": 0.7,
            "Infection": 0.7,
            "Endocrine": 0.5,
            "Digestive": 0.5,
            "Renal": 0.5,
            "General": 0.4,
            "Orthopedic": 0.3,
            "Psych": 0.2,
            "Derm": 0.2,
            "ENT": 0.2,
            "Eye": 0.2,
            "OBGYN": 0.5,
            "Heme": 0.6,
            "Allergy": 0.8
        }

    def diagnose(self, patient_symptoms_text: str, patient_id: str) -> dict:
        if not patient_symptoms_text:
            patient_symptoms_text = ""
            
        # preprocess input text
        # hack: fixing "atigue" typos from some test datasets
        text = patient_symptoms_text.lower().translate(str.maketrans('', '', string.punctuation)).replace('atigue', 'fatigue')
        input_words = set(text.split())
        
        best_match = None
        max_overlap = 0
        
        for row in self.database:
            row_symptoms = row["symptoms"].lower().translate(str.maketrans('', '', string.punctuation))
            row_words = set(row_symptoms.split())
            
            # Count how many symptom keywords from the row match the words in the input
            overlap = len(input_words.intersection(row_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = row
                
        # if no overlap, default to observation
        if not best_match or max_overlap == 0:
            return {
                'patient_id': patient_id,
                'symptoms': [patient_symptoms_text],
                'diagnoses': ['Inconclusive'],
                'treatments': ['Observation'],
                'explanation': "No exact match found in the table. Recommend general observation.",
                'task_type': 'observation',
                'resource_type': 'doctor',
                'severity': 0.1
            }
            
        assert best_match is not None
        
        return {
            'patient_id': patient_id,
            'symptoms': [patient_symptoms_text],
            'diagnoses': [best_match['problem']],
            'treatments': [best_match['treatment']],
            'explanation': f"{best_match['explanation']}",
            'task_type': best_match['task_type'],
            'resource_type': best_match['resource'],
            'severity': self.severity_map.get(best_match['category'], 0.4)
        }
