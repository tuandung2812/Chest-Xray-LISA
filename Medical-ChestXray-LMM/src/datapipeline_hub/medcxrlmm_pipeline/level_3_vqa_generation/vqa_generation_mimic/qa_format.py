QUESTION_TEMPLATES = {
'anatomy_disease_yes_no' : [
    # "Is there {disease} in the {anatomy} of this patient?",
    # "Does {anatomy} suffer from {disease}?",
    # "Does the patient have {disease} at {anatomy}?",
    
    "List all abnormalities in {anatomy}?",
    "What abnormalities are present in {anatomy}?",
    "Does {anatomy} suffer from any abnormalities?",

],
'anatomy_open': [
    "Are there any observations at {anatomy}?",
    "Does the patient's {anatomy} have any abnormalities?",
    # "Is there anything wrong with {anatomy}?",
    "Does {anatomy} suffer from any abnormalities?"
],

'adversarial_easy_at_anatomy_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",
],

'adversarial_easy_anatomy_at_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",
],

'adversarial_hard_at_anatomy_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",
],

'adversarial_hard_anatomy_at_disease_yes_no' : [
    "Is there {disease} in the {anatomy} of this patient?",
    "Does {anatomy} suffer from {disease}?",
    "Does the patient have {disease} at {anatomy}?",],

'adversarial_anatomy_open': [
    # "Are there any observations at {anatomy}?",
    # "Does the patient's {anatomy} have any abnormalities?",
    # "Is there anything wrong with {anatomy}?",
    "List all abnormalities in {anatomy}?",
    "What abnormalities are present in {anatomy}?",
    "Does {anatomy} suffer from any abnormalities?",
],


}

ANSWER_TEMPLATES = {
'anatomy_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is Yes",

'anatomy_open': "The location of {anatomy} is at <seg>. It suffers from {disease}",

'adversarial_easy_at_anatomy_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is No",

'adversarial_easy_anatomy_at_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is No",

'adversarial_hard_at_anatomy_disease_yes_no' :"The location of {anatomy} is at <seg>. The answer is No",

'adversarial_hard_anatomy_at_disease_yes_no' : "The location of {anatomy} is at <seg>. The answer is No",

'adversarial_anatomy_open':  "The location of {anatomy} is at <seg>. It has no abnormalities",

}