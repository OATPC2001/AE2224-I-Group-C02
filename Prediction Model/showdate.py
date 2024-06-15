from datetime import datetime, timedelta
import numpy as np


def date_to_numeric(date_str):
    # Define the reference date (1st March 2015)
    reference_date = datetime(2015, 3, 1)
    
    # Convert input date string to datetime object
    input_date = datetime.strptime(date_str, '%d-%m-%Y')
    
    # Calculate the difference in days between the input date and the reference date
    days_difference = (input_date - reference_date).days
    
    return days_difference


def numeric_to_date(numeric_value):
    reference_date = datetime(2015, 3, 1)
    target_date = reference_date + timedelta(days=numeric_value)
    return target_date.strftime('%d-%m-%Y')
