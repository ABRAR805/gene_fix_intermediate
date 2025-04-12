import requests

def get_trials(gene):
    """
    Fetches clinical trials related to the given gene from ClinicalTrials.gov API.
    
    Args:
    gene (str): Gene name to search for clinical trials (e.g., 'TP53', 'KRAS').
    
    Returns:
    list: A list of trial titles and conditions related to the gene, or an error message if no trials are found.
    """
    # Define the base URL for the ClinicalTrials.gov API
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={gene}&format=json&pageSize=10"
    
    try:
        # Make the API request
        response = requests.get(url)
        
        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract trial details from the response
            trials = []
            if 'studies' in data:
                for study in data['studies']:
                    # Extract the relevant information
                    study_title = study.get('protocolSection', {}).get('identificationModule', {}).get('officialTitle', 'No title available')
                    condition = ', '.join(study.get('conditionsModule', {}).get('conditions', ['No conditions listed']))
                    trial_status = study.get('protocolSection', {}).get('statusModule', {}).get('overallStatus', 'Status unknown')
                    
                    trial_info = {
                        'Study Title': study_title,
                        'Condition': condition,
                        'Status': trial_status
                    }
                    
                    trials.append(trial_info)
                
                if trials:
                    return trials
                else:
                    return ["No trials found for this gene."]
            else:
                return ["No studies found in the response."]
        else:
            return [f"Error: {response.status_code} - Unable to fetch data from ClinicalTrials.gov."]
    
    except requests.exceptions.RequestException as e:
        # Handle network or API request errors
        return [f"Error: {str(e)} - Unable to connect to the trials database."]
