def scrape_statewise_petrol():
    """Scrapes state-wise petrol prices from government portal"""
    url = "https://www.ppac.gov.in/WriteReadData/userfiles/file/RetailPriceofMS&HSD.xlsx"
    
    try:
        # Download the Excel file
        response = requests.get(url)
        response.raise_for_status()
        
        # Read the petrol prices sheet (MS = Motor Spirit = Petrol)
        with open('temp_petrol.xlsx', 'wb') as f:
            f.write(response.content)
            
        df = pd.read_excel('temp_petrol.xlsx', sheet_name='MS Prices')
        
        # Clean the data
        df = df[['Date', 'State', 'Petrol_Price']]
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Petrol_Price'] = df['Petrol_Price'].astype(float)
        
        return df.dropna()
    
    except Exception as e:
        print(f"Failed to get state-wise prices: {str(e)}")
        return pd.DataFrame()