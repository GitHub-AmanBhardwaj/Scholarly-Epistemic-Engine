import arxiv
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm

START_YEAR = 1993
END_YEAR = 2024
OUTPUT_FILE = 'arxiv_ai_metadata.csv'
PAPERS_PER_REQUEST = 10000

client = arxiv.Client(
  page_size = 1000,
  delay_seconds = 2.0,
  num_retries = 5
)

all_papers_data = []
first_write = True

print(f"Starting to fetch all arXiv metadata from {START_YEAR} to {END_YEAR}.")
print(f"Data will be saved incrementally to {OUTPUT_FILE}")

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        if year == datetime.now().year and month > datetime.now().month:
            break
            
        print(f"\n--- Fetching papers for {year}-{month:02d} ---")
        
        start_day = "01"
        if month == 12:
            end_day = "31"
        else:
            next_month_first_day = datetime(year, month + 1, 1)
            last_day_of_month = next_month_first_day - pd.Timedelta(days=1)
            end_day = str(last_day_of_month.day).zfill(2)

        query = f'(cat:cs.AI) AND submittedDate:[{year}{month:02d}{start_day} TO {year}{month:02d}{end_day}]'

        search = arxiv.Search(
            query=query,
            max_results=PAPERS_PER_REQUEST,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        try:
            results = list(client.results(search))
            
            if not results:
                print("No papers found for this period.")
                continue

            print(f"Found {len(results)} papers. Processing...")
            
            month_papers = []
            for result in tqdm(results):
                month_papers.append({
                    'id': result.entry_id,
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'pdf_url':result.pdf_url
                })
            
            df_month = pd.DataFrame(month_papers)
            if first_write:
                df_month.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)
                first_write = False
            else:
                df_month.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
            print(f"Saved {len(results)} papers for {year}-{month:02d}.")

        except Exception as e:
            print(f"An error occurred for {year}-{month:02d}: {e}")
            print("Skipping this month.")
        
        time.sleep(3)

print("\n--- Script Finished ---")
print(f"All fetched data saved in {OUTPUT_FILE}")