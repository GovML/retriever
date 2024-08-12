import arxiv
import time
import argparse

def download_papers(download_path):
    client = arxiv.Client()
    
    # Search for papers related to machine learning
    papers = arxiv.Search(
        query="machine learning",
        max_results=150
    ) 

    results = client.results(papers)
    
    # Download the PDFs  
    for paper in results:  
        f_name = paper.title
        f_name = f_name.replace(' ', '_').replace(':', '')
        f_name += '.pdf'
        print(f_name)
        try:
            paper.download_pdf(dirpath=download_path, filename=f_name)
            time.sleep(1)
        except:
            print(f'Unable to download {f_name}')
            time.sleep(1)

    print('Download complete!')

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Download arXiv papers related to machine learning.")
    parser.add_argument('--download_path', type=str, required=True, help='Path to the directory where PDFs will be saved.')

    args = parser.parse_args()

    # Call the function with the provided download path
    download_papers(args.download_path)
