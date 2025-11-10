"""
SEC Financial Reports Scraper

This module fetches 10-K and 10-Q reports from the SEC EDGAR database.
It uses the SEC's REST API for efficient data retrieval.
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECScraper:
    """Scrapes financial reports from SEC EDGAR database."""
    
    BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    FILING_API_URL = "https://data.sec.gov/submissions/CIK{}.json"
    
    def __init__(self, output_dir: str = "data/raw_reports"):
        """
        Initialize the SEC scraper.
        
        Args:
            output_dir: Directory to save downloaded reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Financial Analyst Bot v1.0)'
        })
    
    def get_cik(self, company_name: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            CIK number as string or None if not found
        """
        try:
            params = {
                'company': company_name,
                'owner': 'exclude',
                'action': 'getcompany',
                'output': 'json'
            }
            
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('cik_lookup'):
                # CIK might have leading zeros stripped
                cik = str(data['cik_lookup'][company_name]).zfill(10)
                logger.info(f"Found CIK {cik} for {company_name}")
                return cik
            
            logger.warning(f"Could not find CIK for {company_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {company_name}: {e}")
            return None
    
    def get_filings(
        self,
        cik: str,
        filing_types: List[str] = ["10-K", "10-Q"],
        limit: int = 10
    ) -> List[Dict]:
        """
        Get recent filings for a company.
        
        Args:
            cik: CIK number (10 digits)
            filing_types: Types of filings to retrieve
            limit: Maximum number of filings to retrieve
            
        Returns:
            List of filing metadata
        """
        try:
            # Pad CIK to 10 digits
            cik = str(cik).zfill(10)
            url = self.FILING_API_URL.format(cik)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings = []
            
            # Extract recent filings
            for filing in data.get('filings', {}).get('recent', []):
                if filing['form'] in filing_types:
                    filings.append({
                        'accession_number': filing['accessionNumber'],
                        'filing_date': filing['filingDate'],
                        'report_date': filing['reportDate'],
                        'form_type': filing['form'],
                        'filing_url': f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={filing['accessionNumber']}&xbrl_type=v"
                    })
                    
                    if len(filings) >= limit:
                        break
            
            logger.info(f"Retrieved {len(filings)} filings for CIK {cik}")
            return filings
            
        except Exception as e:
            logger.error(f"Error getting filings for CIK {cik}: {e}")
            return []
    
    def download_filing(
        self,
        cik: str,
        accession_number: str,
        filing_type: str
    ) -> Optional[str]:
        """
        Download the full text of a filing.
        
        Args:
            cik: CIK number
            accession_number: Accession number of the filing
            filing_type: Type of filing (10-K, 10-Q, etc.)
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            cik = str(cik).zfill(10)
            # Remove hyphens from accession number for URL
            accession_clean = accession_number.replace('-', '')
            
            # Construct URL to filing full text
            url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_number}&xbrl_type=v"
            
            # For direct text file access
            text_url = f"https://www.sec.gov/Archives/edgar/{cik}/{accession_clean}/{accession_clean}.txt"
            
            response = self.session.get(text_url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Direct text not available, trying HTML...")
                # Try to get the main document
                html_url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_number}&xbrl_type=v"
                response = self.session.get(html_url, timeout=15)
            
            response.raise_for_status()
            
            # Save the file
            filename = f"{cik}_{filing_type}_{accession_clean}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading filing {accession_number}: {e}")
            return None
    
    def scrape_company(
        self,
        company_name: str,
        filing_types: List[str] = ["10-K", "10-Q"],
        num_filings: int = 5
    ) -> List[str]:
        """
        Scrape all recent filings for a company.
        
        Args:
            company_name: Name of the company
            filing_types: Types of filings to retrieve
            num_filings: Number of filings to retrieve
            
        Returns:
            List of paths to downloaded files
        """
        # Get CIK
        cik = self.get_cik(company_name)
        if not cik:
            return []
        
        # Get filings metadata
        filings = self.get_filings(cik, filing_types, num_filings)
        
        # Download each filing
        downloaded_files = []
        for filing in filings:
            time.sleep(0.1)  # Rate limiting - be respectful to SEC servers
            
            filepath = self.download_filing(
                cik,
                filing['accession_number'],
                filing['form_type']
            )
            
            if filepath:
                downloaded_files.append(filepath)
        
        return downloaded_files


def scrape_multiple_companies(
    companies: List[str],
    num_filings: int = 5
) -> Dict[str, List[str]]:
    """
    Scrape filings for multiple companies.
    
    Args:
        companies: List of company names
        num_filings: Number of filings per company
        
    Returns:
        Dictionary mapping company names to lists of file paths
    """
    scraper = SECScraper()
    results = {}
    
    for company in companies:
        logger.info(f"Scraping {company}...")
        files = scraper.scrape_company(company, num_filings=num_filings)
        results[company] = files
        time.sleep(1)  # Rate limiting between companies
    
    return results


if __name__ == "__main__":
    # Example usage
    companies = ["Apple", "Microsoft", "Tesla"]
    
    results = scrape_multiple_companies(companies, num_filings=3)
    
    for company, files in results.items():
        print(f"\n{company}: {len(files)} files downloaded")
        for f in files:
            print(f"  - {f}")
