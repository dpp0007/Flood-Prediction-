"""
Data Acquisition Module

Data Source: Central Water Commission (CWC), Government of India
Website: https://ffs.india-water.gov.in/
"""

from src.data_acquisition.cwc_scraper import CWCScraper
from src.data_acquisition.cwc_api_scraper import CWCAPIScraper
from src.data_acquisition.mock_data_generator import MockDataGenerator

__all__ = ["CWCScraper", "CWCAPIScraper", "MockDataGenerator"]
