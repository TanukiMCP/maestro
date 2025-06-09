import asyncio
import csv
import os
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def scrape_partyslate_with_maestro():
    """Use maestro tools to scrape PartySlate wedding planners"""
    try:
        # Import maestro tools
        import sys
        sys.path.append('src')
        from maestro.enhanced_tools import MaestroTools
        
        maestro = MaestroTools()
        await maestro._ensure_initialized()
        
        logger.info("🚀 Starting PartySlate wedding planner scrape with MAESTRO")
        
        # Use maestro_scrape to scrape the directory
        scrape_args = {
            "url": "https://www.partyslate.com/find-vendors/wedding-planner",
            "scrape_type": "directory",
            "max_pages": 50,
            "max_items": 1000,
            "csv_filename": "wedding_planners_partyslate.csv",
            "fields_to_extract": [
                "Planner Name", "Company Name", "City", "State", 
                "Website", "Email", "Phone", "Description", 
                "Profile URL", "Social Links"
            ],
            "use_browser": "true",  # Force browser automation for PartySlate
            "timeout": 60,
            "wait_between_requests": 2.0
        }
        
        logger.info("🕷️ Executing directory scrape...")
        result = await maestro.handle_maestro_scrape(scrape_args)
        
        # Print the result
        if result and len(result) > 0:
            print(result[0].text)
    else:
            print("❌ No result returned from scrape")
            
        logger.info("✅ Scrape completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in scrape: {str(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(scrape_partyslate_with_maestro())
