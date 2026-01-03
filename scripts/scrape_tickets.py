#!/usr/bin/env python3
"""
Scrape Desoutter Freshdesk Support Tickets

This script scrapes support tickets from the Desoutter Freshdesk portal,
including PDF attachments with text extraction for RAG.

Usage:
    # Full scrape (all tickets)
    python scripts/scrape_tickets.py --full
    
    # Last N pages only
    python scripts/scrape_tickets.py --pages 50
    
    # Resume from checkpoint
    python scripts/scrape_tickets.py --resume
    
    # Quick test (last 3 pages)
    python scripts/scrape_tickets.py --test
    
    # Without PDF download (faster)
    python scripts/scrape_tickets.py --pages 100 --no-pdf

Environment Variables Required:
    FRESHDESK_EMAIL - Login email
    FRESHDESK_PASSWORD - Login password
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    FRESHDESK_EMAIL, FRESHDESK_PASSWORD, 
    TICKET_MAX_PAGES, TICKET_DOWNLOAD_PDFS
)
from src.scraper.ticket_scraper_sync import TicketScraperSync
from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main execution (sync version)"""
    parser = argparse.ArgumentParser(description="Scrape Desoutter Freshdesk tickets")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--full", action="store_true", help="Full scrape (all pages)")
    mode_group.add_argument("--pages", type=int, help="Scrape last N pages")
    mode_group.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    mode_group.add_argument("--test", action="store_true", help="Test mode (last 3 pages)")
    mode_group.add_argument("--ids-only", action="store_true", help="Only collect ticket IDs")
    
    # Options
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF downloads")
    parser.add_argument("--no-save", action="store_true", help="Don't save to MongoDB")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    
    args = parser.parse_args()
    
    # Check credentials
    if not FRESHDESK_EMAIL or not FRESHDESK_PASSWORD:
        logger.error("❌ Freshdesk credentials not configured!")
        logger.error("   Set FRESHDESK_EMAIL and FRESHDESK_PASSWORD environment variables")
        sys.exit(1)
    
    start_time = datetime.now()
    download_pdfs = TICKET_DOWNLOAD_PDFS and not args.no_pdf
    
    logger.info("=" * 70)
    logger.info("🎫 Desoutter Freshdesk Ticket Scraper (Sync Version)")
    logger.info("=" * 70)
    logger.info(f"Mode: {'Full' if args.full else 'Pages: ' + str(args.pages) if args.pages else 'Resume' if args.resume else 'Test' if args.test else 'IDs Only'}")
    logger.info(f"PDF Download: {'Enabled' if download_pdfs else 'Disabled'}")
    logger.info(f"MongoDB Save: {'Enabled' if not args.no_save else 'Disabled'}")
    logger.info("=" * 70)
    
    # Create scraper instance
    scraper = TicketScraperSync()
    
    try:
        # Login
        if not scraper.login():
            logger.error("❌ Login failed. Exiting.")
            sys.exit(1)
        
        ticket_ids = []
        tickets = []
        
        # Determine page range
        if args.full:
            start_page = 1
            end_page = TICKET_MAX_PAGES
        elif args.pages:
            start_page = max(1, TICKET_MAX_PAGES - args.pages + 1)
            end_page = TICKET_MAX_PAGES
        elif args.test:
            start_page = max(1, TICKET_MAX_PAGES - 2)
            end_page = TICKET_MAX_PAGES
        elif args.resume or args.ids_only:
            start_page = None
            end_page = None
        
        # Phase 1: Collect ticket IDs (or load from checkpoint)
        if args.resume:
            ticket_ids = scraper.load_ticket_ids()
        elif start_page is not None:
            ticket_ids = scraper.collect_ticket_ids(
                start_page=start_page,
                end_page=end_page,
                delay=args.delay
            )
        
        if args.ids_only:
            logger.info(f"✅ Collected {len(ticket_ids)} ticket IDs")
            return
        
        if not ticket_ids:
            logger.error("❌ No ticket IDs to scrape")
            sys.exit(1)
        
        # Phase 2: Scrape ticket details
        tickets = scraper.scrape_tickets(
            ticket_ids=ticket_ids,
            download_pdfs=download_pdfs,
            delay=args.delay
        )
        
        if not tickets:
            logger.warning("⚠️  No tickets scraped")
            return
        
        # Phase 3: Export for RAG
        scraper.export_for_rag(tickets, "tickets_rag.json")
        
        # Phase 4: Save to MongoDB (optional)
        if not args.no_save:
            logger.info("\n💾 Saving to MongoDB...")
            
            with MongoDBClient() as db:
                # Create indexes
                db.create_ticket_indexes()
                
                # Bulk upsert
                ticket_dicts = [t.to_dict() for t in tickets]
                stats = db.bulk_upsert_tickets(ticket_dicts)
                
                logger.info(f"   Inserted: {stats['inserted']}")
                logger.info(f"   Updated: {stats['updated']}")
                logger.info(f"   Errors: {stats['errors']}")
                logger.info(f"   Total in DB: {db.count_tickets()}")
    
    finally:
        # Close session
        scraper.close()
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n{'='*70}")
    logger.info("✅ SCRAPING COMPLETE!")
    logger.info(f"   Total time: {elapsed:.1f} seconds")
    logger.info(f"   Tickets scraped: {len(tickets)}")
    if tickets:
        resolved = sum(1 for t in tickets if t.is_resolved)
        with_pdf = sum(1 for t in tickets if t.has_pdf_content)
        logger.info(f"   Resolved tickets: {resolved}")
        logger.info(f"   With PDF content: {with_pdf}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
