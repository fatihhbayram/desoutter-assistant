"""
Desoutter Freshdesk Support Portal - Async Ticket Scraper
Scrapes support tickets with PDF attachment extraction for RAG

Features:
- Async HTTP requests (aiohttp) for performance
- Login session management
- PDF text extraction (pdfplumber + PyPDF2)
- Checkpoint system for resumable scraping
- Product detection from ticket content
"""
import asyncio
import aiohttp
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup

from config.settings import (
    FRESHDESK_BASE_URL, FRESHDESK_EMAIL, FRESHDESK_PASSWORD,
    TICKET_REQUEST_DELAY, TICKET_MAX_PAGES, TICKET_CHECKPOINT_EVERY,
    TICKET_DOWNLOAD_PDFS, TICKET_PDF_DIR, DATA_DIR, PART_NUMBER_PATTERNS
)
from src.database.models import TicketModel, TicketComment, TicketAttachment
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# PDF processing (optional)
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("pdfplumber not installed. PDF extraction disabled. Install: pip install pdfplumber")

try:
    from PyPDF2 import PdfReader
    PYPDF2_SUPPORT = True
except ImportError:
    PYPDF2_SUPPORT = False


class TicketScraper:
    """Async scraper for Desoutter Freshdesk support tickets"""
    
    # Known product model patterns for detection
    MODEL_PATTERNS = [
        r'\b(CVI[R]?[L]?\s*[23I]?[I]?)\b',  # CVI3, CVIR, CVIL2, etc.
        r'\b(E[A-Z]{1,3}[B-Z]?[\d]*[-]?[A-Z\d]*)\b',  # EAD, EPB, ERS, etc.
        r'\b(DVT[\d]*)\b',  # DVT series
        r'\b(QST[\d]*)\b',  # QST series
        r'\b(PF[\d]*)\b',   # PF series
        r'\b(XPB[-\s]?(Modular|One))\b',  # XPB series
        r'\b(Connect\s*Unit)\b',  # Connect Unit
    ]
    
    # Tag extraction patterns
    TAG_PATTERNS = {
        'calibration': r'calibrat|kalibrasy',
        'error_code': r'error\s*(code)?|hata\s*(kodu)?|E\d{2,4}',
        'wifi': r'wi[-]?fi|wireless|kablosuz',
        'connection': r'connect|bağlant|pairing',
        'torque': r'torque|tork|moment',
        'battery': r'battery|batarya|pil|şarj',
        'motor': r'motor|spindle',
        'software': r'software|firmware|yazılım|update|güncelle',
        'installation': r'install|kurulum|setup',
        'maintenance': r'maintenance|bakım|service',
    }
    
    def __init__(
        self,
        base_url: str = FRESHDESK_BASE_URL,
        email: str = FRESHDESK_EMAIL,
        password: str = FRESHDESK_PASSWORD
    ):
        """
        Initialize ticket scraper
        
        Args:
            base_url: Freshdesk portal URL
            email: Login email
            password: Login password
        """
        self.base_url = base_url
        self.email = email
        self.password = password
        self.session: Optional[aiohttp.ClientSession] = None
        self.logged_in = False
        
        # Data directories
        self.data_dir = DATA_DIR / "tickets"
        self.pdf_dir = TICKET_PDF_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile patterns
        self.part_patterns = [re.compile(p, re.IGNORECASE) for p in PART_NUMBER_PATTERNS]
        self.model_patterns = [re.compile(p, re.IGNORECASE) for p in self.MODEL_PATTERNS]
        
        # Stats
        self.stats = {
            "pages_scraped": 0,
            "tickets_found": 0,
            "tickets_scraped": 0,
            "tickets_failed": 0,
            "pdfs_downloaded": 0,
            "pdfs_extracted": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        timeout = aiohttp.ClientTimeout(total=60)
        
        # Create session with cookie jar for login
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            cookie_jar=aiohttp.CookieJar()
        )
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            logger.info(f"Ticket scraper stats: {self.stats}")
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    async def login(self) -> bool:
        """
        Login to Freshdesk portal
        
        Returns:
            True if login successful
        """
        if not self.email or not self.password:
            logger.error("Freshdesk credentials not configured. Set FRESHDESK_EMAIL and FRESHDESK_PASSWORD")
            return False
        
        login_url = f"{self.base_url}/en/support/login"
        
        logger.info("Fetching login page...")
        async with self.session.get(login_url) as resp:
            if resp.status != 200:
                logger.error(f"Failed to fetch login page: HTTP {resp.status}")
                return False
            
            html = await resp.text()
        
        # Extract CSRF token
        soup = BeautifulSoup(html, 'html.parser')
        csrf = None
        
        meta = soup.find('meta', {'name': 'csrf-token'})
        if meta:
            csrf = meta.get('content')
        
        inp = soup.find('input', {'name': 'authenticity_token'})
        if inp:
            csrf = inp.get('value')
        
        # Build login payload
        payload = {
            "user_session[email]": self.email,
            "user_session[password]": self.password,
            "user_session[remember_me]": "1",
        }
        if csrf:
            payload["authenticity_token"] = csrf
        
        logger.info("Submitting login...")
        async with self.session.post(login_url, data=payload, allow_redirects=True) as resp:
            html = await resp.text()
            
            # Check if login successful
            if "logout" in html.lower() or "login" not in str(resp.url).lower():
                logger.info("✅ Login successful!")
                self.logged_in = True
                return True
        
        logger.error("❌ Login failed - check credentials")
        return False
    
    # =========================================================================
    # Ticket ID Collection
    # =========================================================================
    
    async def get_ticket_ids_from_page(self, page_num: int) -> List[int]:
        """
        Get ticket IDs from a listing page
        
        Args:
            page_num: Page number to scrape
            
        Returns:
            List of ticket IDs
        """
        url = f"{self.base_url}/en/support/tickets?page={page_num}"
        
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return []
                
                html = await resp.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            ids = []
            
            for link in soup.find_all('a', href=re.compile(r'/tickets/(\d+)')):
                match = re.search(r'/tickets/(\d+)', link['href'])
                if match:
                    ids.append(int(match.group(1)))
            
            return list(set(ids))
            
        except Exception as e:
            logger.warning(f"Error fetching page {page_num}: {e}")
            return []
    
    async def collect_ticket_ids(
        self,
        start_page: int = 1,
        end_page: int = TICKET_MAX_PAGES,
        delay: float = TICKET_REQUEST_DELAY
    ) -> List[int]:
        """
        Collect all ticket IDs from listing pages
        
        Args:
            start_page: First page to scrape
            end_page: Last page to scrape
            delay: Delay between requests
            
        Returns:
            List of all ticket IDs
        """
        if not self.logged_in:
            logger.error("Not logged in. Call login() first.")
            return []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: Collecting Ticket IDs (pages {start_page}-{end_page})")
        logger.info(f"{'='*60}\n")
        
        all_ids = []
        total_pages = end_page - start_page + 1
        
        for i, page in enumerate(range(start_page, end_page + 1), 1):
            ids = await self.get_ticket_ids_from_page(page)
            all_ids.extend(ids)
            self.stats["pages_scraped"] += 1
            
            if i % 20 == 0 or i == total_pages:
                logger.info(f"[{i}/{total_pages}] Found {len(all_ids)} tickets so far")
            
            await asyncio.sleep(delay)
        
        # Deduplicate and sort
        all_ids = sorted(list(set(all_ids)))
        self.stats["tickets_found"] = len(all_ids)
        
        # Save to file
        self._save_json(self.data_dir / "ticket_ids.json", {
            "count": len(all_ids),
            "ids": all_ids,
            "scraped_at": datetime.now().isoformat()
        })
        
        logger.info(f"\n✅ Collected {len(all_ids)} unique ticket IDs")
        return all_ids
    
    # =========================================================================
    # Ticket Detail Scraping
    # =========================================================================
    
    async def scrape_ticket(self, ticket_id: int, download_pdfs: bool = True) -> Optional[TicketModel]:
        """
        Scrape a single ticket's details
        
        Args:
            ticket_id: Ticket ID to scrape
            download_pdfs: Whether to download and extract PDFs
            
        Returns:
            TicketModel or None if failed
        """
        url = f"{self.base_url}/en/support/tickets/{ticket_id}"
        
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                
                # Check if redirected to login
                if "login" in str(resp.url).lower():
                    logger.warning("Session expired, need re-login")
                    return None
                
                html = await resp.text()
            
            # Parse ticket
            ticket = self._parse_ticket_html(html, ticket_id, url)
            
            # Download and extract PDFs
            if download_pdfs and ticket.attachments:
                await self._process_attachments(ticket)
            
            # Detect products and tags
            self._enrich_ticket(ticket)
            
            return ticket
            
        except Exception as e:
            logger.warning(f"Error scraping ticket {ticket_id}: {e}")
            return None
    
    def _parse_ticket_html(self, html: str, ticket_id: int, url: str) -> TicketModel:
        """Parse ticket HTML into model"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Title
        heading = soup.find('h2', class_='heading')
        if heading:
            title = re.sub(r'^#\d+\s*', '', heading.get_text().strip())
        else:
            title = f"Ticket #{ticket_id}"
        
        # Description (original problem)
        description = None
        desc_section = soup.find('section', id='ticket-description')
        if desc_section:
            description = self._parse_comment(desc_section)
        
        # Comments/Replies
        comments = []
        for section in soup.find_all('section', class_=re.compile(r'comment')):
            if section.get('id') == 'ticket-description':
                continue
            
            comment = self._parse_comment(section)
            if comment:
                # Check if agent response
                if 'agent' in str(section.get('class', [])).lower():
                    comment.is_agent = True
                comments.append(comment)
        
        # Attachments
        attachments = []
        for att in soup.find_all('div', class_='attachment'):
            file_type_elem = att.find('span', class_='file-type')
            filename_elem = att.find('a', class_='filename')
            
            if filename_elem:
                att_url = filename_elem.get('href', '')
                if not att_url.startswith('http'):
                    att_url = self.base_url + att_url
                
                attachments.append(TicketAttachment(
                    filename=filename_elem.get('data-original-title') or filename_elem.get_text().strip(),
                    url=att_url,
                    file_type=file_type_elem.get_text().strip().lower() if file_type_elem else "unknown"
                ))
        
        # Check if resolved (has agent response)
        is_resolved = any(c.is_agent for c in comments)
        
        return TicketModel(
            ticket_id=ticket_id,
            title=title,
            url=url,
            description=description,
            comments=comments,
            attachments=attachments,
            is_resolved=is_resolved
        )
    
    def _parse_comment(self, section) -> Optional[TicketComment]:
        """Parse a comment section"""
        author = "Unknown"
        content = ""
        date = None
        
        user_elem = section.find('h4', class_='user-name')
        if user_elem:
            author = user_elem.get_text().strip()
        
        time_elem = section.find('span', class_='timeago')
        if time_elem:
            date = time_elem.get('title', '')
        
        content_elem = section.find('div', class_='p-desc')
        if content_elem:
            # Convert <br> to newlines
            for br in content_elem.find_all('br'):
                br.replace_with('\n')
            
            text = content_elem.get_text()
            lines = [l.strip() for l in text.split('\n')]
            content = '\n'.join(l for l in lines if l)
        
        if not content:
            return None
        
        return TicketComment(
            author=author,
            content=content,
            date=date,
            is_agent=False
        )
    
    # =========================================================================
    # PDF Processing
    # =========================================================================
    
    async def _process_attachments(self, ticket: TicketModel):
        """Download and extract text from PDF attachments"""
        for att in ticket.attachments:
            if att.file_type.lower() != 'pdf' and not att.filename.lower().endswith('.pdf'):
                continue
            
            logger.debug(f"  Downloading PDF: {att.filename}")
            
            # Download PDF
            filepath = await self._download_attachment(att.url, att.filename)
            if not filepath:
                continue
            
            att.local_path = str(filepath)
            self.stats["pdfs_downloaded"] += 1
            
            # Extract text
            text = self._extract_pdf_text(filepath)
            if text:
                att.content = text
                ticket.has_pdf_content = True
                self.stats["pdfs_extracted"] += 1
                logger.debug(f"  Extracted {len(text)} chars from {att.filename}")
    
    async def _download_attachment(self, url: str, filename: str) -> Optional[Path]:
        """Download an attachment file using authenticated session"""
        try:
            async with self.session.get(url, allow_redirects=True) as resp:
                # Check if redirected to login page
                if resp.status != 200:
                    logger.warning(f"Download failed: HTTP {resp.status} for {filename}")
                    return None
                
                # Check if redirected to login (session expired)
                if "login" in str(resp.url).lower():
                    logger.warning(f"Session expired, redirected to login for {filename}")
                    return None
                
                # Check content type - should be PDF, not HTML
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    logger.warning(f"Got HTML instead of PDF for {filename} (auth issue?)")
                    return None
                
                # Generate unique filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                safe_filename = re.sub(r'[^\w\-.]', '_', filename)
                filepath = self.pdf_dir / f"{url_hash}_{safe_filename}"
                
                # Write file
                content = await resp.read()
                
                # Verify it's actually a PDF (starts with %PDF)
                if not content.startswith(b'%PDF'):
                    logger.warning(f"Downloaded content is not a PDF for {filename}")
                    return None
                
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                logger.info(f"  ✅ Downloaded: {filename} ({len(content)} bytes)")
                return filepath
                
        except Exception as e:
            logger.warning(f"Download error for {filename}: {e}")
            return None
    
    def _extract_pdf_text(self, filepath: Path) -> Optional[str]:
        """Extract text from PDF file"""
        text = ""
        
        # Try pdfplumber first (better extraction)
        if PDF_SUPPORT:
            try:
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.debug(f"pdfplumber error: {e}")
        
        # Fallback to PyPDF2
        if PYPDF2_SUPPORT:
            try:
                reader = PdfReader(filepath)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.debug(f"PyPDF2 error: {e}")
        
        return None
    
    # =========================================================================
    # Ticket Enrichment
    # =========================================================================
    
    def _enrich_ticket(self, ticket: TicketModel):
        """Detect products, models, and tags from ticket content"""
        # Collect all text content
        all_text = ticket.title + "\n"
        
        if ticket.description:
            all_text += ticket.description.content + "\n"
        
        for comment in ticket.comments:
            all_text += comment.content + "\n"
        
        for att in ticket.attachments:
            if att.content:
                all_text += att.content + "\n"
        
        # Detect part numbers
        for pattern in self.part_patterns:
            matches = pattern.findall(all_text)
            ticket.related_products.extend(matches)
        
        ticket.related_products = list(set(ticket.related_products))
        
        # Detect model names
        for pattern in self.model_patterns:
            matches = pattern.findall(all_text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                ticket.related_models.append(match.upper())
        
        ticket.related_models = list(set(ticket.related_models))
        
        # Auto-tag based on content
        all_text_lower = all_text.lower()
        for tag, pattern in self.TAG_PATTERNS.items():
            if re.search(pattern, all_text_lower, re.IGNORECASE):
                ticket.tags.append(tag)
    
    # =========================================================================
    # Bulk Scraping
    # =========================================================================
    
    async def scrape_tickets(
        self,
        ticket_ids: List[int],
        download_pdfs: bool = TICKET_DOWNLOAD_PDFS,
        delay: float = TICKET_REQUEST_DELAY,
        checkpoint_every: int = TICKET_CHECKPOINT_EVERY
    ) -> List[TicketModel]:
        """
        Scrape multiple tickets with checkpoint support
        
        Args:
            ticket_ids: List of ticket IDs to scrape
            download_pdfs: Whether to download PDFs
            delay: Delay between requests
            checkpoint_every: Save checkpoint every N tickets
            
        Returns:
            List of scraped TicketModels
        """
        if not self.logged_in:
            logger.error("Not logged in. Call login() first.")
            return []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2: Scraping Ticket Details ({len(ticket_ids)} tickets)")
        logger.info(f"PDF download: {'Enabled' if download_pdfs else 'Disabled'}")
        logger.info(f"{'='*60}\n")
        
        tickets = []
        checkpoint_file = self.data_dir / "checkpoint.json"
        start_idx = 0
        
        # Resume from checkpoint if exists
        if checkpoint_file.exists():
            checkpoint = self._load_json(checkpoint_file)
            tickets = [TicketModel(**t) for t in checkpoint.get("tickets", [])]
            start_idx = checkpoint.get("last_index", 0) + 1
            logger.info(f"Resuming from checkpoint: {start_idx}/{len(ticket_ids)}")
        
        total = len(ticket_ids)
        
        for i, ticket_id in enumerate(ticket_ids[start_idx:], start_idx + 1):
            logger.info(f"[{i}/{total}] Ticket #{ticket_id}...", extra={"end": " "})
            
            ticket = await self.scrape_ticket(ticket_id, download_pdfs)
            
            if ticket:
                tickets.append(ticket)
                self.stats["tickets_scraped"] += 1
                
                pdf_count = sum(1 for a in ticket.attachments if a.content)
                title_preview = ticket.title[:35] + "..." if len(ticket.title) > 35 else ticket.title
                logger.info(f"✓ {title_preview} ({pdf_count} PDFs)")
            else:
                self.stats["tickets_failed"] += 1
                logger.info("✗ Failed")
            
            # Save checkpoint
            if i % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_file, tickets, i - 1)
                logger.info(f"  [checkpoint] {len(tickets)} tickets saved")
            
            await asyncio.sleep(delay)
        
        # Final checkpoint
        self._save_checkpoint(checkpoint_file, tickets, total - 1)
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ Scraping Complete!")
        logger.info(f"  Success: {self.stats['tickets_scraped']}")
        logger.info(f"  Failed: {self.stats['tickets_failed']}")
        logger.info(f"  PDFs downloaded: {self.stats['pdfs_downloaded']}")
        logger.info(f"  PDFs extracted: {self.stats['pdfs_extracted']}")
        logger.info(f"{'='*60}")
        
        return tickets
    
    def _save_checkpoint(self, filepath: Path, tickets: List[TicketModel], last_index: int):
        """Save checkpoint data"""
        self._save_json(filepath, {
            "last_index": last_index,
            "tickets": [t.to_dict() for t in tickets]
        })
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _save_json(self, path: Path, data: dict):
        """Save data to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_json(self, path: Path) -> dict:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_ticket_ids(self, path: Optional[Path] = None) -> List[int]:
        """Load previously saved ticket IDs"""
        if path is None:
            path = self.data_dir / "ticket_ids.json"
        
        data = self._load_json(path)
        ticket_ids = data.get("ids", [])
        logger.info(f"Loaded {len(ticket_ids)} ticket IDs from {path}")
        return ticket_ids
    
    def export_for_rag(self, tickets: List[TicketModel], filename: str = "tickets_rag.json") -> List[dict]:
        """
        Export tickets in RAG-ready format
        
        Args:
            tickets: List of TicketModels
            filename: Output filename
            
        Returns:
            List of RAG documents
        """
        rag_docs = [ticket.to_rag_document() for ticket in tickets]
        
        filepath = self.data_dir / filename
        self._save_json(filepath, rag_docs)
        
        logger.info(f"✅ Exported {len(rag_docs)} RAG documents to {filepath}")
        return rag_docs
