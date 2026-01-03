"""
Desoutter Support Portal - Sync Ticket Scraper with PDF Support
Original working version using requests library

- Ticket'ları çeker
- PDF attachment'ları indirir
- PDF'lerden text çıkarır
- RAG için export eder
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from datetime import datetime
from pathlib import Path
import hashlib

from config.settings import (
    FRESHDESK_BASE_URL, FRESHDESK_EMAIL, FRESHDESK_PASSWORD,
    TICKET_REQUEST_DELAY, TICKET_MAX_PAGES, TICKET_CHECKPOINT_EVERY,
    TICKET_DOWNLOAD_PDFS, TICKET_PDF_DIR, DATA_DIR, PART_NUMBER_PATTERNS
)
from src.database.models import TicketModel, TicketComment, TicketAttachment
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# PDF işleme için
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("pdfplumber not installed. PDF extraction disabled.")

try:
    from PyPDF2 import PdfReader
    PYPDF2_SUPPORT = True
except ImportError:
    PYPDF2_SUPPORT = False


class TicketScraperSync:
    """Sync scraper for Desoutter Freshdesk support tickets (requests-based)"""
    
    # Known product model patterns for detection
    MODEL_PATTERNS = [
        r'\b(CVI[R]?[L]?\s*[23I]?[I]?)\b',
        r'\b(E[A-Z]{1,3}[B-Z]?[\d]*[-]?[A-Z\d]*)\b',
        r'\b(DVT[\d]*)\b',
        r'\b(QST[\d]*)\b',
        r'\b(PF[\d]*)\b',
        r'\b(XPB[-\s]?(Modular|One))\b',
        r'\b(Connect\s*Unit)\b',
    ]
    
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
        self.base_url = base_url
        self.email = email
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        self.logged_in = False
        
        # Directories
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
    
    def login(self) -> bool:
        """Login to Freshdesk portal"""
        if not self.email or not self.password:
            logger.error("Freshdesk credentials not configured")
            return False
        
        login_url = f"{self.base_url}/en/support/login"
        
        logger.info("Fetching login page...")
        resp = self.session.get(login_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # CSRF token
        csrf = None
        meta = soup.find('meta', {'name': 'csrf-token'})
        if meta:
            csrf = meta.get('content')
        inp = soup.find('input', {'name': 'authenticity_token'})
        if inp:
            csrf = inp.get('value')
        
        payload = {
            "user_session[email]": self.email,
            "user_session[password]": self.password,
            "user_session[remember_me]": "1",
        }
        if csrf:
            payload["authenticity_token"] = csrf
        
        logger.info("Submitting login...")
        resp = self.session.post(login_url, data=payload, allow_redirects=True)
        
        if "logout" in resp.text.lower() or "login" not in resp.url.lower():
            logger.info("✅ Login successful!")
            self.logged_in = True
            return True
        
        logger.error("❌ Login failed!")
        return False
    
    # =========================================================================
    # PDF Operations
    # =========================================================================
    
    def download_attachment(self, url: str, filename: str = None) -> Path:
        """Download attachment using authenticated session"""
        if not url.startswith('http'):
            url = self.base_url + url
        
        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code != 200:
                logger.warning(f"Download failed: HTTP {resp.status_code} for {filename}")
                return None
            
            # Check if redirected to login
            if "login" in resp.url.lower():
                logger.warning(f"Session expired for {filename}")
                return None
            
            # Generate filename if not provided
            if not filename:
                if 'Content-Disposition' in resp.headers:
                    cd = resp.headers['Content-Disposition']
                    match = re.search(r'filename="?([^"]+)"?', cd)
                    if match:
                        filename = match.group(1)
                
                if not filename:
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"attachment_{url_hash}.pdf"
            
            # Save file
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            safe_filename = re.sub(r'[^\w\-.]', '_', filename)
            filepath = self.pdf_dir / f"{url_hash}_{safe_filename}"
            
            with open(filepath, 'wb') as f:
                f.write(resp.content)
            
            self.stats["pdfs_downloaded"] += 1
            logger.info(f"  ✅ Downloaded: {filename} ({len(resp.content)} bytes)")
            return filepath
            
        except Exception as e:
            logger.warning(f"Download error for {filename}: {e}")
            return None
    
    def extract_text_from_pdf(self, filepath: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        
        if PDF_SUPPORT:
            try:
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                
                if text.strip():
                    self.stats["pdfs_extracted"] += 1
                    return text.strip()
            except Exception as e:
                logger.debug(f"pdfplumber error: {e}")
        
        if PYPDF2_SUPPORT:
            try:
                reader = PdfReader(filepath)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                if text.strip():
                    self.stats["pdfs_extracted"] += 1
                    return text.strip()
            except Exception as e:
                logger.debug(f"PyPDF2 error: {e}")
        
        return None
    
    def process_attachments(self, attachments: list) -> list:
        """Process and extract content from attachments"""
        processed = []
        
        for att in attachments:
            att_data = TicketAttachment(
                filename=att.get("filename", "unknown"),
                url=att.get("url", ""),
                file_type=att.get("type", "unknown")
            )
            
            # Only process PDFs
            if att_data.file_type.lower() == "pdf" or att_data.filename.lower().endswith('.pdf'):
                logger.info(f"  [*] Downloading PDF: {att_data.filename}")
                
                filepath = self.download_attachment(att_data.url, att_data.filename)
                
                if filepath:
                    logger.info(f"  [*] Extracting text...")
                    text = self.extract_text_from_pdf(filepath)
                    
                    if text:
                        att_data.content = text
                        att_data.local_path = str(filepath)
                        logger.info(f"  [+] Extracted {len(text)} characters")
                    else:
                        logger.warning(f"  [-] No text extracted (scanned PDF?)")
            
            processed.append(att_data)
        
        return processed
    
    # =========================================================================
    # Ticket Operations
    # =========================================================================
    
    def get_ticket_ids_from_page(self, page_num: int) -> list:
        """Get ticket IDs from listing page"""
        url = f"{self.base_url}/en/support/tickets?page={page_num}"
        
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                return []
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            ids = []
            
            for link in soup.find_all('a', href=re.compile(r'/tickets/(\d+)')):
                match = re.search(r'/tickets/(\d+)', link['href'])
                if match:
                    ids.append(int(match.group(1)))
            
            return list(set(ids))
        except:
            return []
    
    def collect_ticket_ids(
        self,
        start_page: int = 1,
        end_page: int = TICKET_MAX_PAGES,
        delay: float = TICKET_REQUEST_DELAY
    ) -> list:
        """Collect all ticket IDs from listing pages"""
        if not self.logged_in:
            logger.error("Not logged in!")
            return []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: Collecting Ticket IDs (pages {start_page}-{end_page})")
        logger.info(f"{'='*60}\n")
        
        all_ids = []
        total_pages = end_page - start_page + 1
        
        for i, page in enumerate(range(start_page, end_page + 1), 1):
            ids = self.get_ticket_ids_from_page(page)
            all_ids.extend(ids)
            self.stats["pages_scraped"] += 1
            
            if i % 20 == 0 or i == total_pages:
                logger.info(f"[{i}/{total_pages}] Found {len(all_ids)} tickets")
            
            time.sleep(delay)
        
        all_ids = sorted(list(set(all_ids)))
        self.stats["tickets_found"] = len(all_ids)
        
        self._save_json(self.data_dir / "ticket_ids.json", {
            "count": len(all_ids),
            "ids": all_ids,
            "scraped_at": datetime.now().isoformat()
        })
        
        logger.info(f"\n✅ Collected {len(all_ids)} unique ticket IDs")
        return all_ids
    
    def get_ticket_detail(self, ticket_id: int, download_pdfs: bool = True) -> TicketModel:
        """Get ticket details"""
        url = f"{self.base_url}/en/support/tickets/{ticket_id}"
        
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200 or "login" in resp.url.lower():
                return None
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            ticket = self._parse_ticket(soup, ticket_id, url)
            
            # Process PDFs
            if download_pdfs and ticket.attachments:
                logger.info(f"  [*] Processing {len(ticket.attachments)} attachments...")
                ticket.attachments = self.process_attachments(
                    [{"filename": a.filename, "url": a.url, "type": a.file_type} 
                     for a in ticket.attachments]
                )
                ticket.has_pdf_content = any(a.content for a in ticket.attachments)
            
            # Enrich with product detection
            self._enrich_ticket(ticket)
            
            return ticket
            
        except Exception as e:
            logger.warning(f"Error fetching ticket {ticket_id}: {e}")
            return None
    
    def _parse_ticket(self, soup, ticket_id: int, url: str) -> TicketModel:
        """Parse ticket HTML"""
        # Title
        heading = soup.find('h2', class_='heading')
        if heading:
            title = re.sub(r'^#\d+\s*', '', heading.get_text().strip())
        else:
            title = f"Ticket #{ticket_id}"
        
        # Description
        description = None
        desc_section = soup.find('section', id='ticket-description')
        if desc_section:
            description = self._parse_comment(desc_section)
        
        # Comments
        comments = []
        for section in soup.find_all('section', class_=re.compile(r'comment')):
            if section.get('id') == 'ticket-description':
                continue
            comment = self._parse_comment(section)
            if comment:
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
    
    def _parse_comment(self, section) -> TicketComment:
        """Parse comment section"""
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
    
    def _enrich_ticket(self, ticket: TicketModel):
        """Detect products and tags from ticket content"""
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
        
        # Auto-tag
        all_text_lower = all_text.lower()
        for tag, pattern in self.TAG_PATTERNS.items():
            if re.search(pattern, all_text_lower, re.IGNORECASE):
                ticket.tags.append(tag)
    
    def scrape_tickets(
        self,
        ticket_ids: list,
        download_pdfs: bool = TICKET_DOWNLOAD_PDFS,
        delay: float = TICKET_REQUEST_DELAY,
        checkpoint_every: int = TICKET_CHECKPOINT_EVERY
    ) -> list:
        """Scrape multiple tickets with checkpoint support"""
        if not self.logged_in:
            logger.error("Not logged in!")
            return []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2: Scraping Ticket Details ({len(ticket_ids)} tickets)")
        logger.info(f"PDF download: {'Enabled' if download_pdfs else 'Disabled'}")
        logger.info(f"{'='*60}\n")
        
        tickets = []
        checkpoint_file = self.data_dir / "checkpoint.json"
        start_idx = 0
        
        # Resume from checkpoint
        if checkpoint_file.exists():
            checkpoint = self._load_json(checkpoint_file)
            start_idx = checkpoint.get("last_index", 0) + 1
            # Load existing tickets
            for t in checkpoint.get("tickets", []):
                try:
                    tickets.append(TicketModel(**t))
                except:
                    pass
            logger.info(f"Resuming from checkpoint: {start_idx}/{len(ticket_ids)}")
        
        total = len(ticket_ids)
        
        for i, ticket_id in enumerate(ticket_ids[start_idx:], start_idx + 1):
            logger.info(f"[{i}/{total}] Ticket #{ticket_id}...")
            
            ticket = self.get_ticket_detail(ticket_id, download_pdfs)
            
            if ticket:
                tickets.append(ticket)
                self.stats["tickets_scraped"] += 1
                
                pdf_count = sum(1 for a in ticket.attachments if a.content)
                title_preview = ticket.title[:35] + "..." if len(ticket.title) > 35 else ticket.title
                logger.info(f"  ✓ {title_preview} ({pdf_count} PDFs)")
            else:
                self.stats["tickets_failed"] += 1
                logger.info(f"  ✗ Failed")
            
            # Checkpoint
            if i % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_file, tickets, i - 1)
                logger.info(f"  [checkpoint] {len(tickets)} tickets saved")
            
            time.sleep(delay)
        
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
    
    def _save_checkpoint(self, filepath: Path, tickets: list, last_index: int):
        """Save checkpoint"""
        self._save_json(filepath, {
            "last_index": last_index,
            "tickets": [t.to_dict() for t in tickets]
        })
    
    def _save_json(self, path: Path, data: dict):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_json(self, path: Path) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_ticket_ids(self, path: Path = None) -> list:
        """Load previously saved ticket IDs"""
        if path is None:
            path = self.data_dir / "ticket_ids.json"
        data = self._load_json(path)
        logger.info(f"Loaded {len(data.get('ids', []))} ticket IDs")
        return data.get("ids", [])
    
    def export_for_rag(self, tickets: list, filename: str = "tickets_rag.json") -> list:
        """Export tickets in RAG-ready format"""
        rag_docs = [t.to_rag_document() for t in tickets]
        
        filepath = self.data_dir / filename
        self._save_json(filepath, rag_docs)
        
        logger.info(f"✅ Exported {len(rag_docs)} RAG documents to {filepath}")
        return rag_docs
    
    def close(self):
        """Close the session"""
        if self.session:
            self.session.close()
            logger.info("Session closed")
