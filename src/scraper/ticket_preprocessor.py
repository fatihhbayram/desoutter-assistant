import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingStats:
    total_input: int = 0
    passed: int = 0
    filtered_bulletin: int = 0   # Service bulletin / quality notice
    filtered_feedback: int = 0   # Thank you / satisfaction only
    filtered_duplicate: int = 0
    filtered_low_quality: int = 0  # No useful technical content


def decode_cid_text(text: str) -> str:
    """Decode CID-encoded PDF text to readable text.
    
    CID format: (cid:XX) where XX is ASCII code
    Example: (cid:68)(cid:101)(cid:115) -> 'Des'
    """
    if not text or '(cid:' not in text:
        return text
    
    def replace_cid(match):
        try:
            char_code = int(match.group(1))
            return chr(char_code)
        except:
            return ''
    
    # Replace (cid:XX) with corresponding character
    decoded = re.sub(r'\(cid:(\d+)\)', replace_cid, text)
    # Clean up extra whitespace
    decoded = re.sub(r'\s+', ' ', decoded).strip()
    return decoded


class TicketPreprocessor:
    def __init__(self, min_question_length: int = 20, min_answer_length: int = 50):
        self.min_question_length = min_question_length
        self.min_answer_length = min_answer_length
        self.seen_hashes = set()
        self.stats = PreprocessingStats()
        # ...patterns and methods as in prompt...
        # Quality feedback patterns (no real question) - French/English support
        self.feedback_patterns = [
            r'(?i)^(thank|thanks|merci|satisfait|parfait|super|perfect|great|excellent)',
            r'(?i)(tres bien|very good|well done|good job|appreciate)',
            r'(?i)(probleme (regle|resolu)|no (problem|issue)|resolved|solved)',
            r"(?i)^(ok|okay|d'accord|got it|understood)",
            r'(?i)(happy|satisfied|content|reussi)',
            r'(?i)^(hi|hello|bonjour|salut)\s*$',
            r'(?i)(5 (star|etoile)|cinq etoiles|full marks)',
        ]
        self.bulletin_patterns = [
            r'(?i)(bulletin\s*qualite|quality\s*bulletin|service\s*bulletin)',
            r'(?i)(sb[-\s]?\d+|qb[-\s]?\d+|tsb[-\s]?\d+)',
            r'(?i)(bulletin.*selon|bulletin.*according|per\s*bulletin)',
            r'(?i)(piece\s*changee|part\s*replace|component\s*swap)',
            r'(?i)(rappel|recall|campagne|campaign)',
            r'(?i)(demande\s*garantie|warranty\s*claim)',
            r'(?i)(remplacement\s*effectue|replacement\s*done|swap\s*completed)',
            r'(?i)(action\s*effectuee|action\s*taken|work\s*performed)',
            r'(?i)(remplacement\s*gratuit|free\s*replacement|no\s*charge)',
            r'(?i)(mise\s*a\s*jour\s*terrain|field\s*update|field\s*action)',
        ]
        self.noise_patterns = [
            r'<[^>]+>',
            r'--+\s*Original Message\s*--+.*',
            r'(?i)sent from my (iphone|android|mobile).*',
            r'(?i)best regards.*',
            r'(?i)kind regards.*',
            r'(?i)thanks.*regards.*',
            r'(?i)sincerely.*',
            r'\[cid:[^\]]+\]',
            r'_{3,}',
            r'-{3,}',
            r'={3,}',
            r'(?i)disclaimer:.*',
            r'(?i)confidential.*notice.*',
            r'https?://\S+',
            r'\S+@\S+\.\S+',
            r'(?i)ticket\s*#?\s*\d+',
        ]
        self.product_keywords = [
            'cvi3', 'cvi', 'cvic', 'omega', 'delta', 'pf4000', 'pf6000',
            'controller', 'motor', 'spindle', 'tool', 'torque', 'angle',
            'screwdriver', 'nutrunner', 'calibration', 'error', 'fault',
            'desoutter', 'atlas copco', 'cleco'
        ]
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def extract_product(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        product_patterns = [
            (r'cvi3[-\s]?\w*', 'CVI3'),
            (r'cvic[-\s]?\w*', 'CVIC'),
            (r'pf[-\s]?4000', 'PF4000'),
            (r'pf[-\s]?6000', 'PF6000'),
            (r'omega[-\s]?\w*', 'Omega'),
            (r'delta[-\s]?\w*', 'Delta'),
        ]
        for pattern, product in product_patterns:
            if re.search(pattern, text_lower):
                return product
        return None

    def compute_hash(self, text: str) -> str:
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def is_duplicate(self, question: str, answer: str, title: str = '', pdf_content: str = '') -> bool:
        # Include title and pdf_content in hash for better uniqueness
        content_hash = self.compute_hash(f"{title}|{question}|{answer}|{pdf_content[:500]}")
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False

    def is_product_related(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.product_keywords)

    def calculate_quality_score(self, question: str, answer: str) -> float:
        score = 0.0
        q_len = len(question)
        a_len = len(answer)
        if q_len >= 30: score += 0.15
        if q_len >= 50: score += 0.10
        if a_len >= 100: score += 0.20
        if a_len >= 200: score += 0.15
        if self.is_product_related(question) or self.is_product_related(answer):
            score += 0.20
        technical_terms = ['error', 'code', 'calibrat', 'torque', 'setting', 'config', 
                          'install', 'connect', 'motor', 'controller', 'parameter']
        combined = (question + answer).lower()
        tech_count = sum(1 for term in technical_terms if term in combined)
        score += min(tech_count * 0.05, 0.20)
        return min(score, 1.0)

    def is_bulletin_action(self, text: str) -> bool:
        for pattern in self.bulletin_patterns:
            if re.search(pattern, text):
                troubleshoot_terms = ['error', 'hata', 'problem', 'sorun', 'how to', 'why', 'pourquoi', 'comment']
                if any(term in text.lower() for term in troubleshoot_terms):
                    return False
                return True
        return False

    def is_feedback_only(self, text: str) -> bool:
        if len(text.strip()) > 100:
            return False
        for pattern in self.feedback_patterns:
            if re.search(pattern, text):
                return True
        return False

    def has_useful_content(self, question: str, answer: str, title: str = '') -> bool:
        combined = (question + ' ' + answer + ' ' + title).lower()
        if len(combined.strip()) < 30:  # Reduced from 50
            return False
        useful_indicators = [
            'error', 'code', 'how', 'why', 'problem', 'issue', 'setting', 'config', 'parameter',
            'install', 'connect', 'calibrat', 'motor', 'controller', 'tool', 'torque',
            'panne', 'defaut', 'probleme', 'parametre', 'installer', 'connecter', 'calibrer', 'moteur', 'outil', 'couple',
            # Product-related
            'cvi', 'cvinet', 'e-lit', 'elit', 'eabc', 'efde', 'dss', 'dst', 'drs', 'erxs', 'epb', 'twincvi',
            # Fault/error patterns
            'fault', 'i0', 'e0', 'e1', 'e2', 'qcm', 'doa', 'pcb', 'failure', 'crash', 'jammed', 'broken',
            # French technical terms
            'panne', 'défaut', 'erreur', 'blocage'
        ]
        return any(term in combined for term in useful_indicators)

    def _extract_text(self, field) -> str:
        """Extract text from field (handles dict with 'content' key or plain string)"""
        if field is None:
            return ""
        if isinstance(field, dict):
            return field.get('content', '') or ""
        return str(field)

    def process_ticket(self, ticket: Dict) -> Optional[Dict]:
        self.stats.total_input += 1
        
        # Extract question from description/subject/title
        desc_text = self._extract_text(ticket.get('description'))
        subject_text = self._extract_text(ticket.get('subject'))
        title_text = self._extract_text(ticket.get('title'))
        question = self.clean_text(desc_text or subject_text or title_text)
        
        # Extract answer from resolution/agent_reply/conversation
        answer_parts = []
        
        # Resolution field
        resolution = ticket.get('resolution')
        if resolution:
            res_text = self._extract_text(resolution)
            if res_text:
                answer_parts.append(self.clean_text(res_text))
        
        # Agent replies from comments
        comments = ticket.get('comments', [])
        for comment in comments:
            if isinstance(comment, dict):
                if comment.get('is_agent') or 'agent' in str(comment.get('author', '')).lower():
                    answer_parts.append(self.clean_text(comment.get('content', '')))
            elif isinstance(comment, str):
                answer_parts.append(self.clean_text(comment))
        
        # PDF content from attachments
        pdf_content = []
        attachments = ticket.get('attachments', [])
        for att in attachments:
            if isinstance(att, dict) and att.get('content'):
                raw_content = att.get('content', '')
                # Decode CID-encoded PDF text first
                decoded_content = decode_cid_text(raw_content)
                pdf_text = self.clean_text(decoded_content)
                if pdf_text and len(pdf_text) > 100:  # Only meaningful PDF content
                    pdf_content.append(pdf_text)
        
        answer = '\n\n'.join(filter(None, answer_parts))
        pdf_combined = '\n\n'.join(pdf_content)
        
        # Get title for filtering
        title = ticket.get('title', '') or ticket.get('subject', '')
        
        # Combine all text for filtering
        combined_text = f"{title} {question} {answer} {pdf_combined}"
        
        if self.is_bulletin_action(combined_text):
            self.stats.filtered_bulletin += 1
            logger.debug(f"Filtered {ticket.get('ticket_id')}: bulletin action only")
            return None
        if self.is_feedback_only(question):
            self.stats.filtered_feedback += 1
            logger.debug(f"Filtered {ticket.get('ticket_id')}: feedback only")
            return None
        if not self.has_useful_content(question, answer + pdf_combined, title):
            self.stats.filtered_low_quality += 1
            logger.debug(f"Filtered {ticket.get('ticket_id')}: no useful content")
            return None
        if self.is_duplicate(question, answer, title, pdf_combined):
            self.stats.filtered_duplicate += 1
            logger.debug(f"Filtered {ticket.get('ticket_id')}: duplicate")
            return None
        self.stats.passed += 1
        quality = self.calculate_quality_score(question, answer + pdf_combined)
        return {
            'ticket_id': ticket.get('ticket_id'),
            'title': title,
            'question': question,
            'answer': answer,
            'pdf_content': pdf_combined if pdf_combined else None,
            'product': self.extract_product(combined_text),
            'related_products': ticket.get('related_products', []),
            'related_models': ticket.get('related_models', []),
            'tags': ticket.get('tags', []),
            'category': ticket.get('category', 'support'),
            'quality_score': quality,
            'created_at': ticket.get('created_at'),
            'source': 'freshdesk_ticket',
            'has_pdf': bool(pdf_combined)
        }

    def process_batch(self, tickets: List[Dict]) -> Tuple[List[Dict], PreprocessingStats]:
        processed = []
        for ticket in tickets:
            result = self.process_ticket(ticket)
            if result:
                processed.append(result)
        logger.info(f"Preprocessing complete: {self.stats.passed}/{self.stats.total_input} passed")
        logger.info(f"Filtered - Bulletin: {self.stats.filtered_bulletin}, "
                   f"Feedback: {self.stats.filtered_feedback}, "
                   f"Duplicate: {self.stats.filtered_duplicate}, "
                   f"Low quality: {self.stats.filtered_low_quality}")
        return processed, self.stats