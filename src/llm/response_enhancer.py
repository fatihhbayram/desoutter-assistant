"""
Response Enhancer
Ensures bulletin information is included even when generic response is generated.

Phase 3 of RAG Performance Enhancement v2.0

This acts as a safety net to catch cases where:
1. A relevant bulletin was retrieved but not mentioned in LLM response
2. The LLM focused on generic troubleshooting instead of specific bulletin info
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class BulletinInfo:
    """Information about a found bulletin."""
    bulletin_id: str
    source: str
    summary: str
    is_mentioned_in_response: bool


class ResponseEnhancer:
    """
    Enhances LLM responses by checking for relevant bulletins
    and prepending known issue information if missed.
    """
    
    # Patterns to extract bulletin description
    SUMMARY_PATTERNS = [
        r'Description of the issue[:\s]+([^.]+\.)',
        r'Failure description[:\s]+([^.]+\.)',
        r'Description[:\s]+([^.]+\.)',
        r'Visible symptom[:\s]+([^.]+\.)',
        r'Cause of issue[:\s]+([^.]+\.)',
    ]
    
    def enhance_response(
        self,
        query: str,
        product: str,
        response: str,
        retrieved_docs: List[Dict]
    ) -> str:
        """
        Enhance response with bulletin information if available but missed.
        
        Logic:
        1. Check if any retrieved docs are bulletins
        2. If bulletin found but not mentioned in response, prepend it
        3. If no bulletin in retrieval, return response unchanged
        
        Args:
            query: Original user query
            product: Product model name
            response: Generated LLM response
            retrieved_docs: List of retrieved documents
            
        Returns:
            Enhanced response with bulletin info if applicable
        """
        # Find bulletins in retrieved docs
        bulletins = self._find_bulletins(retrieved_docs)
        
        if not bulletins:
            # No bulletins retrieved - nothing to enhance
            return response
        
        # Check which bulletins are already referenced in response
        unreferenced_bulletins = []
        
        for bulletin in bulletins:
            if not self._is_bulletin_mentioned(bulletin.bulletin_id, response):
                unreferenced_bulletins.append(bulletin)
        
        if not unreferenced_bulletins:
            # All bulletins are already mentioned
            return response
        
        # Prepend bulletin information for unreferenced bulletins
        bulletin_section = self._format_bulletin_section(unreferenced_bulletins[:2])  # Max 2
        
        return f"{bulletin_section}\n\n---\n\n{response}"
    
    def _find_bulletins(self, docs: List[Dict]) -> List[BulletinInfo]:
        """
        Find bulletins in retrieved documents.
        
        Returns:
            List of BulletinInfo with extracted details
        """
        bulletins = []
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '').upper()
            doc_type = metadata.get('doc_type', '')
            text = doc.get('text', '')
            
            # Check if this is a service bulletin
            is_bulletin = (
                source.startswith('ESDE') or
                source.startswith('ESB') or
                doc_type == 'service_bulletin' or
                doc.get('is_bulletin', False) or
                doc.get('is_bulletin_match', False)
            )
            
            if not is_bulletin:
                continue
            
            # Extract bulletin ID
            bulletin_id = self._extract_bulletin_id(source)
            if not bulletin_id:
                continue
            
            # Extract summary
            summary = self._extract_summary(text)
            
            bulletins.append(BulletinInfo(
                bulletin_id=bulletin_id,
                source=metadata.get('source', source),
                summary=summary,
                is_mentioned_in_response=False  # Will be checked later
            ))
        
        return bulletins
    
    def _extract_bulletin_id(self, source: str) -> Optional[str]:
        """Extract bulletin ID from source filename."""
        # Match patterns like ESDE22012, ESDE23007, ESB12345
        match = re.search(r'(ESD[EBe][\d-]+)', source, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Try simpler pattern
        match = re.search(r'(ESDE\d+)', source, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None
    
    def _extract_summary(self, text: str) -> str:
        """Extract key summary from bulletin text."""
        text_lower = text.lower()
        
        for pattern in self.SUMMARY_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                # Capitalize first letter
                return summary[0].upper() + summary[1:] if summary else summary
        
        # Fallback: first 150 chars
        if len(text) > 150:
            return text[:150].strip() + "..."
        return text.strip()
    
    def _is_bulletin_mentioned(self, bulletin_id: str, response: str) -> bool:
        """Check if bulletin ID is mentioned in response."""
        response_upper = response.upper()
        
        # Check for exact match
        if bulletin_id in response_upper:
            return True
        
        # Check for partial match (e.g., "ESDE22012" should match "22012")
        id_numbers = re.findall(r'\d+', bulletin_id)
        for num in id_numbers:
            if len(num) >= 4 and num in response_upper:
                return True
        
        return False
    
    def _format_bulletin_section(self, bulletins: List[BulletinInfo]) -> str:
        """Format bulletin references as response section."""
        sections = []
        
        for bulletin in bulletins:
            section = f"""⚠️ **KNOWN ISSUE: {bulletin.bulletin_id}**

{bulletin.summary}

*See bulletin {bulletin.bulletin_id} for complete resolution steps.*"""
            sections.append(section)
        
        return "\n\n".join(sections)


def get_response_enhancer() -> ResponseEnhancer:
    """Factory function to create response enhancer."""
    return ResponseEnhancer()
