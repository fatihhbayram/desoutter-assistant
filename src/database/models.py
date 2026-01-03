"""
Data models for products - Enhanced Schema v2
Includes categorization, platform relationships, and wireless detection
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# =============================================================================
# SUB-MODELS FOR ENHANCED CATEGORIZATION
# =============================================================================

class WirelessInfo(BaseModel):
    """Wireless capability information (battery tools only)"""
    capable: bool = False
    detection_method: str = "not_applicable"  # "model_name_C" | "existing_field" | "standalone_text_found" | "not_applicable"
    compatible_platforms: List[str] = []       # ["CVI3", "Connect"]
    compatible_platform_ids: List[str] = []    # MongoDB ObjectId references (as strings)


class PlatformConnection(BaseModel):
    """Platform connection info (cable tools only)"""
    required: bool = True
    compatible_platforms: List[str] = []       # ["CVI3", "CVIR II", "ESP-C"]
    compatible_platform_ids: List[str] = []    # MongoDB ObjectId references


class ModularSystem(BaseModel):
    """Modular system info (drilling tools only)"""
    is_base_tool: bool = False                 # XPB-Modular, XPB-One
    is_attachment: bool = False                # Tightening Head, Drilling Head
    attachment_type: Optional[str] = None      # "tightening" | "drilling"
    compatible_base_tools: List[str] = []      # ["XPB-Modular", "XPB-One"]


# =============================================================================
# MAIN PRODUCT MODEL (BACKWARD COMPATIBLE)
# =============================================================================

class ProductModel(BaseModel):
    """Enhanced Product data model - Schema v2"""
    
    # === EXISTING FIELDS (BACKWARD COMPATIBLE - NO CHANGES) ===
    product_id: str = Field(..., description="Unique product ID (part number)")
    model_name: str = Field(..., description="Product model name")
    part_number: str = Field(..., description="Manufacturer part number")
    series_name: str = Field(default="", description="Product series name")
    category: str = Field(default="", description="Product category (legacy)")
    product_url: str = Field(..., description="Product page URL")
    image_url: str = Field(default="-", description="Product image URL")
    description: str = Field(default="-", description="Product description")
    
    # Technical Specifications (unchanged)
    min_torque: str = Field(default="-", description="Minimum torque")
    max_torque: str = Field(default="-", description="Maximum torque")
    speed: str = Field(default="-", description="Speed/RPM")
    output_drive: str = Field(default="-", description="Output drive size")
    wireless_communication: str = Field(default="No", description="Wireless capability (legacy Yes/No)")
    weight: str = Field(default="-", description="Product weight")
    
    # Metadata (unchanged)
    scraped_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = Field(default="active", description="Product status")
    
    # === NEW FIELDS (SCHEMA V2) ===
    
    # Enhanced categorization
    tool_category: str = Field(
        default="unknown",
        description="Tool category: battery_tightening | cable_tightening | electric_drilling | platform"
    )
    tool_type: Optional[str] = Field(
        default=None,
        description="Tool type: pistol | angle_head | inline | screwdriver | drill | fixtured"
    )
    product_family: str = Field(
        default="",
        description="Product family code extracted from part number (EPB, EAD, XPB, etc.)"
    )
    
    # Connection/compatibility info (conditional based on tool_category)
    wireless: Optional[WirelessInfo] = Field(
        default=None,
        description="Wireless capability details (battery_tightening only)"
    )
    platform_connection: Optional[PlatformConnection] = Field(
        default=None,
        description="Platform connection requirements (cable_tightening only)"
    )
    modular_system: Optional[ModularSystem] = Field(
        default=None,
        description="Modular system info (electric_drilling only)"
    )
    
    # Schema version tracking for migrations
    schema_version: int = Field(default=1, description="Schema version (1=legacy, 2=enhanced)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "6151659030",
                "model_name": "EPBC14-T4000-S6S4-T",
                "part_number": "6151659030",
                "series_name": "EPB - Transducerized Pistol Battery Tool",
                "category": "Battery Tightening Tools",
                "product_url": "https://www.desouttertools.com/en/products/6151659030",
                "image_url": "https://example.com/image.jpg",
                "description": "Transducerized battery pistol nutrunner with WiFi",
                "min_torque": "1.5 Nm",
                "max_torque": "8 Nm",
                "speed": "0 - 1800 RPM",
                "output_drive": "3/8\"",
                "wireless_communication": "Yes",
                "weight": "1.2 kg",
                "status": "active",
                "tool_category": "battery_tightening",
                "tool_type": "pistol",
                "product_family": "EPB",
                "wireless": {
                    "capable": True,
                    "detection_method": "model_name_C",
                    "compatible_platforms": ["CVI3", "Connect"],
                    "compatible_platform_ids": []
                },
                "schema_version": 2
            }
        }
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return self.model_dump(exclude_none=True)


# =============================================================================
# TICKET MODELS (FRESHDESK SUPPORT PORTAL)
# =============================================================================

class TicketComment(BaseModel):
    """Single comment/reply on a ticket"""
    author: str = Field(default="Unknown", description="Comment author name")
    content: str = Field(..., description="Comment text content")
    date: Optional[str] = Field(default=None, description="Comment date")
    is_agent: bool = Field(default=False, description="True if from support agent")


class TicketAttachment(BaseModel):
    """Attachment on a ticket (PDF, images, etc.)"""
    filename: str = Field(..., description="Original filename")
    url: str = Field(default="", description="Download URL")
    file_type: str = Field(default="unknown", description="File type (pdf, image, etc.)")
    content: Optional[str] = Field(default=None, description="Extracted text content (for PDFs)")
    local_path: Optional[str] = Field(default=None, description="Local file path after download")


class TicketModel(BaseModel):
    """Support ticket from Freshdesk portal"""
    
    # Core identification
    ticket_id: int = Field(..., description="Unique ticket ID from Freshdesk")
    title: str = Field(..., description="Ticket title/subject")
    url: str = Field(..., description="Ticket URL")
    
    # Problem description
    description: Optional[TicketComment] = Field(default=None, description="Original problem description")
    
    # Responses and solutions
    comments: List[TicketComment] = Field(default_factory=list, description="All replies/comments")
    
    # Attachments with extracted content
    attachments: List[TicketAttachment] = Field(default_factory=list, description="Attached files")
    
    # Metadata
    scraped_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_resolved: bool = Field(default=False, description="Has agent response")
    
    # Product association (auto-detected from content)
    related_products: List[str] = Field(default_factory=list, description="Detected product part numbers")
    related_models: List[str] = Field(default_factory=list, description="Detected model names")
    
    # RAG-specific
    tags: List[str] = Field(default_factory=list, description="Auto-generated tags for retrieval")
    has_pdf_content: bool = Field(default=False, description="Has extracted PDF text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": 12345,
                "title": "CVI3 calibration error E102",
                "url": "https://support.desouttertools.com/en/support/tickets/12345",
                "description": {
                    "author": "John Doe",
                    "content": "Getting E102 error during calibration...",
                    "date": "2025-01-02",
                    "is_agent": False
                },
                "comments": [
                    {
                        "author": "Support Agent",
                        "content": "Please follow these steps...",
                        "date": "2025-01-03",
                        "is_agent": True
                    }
                ],
                "attachments": [
                    {
                        "filename": "calibration_guide.pdf",
                        "url": "https://...",
                        "file_type": "pdf",
                        "content": "Extracted PDF text..."
                    }
                ],
                "is_resolved": True,
                "related_products": ["6151659030"],
                "related_models": ["CVI3"],
                "tags": ["calibration", "error_code", "CVI3"]
            }
        }
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return self.model_dump(exclude_none=True)
    
    def to_rag_document(self) -> dict:
        """Convert ticket to RAG-ready document format"""
        parts = []
        
        # Title
        parts.append(f"# Support Ticket #{self.ticket_id}: {self.title}")
        parts.append("")
        
        # Problem description
        if self.description:
            parts.append("## Problem")
            parts.append(f"**Reported by:** {self.description.author}")
            if self.description.date:
                parts.append(f"**Date:** {self.description.date}")
            parts.append("")
            parts.append(self.description.content)
            parts.append("")
        
        # Solutions/Responses
        agent_comments = [c for c in self.comments if c.is_agent]
        if agent_comments:
            parts.append("## Solution")
            for comment in agent_comments:
                parts.append(f"**{comment.author}** ({comment.date or 'N/A'}):")
                parts.append(comment.content)
                parts.append("")
        
        # PDF content
        pdf_attachments = [a for a in self.attachments if a.content]
        if pdf_attachments:
            parts.append("## Attached Documentation")
            for att in pdf_attachments:
                parts.append(f"### {att.filename}")
                parts.append(att.content[:5000])  # Limit PDF content
                parts.append("")
        
        return {
            "id": f"ticket_{self.ticket_id}",
            "content": "\n".join(parts),
            "metadata": {
                "type": "support_ticket",
                "ticket_id": self.ticket_id,
                "title": self.title,
                "url": self.url,
                "is_resolved": self.is_resolved,
                "has_pdf_content": self.has_pdf_content,
                "related_products": self.related_products,
                "related_models": self.related_models,
                "tags": self.tags,
                "source": "freshdesk"
            }
        }
