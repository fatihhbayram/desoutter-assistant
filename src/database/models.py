"""
Data models for products
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class ProductModel(BaseModel):
    """Product data model"""
    
    product_id: str = Field(..., description="Unique product ID (part number)")
    model_name: str = Field(..., description="Product model name")
    part_number: str = Field(..., description="Manufacturer part number")
    series_name: str = Field(default="", description="Product series name")
    category: str = Field(default="", description="Product category")
    product_url: str = Field(..., description="Product page URL")
    image_url: str = Field(default="-", description="Product image URL")
    description: str = Field(default="-", description="Product description")
    
    # Specifications
    min_torque: str = Field(default="-", description="Minimum torque")
    max_torque: str = Field(default="-", description="Maximum torque")
    speed: str = Field(default="-", description="Speed/RPM")
    output_drive: str = Field(default="-", description="Output drive size")
    wireless_communication: str = Field(default="No", description="Wireless capability")
    weight: str = Field(default="-", description="Product weight")
    
    # Metadata
    scraped_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = Field(default="active", description="Product status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "6151659030",
                "model_name": "EPB 8-1800-10S",
                "part_number": "6151659030",
                "series_name": "EPB - Transducerized Pistol Battery Tool",
                "category": "Battery Tightening Tools",
                "product_url": "https://www.desouttertools.com/en/products/6151659030",
                "image_url": "https://example.com/image.jpg",
                "description": "Transducerized battery pistol nutrunner",
                "min_torque": "1.5 Nm",
                "max_torque": "8 Nm",
                "speed": "0 - 1800 RPM",
                "output_drive": "3/8\"",
                "wireless_communication": "Yes",
                "weight": "1.2 kg",
                "status": "active"
            }
        }
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return self.model_dump(exclude_none=True)
