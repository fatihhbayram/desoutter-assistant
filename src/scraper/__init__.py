"""Scraper module - Schema v2 Support"""
from .desoutter_scraper import DesoutterScraper
from .parsers import ProductParser
from .product_categorizer import (
    categorize_product,
    detect_tool_category,
    detect_wireless_info,
    detect_platform_connection,
    detect_modular_system,
    extract_product_family,
    detect_tool_type
)

__all__ = [
    'DesoutterScraper',
    'ProductParser',
    'categorize_product',
    'detect_tool_category',
    'detect_wireless_info',
    'detect_platform_connection',
    'detect_modular_system',
    'extract_product_family',
    'detect_tool_type'
]
