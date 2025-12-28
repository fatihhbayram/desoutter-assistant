"""
HTML parsing utilities for extracting product data
"""
import re
from urllib.parse import urlparse, parse_qs, unquote, urljoin
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProductParser:
    """Parser for product HTML pages"""
    
    @staticmethod
    def extract_part_numbers(html: str, patterns: List[str] = None) -> List[str]:
        """
        Extract part numbers from HTML
        
        Args:
            html: HTML content
            patterns: List of regex patterns for part numbers
            
        Returns:
            List of unique part numbers
        """
        if patterns is None:
            patterns = [
                r'\b(6151[0-9]{6})\b',  # 6151 series (CVI3)
                r'\b(6159[0-9]{6})\b',  # 6159 series (Axon, CVIR, CVIL, Connect, etc.)
                r'\b(8920[0-9]{6})\b',  # 8920 series
            ]
        
        found_numbers = set()
        for pattern in patterns:
            matches = re.findall(pattern, html)
            found_numbers.update(matches)
        
        return sorted(list(found_numbers))
    
    @staticmethod
    def extract_series_name(html: str) -> str:
        """
        Extract series name from series page
        
        Args:
            html: HTML content
            
        Returns:
            Series name
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Try h1 tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        # Try title tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        return "Unknown Series"
    
    @staticmethod
    def parse_product_details(html: str, part_number: str) -> Dict:
        """
        Parse product details from product page HTML
        
        Args:
            html: HTML content
            part_number: Product part number
            
        Returns:
            Dictionary of product details
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Store original HTML for regex searches
        html_content = html
        
        # Model name from h1
        model_name = part_number
        h1_tag = soup.find('h1')
        if h1_tag:
            model_name = h1_tag.get_text(strip=True)
        
        # Description from meta tag
        description = ""
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc['content'].strip()
        
        # Image URL - Try multiple strategies
        image_url = "-"
        
        # Strategy 1: og:image meta tags
        og_image = soup.find('meta', {'property': 'og:image'})
        if og_image and og_image.get('content'):
            image_url = og_image['content']
        else:
            # try secure og image
            og_image_secure = soup.find('meta', {'property': 'og:image:secure_url'})
            if og_image_secure and og_image_secure.get('content'):
                image_url = og_image_secure['content']
            else:
                # twitter card image
                tw_img = soup.find('meta', {'name': 'twitter:image'})
                if tw_img and tw_img.get('content'):
                    image_url = tw_img['content']
        
        # Strategy 2: If no meta tags found, search for datocms-assets images in HTML
        if image_url == "-":
            # Look for product images from datocms (common pattern in the HTML)
            datocms_pattern = r'(https://www\.datocms-assets\.com/104564/\d+-[^"\'?\s]+\.(?:jpg|jpeg|png|webp))'
            matches = re.findall(datocms_pattern, html_content)
            if matches:
                # Filter out logos and common images, prefer first match
                for match in matches:
                    # Skip known non-product images
                    if 'logo' not in match.lower() and '1747142541' not in match and '1699612149' not in match:
                        image_url = match
                        break
                # If all filtered out, use first match anyway
                if image_url == "-" and matches:
                    image_url = matches[0]

        # Determine base URL from page (og:url or canonical) to resolve relative URLs
        base_url = None
        og_url = soup.find('meta', {'property': 'og:url'})
        if og_url and og_url.get('content'):
            base_url = og_url['content']
        else:
            canon = soup.find('link', {'rel': 'canonical'})
            if canon and canon.get('href'):
                base_url = canon['href']
        if base_url:
            parsed = urlparse(base_url)
            base_root = f"{parsed.scheme}://{parsed.netloc}"
        else:
            # fallback domain often used in scraping target
            base_root = "https://www.desouttertools.com"

        def _normalize_url(u: str) -> str:
            if not u:
                return ""
            u = u.strip()
            # handle Next.js image optimizer: /_next/image?url=%2Fpath%2Fimg.jpg&w=640
            if u.startswith("/_next/image") or "_next/image" in u:
                # extract url param
                try:
                    parsed = urlparse(u)
                    qs = parse_qs(parsed.query)
                    url_param = qs.get('url') or qs.get('u')
                    if url_param:
                        decoded = unquote(url_param[0])
                        if decoded.startswith('//'):
                            return 'https:' + decoded
                        if decoded.startswith('/'):
                            return urljoin(base_root, decoded)
                        return decoded
                except Exception:
                    pass

            # protocol-relative
            if u.startswith('//'):
                return 'https:' + u

            # absolute
            if u.startswith('http'):
                return u

            # relative
            if u.startswith('/'):
                return urljoin(base_root, u)

            return urljoin(base_root + '/', u)

        def _pick_from_srcset(srcset: str) -> str:
            # parse candidates like: '/path/img.jpg 200w, /path/img@2x.jpg 400w'
            candidates = []
            for part in srcset.split(','):
                part = part.strip()
                if not part:
                    continue
                pieces = part.split()
                url_part = pieces[0]
                size = 0
                if len(pieces) > 1:
                    m = re.search(r'(\d+)w', pieces[1])
                    if m:
                        size = int(m.group(1))
                candidates.append((size, url_part))
            if not candidates:
                return ""
            # choose candidate with largest size (0 will be fallback)
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

        # Fallbacks: search for images in product description, handle lazy attrs and srcset
        try:
            desc_node = soup.select_one('#spy_Product_description')
            img_candidates = []
            if desc_node:
                # First, try the specific nested div the user provided (XPath -> CSS approximation)
                try_selectors = [
                    '#spy_Product_description div div div:nth-of-type(1) div div:nth-of-type(1) div',
                    '#spy_Product_description > div > div > div:nth-of-type(1) > div > div:nth-of-type(1) > div'
                ]
                for sel in try_selectors:
                    target = soup.select_one(sel)
                    if target:
                        # check style background-image
                        style = target.get('style', '')
                        m = re.search(r'background-image\s*:\s*url\(([^)]+)\)', style)
                        if m:
                            url_val = m.group(1).strip('"\'')
                            img_candidates.append(url_val)
                        # look for img inside the target
                        inner_img = target.find('img')
                        if inner_img:
                            for attr in ('src', 'data-src', 'data-original', 'data-lazy'):
                                val = inner_img.get(attr)
                                if val:
                                    img_candidates.append(val)
                            ss = inner_img.get('srcset') or inner_img.get('data-srcset')
                            if ss:
                                pick = _pick_from_srcset(ss)
                                if pick:
                                    img_candidates.append(pick)
                        # check common data- attributes on the target
                        for data_attr in ('data-src', 'data-image', 'data-bg', 'data-background', 'data-original'):
                            val = target.get(data_attr)
                            if val:
                                img_candidates.append(val)
                        # if we've found candidates from the specific selector, prefer them
                        if img_candidates:
                            break

                # Generic fallback: collect <img> tags in description as before
                if not img_candidates:
                    imgs = desc_node.find_all('img')
                    for img in imgs:
                        for attr in ('src', 'data-src', 'data-original', 'data-lazy'):
                            val = img.get(attr)
                            if val:
                                img_candidates.append(val)
                        # srcset handling
                        ss = img.get('srcset') or img.get('data-srcset')
                        if ss:
                            pick = _pick_from_srcset(ss)
                            if pick:
                                img_candidates.append(pick)

            # inline background-image
            if not img_candidates:
                for el in desc_node.find_all(True, attrs={'style': True}):
                    style = el['style']
                    m = re.search(r'background-image\s*:\s*url\(([^)]+)\)', style)
                    if m:
                        url_val = m.group(1).strip('"\'')
                        img_candidates.append(url_val)

            # normalize and pick first valid
            for cand in img_candidates:
                norm = _normalize_url(cand)
                if norm:
                    image_url = norm
                    break
        except Exception:
            # keep whatever og:image provided or '-' if none
            pass
        
        # Extract specifications from HTML text
        html_text = html
        
        # Torque
        min_torque, max_torque = ProductParser._extract_torque(html_text)
        
        # Speed/RPM
        speed = ProductParser._extract_speed(html_text)
        
        # Output drive
        output_drive = ProductParser._extract_output_drive(html_text)
        
        # Wireless communication
        wireless = ProductParser._extract_wireless(html_text)
        
        # Weight
        weight = ProductParser._extract_weight(html_text)
        
        return {
            "model_name": model_name,
            "description": description,
            "image_url": image_url,
            "min_torque": min_torque,
            "max_torque": max_torque,
            "speed": speed,
            "output_drive": output_drive,
            "wireless_communication": wireless,
            "weight": weight,
        }
    
    @staticmethod
    def _extract_torque(html_text: str) -> tuple:
        """Extract torque specifications"""
        min_torque = "-"
        max_torque = "-"
        
        # Strategy 1: Look for explicit range with "to" (most reliable)
        # Matches: "0.45 to 1.8 Nm", "0.45Nm to 1.8Nm", "-0.45 to 1.8 Nm"
        torque_to_match = re.search(
            r'(-?[0-9.]+)\s*(?:Nm)?\s+to\s+(-?[0-9.]+)\s*Nm',
            html_text,
            re.IGNORECASE
        )
        if torque_to_match:
            # Take absolute values to handle negative signs (e.g., "M20 -0.45Nm" → 0.45)
            min_val = abs(float(torque_to_match.group(1)))
            max_val = abs(float(torque_to_match.group(2)))
            min_torque = f"{min_val} Nm"
            max_torque = f"{max_val} Nm"
            return min_torque, max_torque
        
        # Strategy 2: Look for range with dash/hyphen ONLY if both sides have Nm
        # Matches: "1.5Nm-8Nm", "1.5 Nm - 8 Nm"
        # Avoids: "M20 -0.45Nm" (only one Nm), "5-25 Nm" (only one Nm)
        torque_dash_match = re.search(
            r'([0-9.]+)\s*Nm\s*[-–]\s*([0-9.]+)\s*Nm',
            html_text,
            re.IGNORECASE
        )
        if torque_dash_match:
            min_torque = f"{torque_dash_match.group(1)} Nm"
            max_torque = f"{torque_dash_match.group(2)} Nm"
            return min_torque, max_torque
        
        # Strategy 3: Single value with validation
        # Use negative lookbehind to exclude model codes (M20, M10, etc.)
        # Also use negative lookbehind for dash to avoid "5-25 Nm" → "25"
        # Matches: "8 Nm", "0.45 Nm"
        # Avoids: "M20", "-25 Nm" (from "5-25 Nm")
        single_match = re.search(
            r'(?<![A-Z-])([0-9.]+)\s*Nm',
            html_text,
            re.IGNORECASE
        )
        if single_match:
            try:
                torque_val = float(single_match.group(1))
                # Only accept if it looks like a reasonable torque value (0.01 to 500 Nm)
                if 0.01 <= torque_val <= 500:
                    max_torque = f"{single_match.group(1)} Nm"
            except ValueError:
                pass
        
        return min_torque, max_torque
    
    @staticmethod
    def _extract_speed(html_text: str) -> str:
        """Extract speed/RPM specifications"""
        # Pattern: "0 to 1800 rpm" or "800-1800 RPM"
        rpm_match = re.search(
            r'([0-9]+)\s*(?:to|-|–)\s*([0-9,]+)\s*(?:rpm|RPM)',
            html_text,
            re.IGNORECASE
        )
        if rpm_match:
            min_rpm = rpm_match.group(1)
            max_rpm = rpm_match.group(2).replace(',', '')
            return f"{min_rpm} - {max_rpm} RPM"
        
        return "-"
    
    @staticmethod
    def _extract_output_drive(html_text: str) -> str:
        """Extract output drive size"""
        # Pattern: "Sq 3/8" or "Hex 1/4" or "output 1/2"
        drive_match = re.search(
            r'(?:Sq|Square|Hex|output drive|output)\s+([0-9/\".\s]+)',
            html_text,
            re.IGNORECASE
        )
        if drive_match:
            return drive_match.group(1).strip()
        
        return "-"
    
    @staticmethod
    def _extract_wireless(html_text: str) -> str:
        """
        Extract wireless communication capability.
        
        Battery tools wireless detection based on model code:
        - EPBC, EABC, EABS, ELC, BLRTC, etc. → Wireless (C = Connected/Communication)
        - EPB, EPBA, EAB, BLRTA, etc. → Standalone battery, NOT wireless
        
        Cable tools: Check for actual wireless keywords in text
        """
        # Strategy 1: Check model code for wireless indicators
        # Wireless battery tool patterns (C suffix = Connected/Communication)
        wireless_model_patterns = [
            r'\bEPBC\b',      # Electric Pistol Battery Connected
            r'\bEABC\b',      # Electric Angle Battery Connected  
            r'\bEABS\b',      # Electric Angle Battery Straight (wireless)
            r'\bEIBS\b',      # Electric Inline Battery Straight (wireless)
            r'\bELC\b',       # Electric Lit Connected
            r'\bELS\b',       # Electric Lit Straight (wireless)
            r'\bBLRTC\b',     # Battery Low Reaction Torque Connected
            r'\bEPBCHT\b',    # Electric Pistol Battery Connected High Torque
            r'\bEABCHT\b',    # Electric Angle Battery Connected High Torque
            r'\bEABCHT\b',    # Electric Angle Battery Connected High Torque
        ]
        
        for pattern in wireless_model_patterns:
            if re.search(pattern, html_text, re.IGNORECASE):
                return "Yes"
        
        # Strategy 2: For cable tools, check for explicit wireless keywords
        # Only if we haven't matched a battery model code above
        wireless_keywords = [
            r'wi-?fi\s+(?:capable|enabled|module)',  # "WiFi capable", "WiFi enabled"
            r'bluetooth\s+(?:capable|enabled|module)',  # "Bluetooth capable"
            r'2\.4\s*GHz\s+wireless',  # "2.4 GHz wireless"
            r'wireless\s+communication',  # "wireless communication"
        ]
        
        for pattern in wireless_keywords:
            if re.search(pattern, html_text, re.IGNORECASE):
                return "Yes"
        
        return "No"
    
    @staticmethod
    def _extract_weight(html_text: str) -> str:
        """Extract product weight"""
        # Pattern: "1.2 kg" or "500 g" (but not frequency like "2.4 GHz")
        weight_match = re.search(
            r'(?:weight|Weight)\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(kg|g)',
            html_text,
            re.IGNORECASE
        )
        if weight_match:
            return f"{weight_match.group(1)} {weight_match.group(2)}"
        
        # Fallback: any number followed by kg/g (excluding Hz/GHz)
        fallback_match = re.search(
            r'([0-9]+(?:\.[0-9]+)?)\s*(kg|g)(?!hz)',
            html_text,
            re.IGNORECASE
        )
        if fallback_match:
            return f"{fallback_match.group(1)} {fallback_match.group(2)}"
        
        return "-"
