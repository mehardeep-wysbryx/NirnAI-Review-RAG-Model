"""
Utility functions for NirnAI RAG Review.
Includes normalization helpers, fingerprint builders, and extraction functions.
Updated to handle actual NirnAI merged case JSON format.
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def normalize_name(name: Optional[str]) -> str:
    """
    Normalize a person's name for comparison.
    - Lowercase
    - Remove extra whitespace
    - Standardize common prefixes (S/O, D/O, W/O, etc.)
    """
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)
    
    # Standardize relationship prefixes
    replacements = {
        r'\bs/o\b': 'son of',
        r'\bd/o\b': 'daughter of',
        r'\bw/o\b': 'wife of',
        r'\bh/o\b': 'husband of',
        r'\bc/o\b': 'care of',
    }
    
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    
    return name


def normalize_doc_no(doc_no: Optional[str]) -> str:
    """
    Normalize document number for comparison.
    - Remove leading zeros
    - Standardize separator (/ or -)
    - Extract numeric parts
    """
    if not doc_no:
        return ""
    
    doc_no = str(doc_no).strip()
    
    # Handle formats like "39 of 2026" or "39/2026"
    match = re.search(r'(\d+)\s*(?:of|[/\-])\s*(\d+)', doc_no, re.IGNORECASE)
    if match:
        num, year = match.groups()
        num = str(int(num))
        year = str(int(year))
        return f"{num}/{year}"
    
    # Just return cleaned version if no pattern match
    return re.sub(r'\s+', '', doc_no)


def normalize_extent(extent: Optional[str], unit: Optional[str] = None) -> Tuple[float, str]:
    """
    Normalize extent/area values.
    Returns (numeric_value, normalized_unit).
    """
    if not extent:
        return (0.0, "")
    
    extent_str = str(extent).lower().strip()
    
    # Extract numeric value
    numeric_match = re.search(r'([\d.]+)', extent_str)
    if not numeric_match:
        return (0.0, unit or "")
    
    value = float(numeric_match.group(1))
    
    # Detect unit from string if not provided
    if not unit:
        unit_patterns = {
            r'sq\.?\s*yds?\.?|square\s*yards?': 'sq.yds',
            r'sq\.?\s*ft\.?|square\s*feet|sqft': 'sq.ft',
            r'sq\.?\s*m\.?|square\s*met': 'sq.m',
            r'cents?': 'cents',
            r'acres?': 'acres',
            r'guntas?': 'guntas',
        }
        
        for pattern, normalized in unit_patterns.items():
            if re.search(pattern, extent_str, re.IGNORECASE):
                unit = normalized
                break
    
    return (value, unit or "")


def normalize_survey_no(survey_no: Optional[str]) -> str:
    """
    Normalize survey number for comparison.
    - Remove prefixes like 'Survey No.', 'Sy. No.'
    - Standardize separators
    """
    if not survey_no:
        return ""
    
    survey_no = str(survey_no).strip()
    
    # Remove common prefixes
    prefixes = [
        r'survey\s*no\.?\s*:?\s*',
        r'sy\.?\s*no\.?\s*:?\s*',
        r's\.?\s*no\.?\s*:?\s*',
    ]
    
    for prefix in prefixes:
        survey_no = re.sub(prefix, '', survey_no, flags=re.IGNORECASE)
    
    return survey_no.strip()


def normalize_date(date_str: Optional[str]) -> str:
    """
    Normalize date string to DD-MM-YYYY format.
    """
    if not date_str:
        return ""
    
    date_str = str(date_str).strip()
    
    # Handle formats like "06/Jan/2026" or "06-01-2026"
    month_names = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    # Try DD/Mon/YYYY format
    match = re.match(r'(\d{1,2})[/\-]([A-Za-z]{3})[/\-](\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        month_num = month_names.get(month.lower(), month)
        return f"{day.zfill(2)}-{month_num}-{year}"
    
    # Try common formats
    patterns = [
        # DD/MM/YYYY or DD-MM-YYYY
        (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', r'\1-\2-\3'),
        # YYYY-MM-DD
        (r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', r'\3-\2-\1'),
    ]
    
    for pattern, replacement in patterns:
        match = re.match(pattern, date_str)
        if match:
            return re.sub(pattern, replacement, date_str)
    
    return date_str


# =============================================================================
# EXTRACTION FROM ACTUAL NIRNAI FORMAT
# =============================================================================

def safe_get(data: Dict, *keys, default: Any = None) -> Any:
    """Safely navigate nested dictionary."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key] if len(current) > key else default
        else:
            return default
    return current if current is not None else default


def _filter_stamp_paper_noise(text: str) -> str:
    """
    Filter out stamp paper noise from OCR text.
    Removes common stamp paper patterns that aren't relevant to deed content.
    """
    # Patterns that indicate stamp paper content (not deed content)
    stamp_noise_patterns = [
        r'(?i)twenty\s*rupees?',
        r'(?i)hundred\s*rupees?',
        r'(?i)fifty\s*rupees?',
        r'(?i)thousand\s*rupees?',
        r'(?i)india\s*non\s*judicial',
        r'(?i)non\s*judicial\s*stamp',
        r'(?i)stamp\s*s\.?\s*no\.?\s*[:\s]*\d+[a-z]*\s*\d+',
        r'(?i)denomination[:\s]*rs\.?\s*\d+',
        r'(?i)purchased\s*by',
        r'(?i)for\s*whom',
        r'(?i)satyameva?\s*jayate?',
        r'(?i)सत्यमेव\s*जयते',
        r'(?i)भारत\s*सरकार',
        r'(?i)government\s*of\s*india',
        r'(?i)PEES?\s*OPER',
        r'(?i)WEN\s*EN',
        r'(?i)\d+/\d+\s*Rs\.',
        r'(?i)रू\.\d+',
        r'(?i)बीस\s*रूप',
        r'(?i)भारतीय',
        r'(?i)ग्रीयायिक',
    ]
    
    filtered_text = text
    for pattern in stamp_noise_patterns:
        filtered_text = re.sub(pattern, ' ', filtered_text)
    
    # Remove lines that are just numbers/noise
    lines = filtered_text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that are mostly numbers or very short
        if len(line.strip()) < 3:
            continue
        # Skip lines that are just repeating patterns
        if re.match(r'^[\d\s\.\-/]+$', line.strip()):
            continue
        # Skip lines with excessive special characters
        if len(re.findall(r'[a-zA-Z]', line)) < len(line) * 0.3 and len(line) > 10:
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _extract_deed_content_section(text: str) -> str:
    """
    Extract the actual deed content section from OCR text.
    Looks for deed-specific markers and content areas.
    """
    # Try to find the start of actual deed content
    deed_start_patterns = [
        r'(?i)deed\s*of\s*(?:gift|sale|donation|partition|settlement)',
        r'(?i)gift\s*settlement\s*deed',
        r'(?i)sale\s*deed',
        r'(?i)signed\s*by[:\s]*',
        r'(?i)schedule[:\s]*',
        r'(?i)property\s*(?:details|description)',
    ]
    
    earliest_start = len(text)
    for pattern in deed_start_patterns:
        match = re.search(pattern, text)
        if match and match.start() < earliest_start:
            earliest_start = match.start()
    
    if earliest_start < len(text):
        # Get content from deed start, but include some context before
        start = max(0, earliest_start - 100)
        return text[start:]
    
    return text


def _extract_schedule_section(text: str) -> str:
    """
    Extract the Schedule section from deed text.
    The Schedule section contains the actual property description (survey no, house no, extent, boundaries).
    This is different from party addresses which appear elsewhere.
    """
    # Look for Schedule section markers
    schedule_start_patterns = [
        r'(?i)schedule\s*(?:of\s*property)?[:\s]*',
        r'(?i)property\s*schedule[:\s]*',
        r'(?i)scheduled\s*property[:\s]*',
        r'(?i)the\s*scheduled\s*property',
        r'(?i)description\s*of\s*(?:the\s*)?property',
        r'(?i)property\s*description',
        r'(?i)situated\s*(?:at|in)',
        r'(?i)comprised\s*in\s*survey',
        r'(?i)bearing\s*(?:house\s*)?number',
        r'(?i)admeasuring\s*(?:an\s*)?extent',
    ]
    
    # Look for end of schedule section markers
    schedule_end_patterns = [
        r'(?i)witnesses?[:\s]*',
        r'(?i)annexure',
        r'(?i)declaration',
        r'(?i)stamp\s*duty',
        r'(?i)registration\s*fee',
        r'(?i)this\s*(?:is\s*the\s*)?settlement\s*(?:deed|document)',
        r'(?i)signed\s*(?:and\s*)?sealed',
    ]
    
    # Find the start of schedule section
    schedule_start = 0
    for pattern in schedule_start_patterns:
        match = re.search(pattern, text)
        if match:
            schedule_start = match.start()
            break
    
    # Find the end of schedule section
    schedule_end = len(text)
    remaining_text = text[schedule_start:]
    for pattern in schedule_end_patterns:
        match = re.search(pattern, remaining_text)
        if match and match.start() > 50:  # Must be at least 50 chars after start
            schedule_end = schedule_start + match.start()
            break
    
    # Extract schedule section (limit to reasonable size)
    schedule_text = text[schedule_start:min(schedule_end, schedule_start + 2000)]
    
    return schedule_text if schedule_text else text[:2000]


def extract_from_attachments(attachments: List[str]) -> Dict:
    """
    Extract key fields from attachments (OCR'd document text).
    This parses the raw text to find key information.
    IMPROVED: Filters stamp paper noise and extracts deed content.
    """
    extracted = {
        "raw_text": "",
        "cleaned_text": "",
        "doc_no": None,
        "deed_type": None,
        "executant": None,
        "claimant": None,
        "extent": None,
        "survey_no": None,
        "house_no": None,
        "village": None,
        "mandal": None,
        "district": None,
        "state": None,
        "boundaries": {},
        "registration_date": None,
        "execution_date": None,
        "sro": None,
        "market_value": None,
        "consideration_value": None,
    }
    
    if not attachments:
        return extracted
    
    # Combine all attachment text
    full_text = "\n".join(attachments) if isinstance(attachments, list) else str(attachments)
    extracted["raw_text"] = full_text[:5000]  # Limit for token control
    
    # Filter stamp paper noise and extract deed content
    filtered_text = _filter_stamp_paper_noise(full_text)
    deed_content = _extract_deed_content_section(filtered_text)
    extracted["cleaned_text"] = deed_content[:4000]
    
    # Extract document number - try multiple patterns
    doc_patterns = [
        # "Doct No/Year: 1101/2026" pattern
        r'(?:Doct?\s*No[/\s]*Year|Doc\.?\s*No\.?)[:\s]*(\d+)[/\s]*(?:of\s*)?(\d{4})',
        # "CS No/Year: 1116/2026" pattern
        r'CS\s*No[/\s]*Year[:\s]*(\d+)[/\s]*(\d{4})',
        # "registered as document No. 1101 of 2026"
        r'document\s*No\.?\s*(\d+)\s*(?:of|/)\s*(\d{4})',
        # General pattern
        r'(?:DOC\.?\s*NO\.?|Doc\s*No)[:\s]*(\d+)[/\s]*(?:of\s*)?(\d{4})',
    ]
    
    for pattern in doc_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            num, year = match.groups()
            extracted["doc_no"] = f"{int(num)}/{year}"
            break
    
    # Extract dates - execution vs registration
    # Execution date pattern (from deed text)
    exec_date_match = re.search(r'dated?\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s*)?([A-Za-z]+),?\s*(\d{4})', full_text, re.IGNORECASE)
    if exec_date_match:
        day, month, year = exec_date_match.groups()
        extracted["execution_date"] = f"{day}-{month[:3]}-{year}"
    
    # Also try DD-MM-YYYY format
    if not extracted["execution_date"]:
        exec_date_match = re.search(r'Date[:\s]*(\d{2}[-/]\d{2}[-/]\d{4})', full_text)
        if exec_date_match:
            extracted["execution_date"] = exec_date_match.group(1).replace('/', '-')
    
    # Registration date (usually in endorsement)
    reg_date_match = re.search(r'(?:registered\s*on|Presentation\s*Endorsement)[^\d]*(\d{1,2})(?:st|nd|rd|th)?\s*(?:day\s*of\s*)?([A-Za-z]+),?\s*(\d{4})', full_text, re.IGNORECASE)
    if reg_date_match:
        day, month, year = reg_date_match.groups()
        extracted["registration_date"] = f"{day}-{month[:3]}-{year}"
    
    # Extract deed type
    deed_patterns = [
        r'(?i)(deed\s*of\s*(?:gift|donation)\s*of\s*immovable\s*property)',
        r'(?i)(gift\s*settlement\s*deed)',
        r'(?i)(settlement\s*deed)',
        r'(?i)(sale\s*deed)',
        r'(?i)(partition\s*deed)',
        r'(?i)(release\s*deed)',
        r'(?i)(mortgage\s*deed)',
    ]
    for pattern in deed_patterns:
        match = re.search(pattern, full_text)
        if match:
            extracted["deed_type"] = match.group(1).strip()
            break
    
    # Extract value/consideration from deed text
    value_match = re.search(r'(?:valued\s*at|worth|market\s*value)[:\s]*Rs\.?\s*([\d,]+)', full_text, re.IGNORECASE)
    if value_match:
        extracted["market_value"] = value_match.group(1).replace(',', '')
    
    # Extract names (executant/claimant patterns)
    # Look for "Signed by" patterns in deed
    signed_by_matches = re.findall(r'Signed\s*by[:\-\s]*([A-Za-z\s]+?)(?:,|Age)', full_text, re.IGNORECASE)
    if len(signed_by_matches) >= 1:
        extracted["executant"] = signed_by_matches[0].strip()
    if len(signed_by_matches) >= 2:
        extracted["claimant"] = signed_by_matches[1].strip()
    
    # Also look for "(DE)" for executant and "(DR)" for claimant in registration text
    if not extracted["executant"]:
        de_match = re.search(r'\(DE\)\s*([A-Za-z\s]+?)(?:\(|$|\n)', full_text)
        if de_match:
            extracted["executant"] = de_match.group(1).strip()
    
    if not extracted["claimant"]:
        dr_match = re.search(r'\(DR\)\s*([A-Za-z\s]+?)(?:\(|$|\n)', full_text)
        if dr_match:
            extracted["claimant"] = dr_match.group(1).strip()
    
    # Extract survey number - look in deed content area
    survey_patterns = [
        r'(?:Survey\s*(?:No\.?|Number)?|Sy\.?\s*No\.?|S\.?\s*No\.?)[:\s]*(\d+(?:[/\-]\d+)?)',
        r'survey\s*number\s*(\d+)',
        r'comprised\s*in\s*survey\s*(?:number\s*)?(\d+)',
    ]
    for pattern in survey_patterns:
        match = re.search(pattern, deed_content, re.IGNORECASE)
        if match:
            extracted["survey_no"] = match.group(1)
            break
    
    # IMPROVED: Extract house/door number from SCHEDULE section only (not party addresses)
    # First, try to find the Schedule section
    schedule_section = _extract_schedule_section(deed_content)
    
    # Extract house/door number from Schedule section (property address, not party address)
    house_patterns = [
        # "bearing house number 5-87" pattern (most reliable - describes the property)
        r'bearing\s*(?:house\s*)?(?:number|no\.?)\s*(\d+[-/]?\d*)',
        # "Door No.5-87" or "House No. 5-87" in schedule context
        r'(?:Door\.?\s*No\.?|House\.?\s*No\.?|D\.?\s*No\.?)[:\s]*(\d+[-/]\d+)',
        # Just number with hyphen like "5-87" after schedule marker
        r'no\.?\s*(\d+[-/]\d+)',
    ]
    
    # First try to find in schedule section
    for pattern in house_patterns:
        match = re.search(pattern, schedule_section, re.IGNORECASE)
        if match:
            extracted["house_no"] = match.group(1)
            break
    
    # If not found in schedule, try looking for "bearing" pattern in full deed
    if not extracted["house_no"]:
        bearing_match = re.search(r'bearing\s*(?:house\s*)?(?:number|no\.?)\s*(\d+[-/]?\d*)', deed_content, re.IGNORECASE)
        if bearing_match:
            extracted["house_no"] = bearing_match.group(1)
    
    # IMPROVED: Extract extent from SCHEDULE section only (not random OCR text)
    # Priority: Schedule section > "admeasuring" phrase > "area is" phrase
    extent_patterns_schedule = [
        # "admeasuring an extent of 145 Sq. Yds" - most reliable
        r'admeasuring\s*(?:an\s*)?(?:extent\s*(?:of\s*)?)?(\d+\.?\d*)\s*(?:Sq\.?\s*(?:Yds?|Yards?)\.?)',
        r'admeasuring\s*(?:an\s*)?(?:extent\s*(?:of\s*)?)?(\d+\.?\d*)\s*(?:Sq\.?\s*(?:Ft|Feet)\.?)',
        r'admeasuring\s*(?:an\s*)?(?:extent\s*(?:of\s*)?)?(\d+\.?\d*)\s*(?:Sq\.?\s*M\.?)',
        # "extent of 145 sq.yds" in schedule
        r'extent\s*(?:of\s*)?(\d+\.?\d*)\s*(?:Sq\.?\s*(?:Yds?|Yards?))',
        r'extent\s*(?:of\s*)?(\d+\.?\d*)\s*(?:Sq\.?\s*(?:Ft|Feet))',
        # "residential area of 145 sq.m" pattern
        r'(?:residential\s*)?area\s*(?:of\s*)?(\d+\.?\d*)\s*(?:Sq\.?\s*(?:Yds?|M|Ft))',
    ]
    
    # Try schedule section first
    for pattern in extent_patterns_schedule:
        match = re.search(pattern, schedule_section, re.IGNORECASE)
        if match:
            extracted["extent"] = match.group(0)
            break
    
    # If not found in schedule, try the "admeasuring" pattern in full deed
    if not extracted["extent"]:
        for pattern in extent_patterns_schedule[:3]:  # Only "admeasuring" patterns
            match = re.search(pattern, deed_content, re.IGNORECASE)
            if match:
                extracted["extent"] = match.group(0)
                break
    
    # Extract location from deed content
    village_patterns = [
        r'(?:Village|Vill)[:\s]*([A-Za-z\s]+?)(?:,|\n|Mandal|District|Panchayat)',
        r'situated\s*(?:at|in)\s*([A-Za-z\s]+?)(?:\s*Village|\s*Panchayat)',
    ]
    for pattern in village_patterns:
        match = re.search(pattern, deed_content, re.IGNORECASE)
        if match:
            extracted["village"] = match.group(1).strip()
            break
    
    mandal_match = re.search(r'Mandal[:\s]*([A-Za-z\s]+?)(?:,|\n|District)', deed_content, re.IGNORECASE)
    if mandal_match:
        extracted["mandal"] = mandal_match.group(1).strip()
    
    district_match = re.search(r'(?:District|Dist\.?)[:\s]*([A-Za-z\s]+?)(?:,|\n|State|registered|\.|$)', deed_content, re.IGNORECASE)
    if district_match:
        extracted["district"] = district_match.group(1).strip()
    
    # Extract boundaries from DEED CONTENT (not stamp paper)
    # Look for boundary section specifically
    boundary_section = re.search(r'(?:bound(?:aries|ed)|between\s*this)[:\s]*(.*?)(?:this\s*area|between\s*this|The\s*dimensions|\n\n)', 
                                  deed_content, re.IGNORECASE | re.DOTALL)
    
    boundary_text = boundary_section.group(1) if boundary_section else deed_content
    
    # Extract individual boundaries with stricter patterns
    boundary_patterns = {
        'north': r'(?:North|N(?:orth)?)[:\s]*([A-Za-z][A-Za-z\s\']+?(?:house|road|land|property|nayak|plot)[A-Za-z\s\']*?)(?:,|South|East|West|$)',
        'south': r'(?:South|S(?:outh)?)[:\s]*([A-Za-z][A-Za-z\s\']+?(?:house|road|land|property|nayak|plot)[A-Za-z\s\']*?)(?:,|North|East|West|$)',
        'east': r'(?:East|E(?:ast)?)[:\s]*([A-Za-z][A-Za-z\s\']+?(?:house|road|land|property|nayak|plot)[A-Za-z\s\']*?)(?:,|North|South|West|$)',
        'west': r'(?:West|W(?:est)?)[:\s]*([A-Za-z][A-Za-z\s\']+?(?:house|road|land|property|nayak|plot)[A-Za-z\s\']*?)(?:,|North|South|East|$)',
    }
    
    for direction, pattern in boundary_patterns.items():
        match = re.search(pattern, boundary_text, re.IGNORECASE)
        if match:
            boundary_val = match.group(1).strip()
            # Validate it's not stamp paper noise
            if not re.search(r'(?i)rupee|judicial|stamp|india|twenty|hundred', boundary_val):
                extracted["boundaries"][direction] = boundary_val
    
    # Fallback: simpler boundary extraction from deed content
    if not extracted["boundaries"]:
        simple_patterns = {
            'north': r'\[N\][:\s]*([^\[\]]+?)(?:\[|\n|$)',
            'south': r'\[S\][:\s]*([^\[\]]+?)(?:\[|\n|$)',
            'east': r'\[E\][:\s]*([^\[\]]+?)(?:\[|\n|$)',
            'west': r'\[W\][:\s]*([^\[\]]+?)(?:\[|\n|$)',
        }
        for direction, pattern in simple_patterns.items():
            match = re.search(pattern, full_text)
            if match:
                boundary_val = match.group(1).strip()
                if not re.search(r'(?i)rupee|judicial|stamp', boundary_val):
                    extracted["boundaries"][direction] = boundary_val
    
    # Extract SRO
    sro_patterns = [
        r'Sub-?Registrar[,\s]*([A-Z\s]+?)(?:\(|\n|along)',
        r'SRO[:\s]*([A-Za-z\s]+?)(?:\(|\n|,)',
        r'registered\s*(?:at|before)\s*(?:the\s*)?(?:Sub-?Registrar|SRO)[,\s]*([A-Za-z\s]+)',
    ]
    for pattern in sro_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted["sro"] = match.group(1).strip()
            break
    
    return extracted


def extract_from_encumbrance_details(ec_details: List[Dict]) -> Dict:
    """
    Extract key fields from encumbranceDetails array.
    Handles the actual NirnAI EC format.
    IMPROVED: Better document number extraction and market value parsing.
    """
    extracted = {
        "transactions": [],
        "mortgage_flag": False,
        "property_description": None,
        "sro": None,
        "market_value": None,
        "consideration_value": None,
        "boundaries": {},
        "extent": None,
        "survey_no": None,
        "house_no": None,
    }
    
    if not ec_details or not isinstance(ec_details, list):
        return extracted
    
    for entry in ec_details:
        if not isinstance(entry, dict):
            continue
        
        # Extract property description
        desc = entry.get("description", "")
        if desc:
            if not extracted["property_description"]:
                extracted["property_description"] = desc
            
            # Extract boundaries from EC description [N]: [S]: [E]: [W]: format
            boundary_map = {'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west'}
            for short, full in boundary_map.items():
                pattern = rf'\[{short}\][:\s]*([^\[\]]+?)(?:\[|$)'
                match = re.search(pattern, desc)
                if match and not extracted["boundaries"].get(full):
                    extracted["boundaries"][full] = match.group(1).strip()
            
            # Extract extent from description
            extent_match = re.search(r'EXTENT[:\s]*([\d\.]+)\s*(?:SQ\.?\s*(?:YDS?|FT|M))', desc, re.IGNORECASE)
            if extent_match and not extracted["extent"]:
                extracted["extent"] = extent_match.group(0)
            
            # Extract survey from description
            survey_match = re.search(r'SURVEY[:\s]*(\d+)', desc, re.IGNORECASE)
            if survey_match and not extracted["survey_no"]:
                extracted["survey_no"] = survey_match.group(1)
            
            # Extract house number from description
            house_match = re.search(r'HOUSE[:\s]*([\d\-]+)', desc, re.IGNORECASE)
            if house_match and not extracted["house_no"]:
                extracted["house_no"] = house_match.group(1)
        
        # Parse identifiers for doc number and SRO
        # IMPROVED: Handle format like "0/0\n1101/2026 [1]\nof SRO..."
        # The actual doc number has a 4-digit year
        identifiers = entry.get("identifiers", "")
        
        # Look for document number with 4-digit year (actual document number)
        doc_patterns = [
            r'(\d+)/(\d{4})\s*\[',  # "1101/2026 [1]" format
            r'(\d+)/(\d{4})(?:\s|$|\n)',  # "1101/2026" at end or with space
            r'(\d{2,5})/(\d{4})',  # General pattern with 4-digit year
        ]
        
        doc_no = None
        for pattern in doc_patterns:
            match = re.search(pattern, identifiers)
            if match:
                num, year = match.groups()
                # Skip if it's "0/0" type placeholder
                if num != "0" and int(year) >= 1900:
                    doc_no = f"{int(num)}/{year}"
                    break
        
        # Extract SRO from identifiers
        sro_patterns = [
            r'SRO\s*\n?\s*([A-Z][A-Z\s]+?)(?:\(|\n|$)',
            r'of\s*SRO\s*\n?\s*([A-Z][A-Z\s]+)',
        ]
        for pattern in sro_patterns:
            match = re.search(pattern, identifiers, re.IGNORECASE)
            if match and not extracted["sro"]:
                extracted["sro"] = match.group(1).strip()
                break
        
        # Parse deed value for deed type AND market/consideration values
        deed_value = entry.get("deedValue", "")
        deed_type = None
        deed_code = None
        
        # Extract deed code and type
        code_match = re.search(r'^(\d+)\s*\n?([A-Za-z\s]+)', deed_value)
        if code_match:
            deed_code = code_match.group(1)
            deed_type_raw = code_match.group(2).strip()
            if "Gift" in deed_type_raw:
                deed_type = "Gift Settlement"
            elif "Sale" in deed_type_raw:
                deed_type = "Sale Deed"
            elif "Mortgage" in deed_type_raw:
                deed_type = "Mortgage Deed"
                extracted["mortgage_flag"] = True
            elif "Partition" in deed_type_raw:
                deed_type = "Partition Deed"
            else:
                deed_type = deed_type_raw
        
        # Extract market value and consideration value
        mkt_match = re.search(r'Mkt\.?\s*Value[:\s]*Rs\.?\s*([\d,]+)', deed_value, re.IGNORECASE)
        if mkt_match:
            extracted["market_value"] = mkt_match.group(1).replace(',', '')
        
        cons_match = re.search(r'Cons\.?\s*Value[:\s]*Rs\.?\s*([\d,]+)', deed_value, re.IGNORECASE)
        if cons_match:
            extracted["consideration_value"] = cons_match.group(1).replace(',', '')
        
        # Parse dates - extract all three types (R)egistration, (E)xecution, (P)resentation
        dates_str = entry.get("dates", "")
        reg_date = None
        exec_date = None
        
        reg_match = re.search(r'\(R\)\s*([\d\-]+)', dates_str)
        if reg_match:
            reg_date = reg_match.group(1)
        
        exec_match = re.search(r'\(E\)\s*([\d\-]+)', dates_str)
        if exec_match:
            exec_date = exec_match.group(1)
        
        # Parse parties
        parties = entry.get("parties", "")
        executant = None
        claimant = None
        
        # Extract executant (DE) and claimant (DR) from parties
        de_match = re.search(r'\(DE\)\s*([A-Za-z\s]+?)(?:\(|$|\n|\d)', parties)
        if de_match:
            executant = de_match.group(1).strip()
        
        dr_match = re.search(r'\(DR\)\s*([A-Za-z\s]+?)(?:\(|$|\n|\d)', parties)
        if dr_match:
            claimant = dr_match.group(1).strip()
        
        txn = {
            "doc_no": doc_no,
            "deed_code": deed_code,
            "deed_type": deed_type,
            "registration_date": reg_date,
            "execution_date": exec_date,
            "parties": parties,
            "executant": executant,
            "claimant": claimant,
            "market_value": extracted["market_value"],
            "consideration_value": extracted["consideration_value"],
            "description": desc[:300] if desc else None,
        }
        
        extracted["transactions"].append(txn)
        
        # Check for mortgage in deed type
        if deed_type and "mortgage" in deed_type.lower():
            extracted["mortgage_flag"] = True
    
    return extracted


def extract_from_report_json(report: Dict) -> Dict:
    """
    Extract key fields from reportJson.
    Handles the actual NirnAI report format.
    """
    extracted = {
        "property_details": {},
        "schedule": {},
        "sections_text": [],
        "documents_scrutinized": [],
        "boundaries": {},
    }
    
    if not report or not isinstance(report, dict):
        return extracted
    
    # Extract property details from flat fields
    extracted["property_details"] = {
        "code": report.get("code"),
        "applicant": report.get("applicant"),
        "owner": report.get("ownerName1"),
        "doc_no": normalize_doc_no(report.get("registrationNo")),
        "deed_type": report.get("natureOfDeed"),
        "sro": report.get("registeredSRO"),
        "survey_no": normalize_survey_no(report.get("surveyNoDeed")),
        "house_no": report.get("houseNoOld") or report.get("houseNoGP"),
        "flat_no": report.get("flatNo") or report.get("flatNoDeed"),
        "plot_no": report.get("plotNo"),
        "assessment_no": report.get("assessmentNo"),
        "extent": report.get("propertyExtent"),
        "village": report.get("aliasName"),
        "taluk": report.get("taluk"),
        "district": report.get("district"),
        "state": report.get("state"),
        "mutation": report.get("mutation"),
        "accessibility": report.get("accessibility"),
        "document_age": report.get("mortgateDocumentAge"),
        "loan_amount": report.get("loanAmount"),
        "property_type": report.get("propertyType"),
    }
    
    # Extract boundaries
    boundaries_list = report.get("boundaries", [])
    if boundaries_list and isinstance(boundaries_list, list) and len(boundaries_list) > 0:
        b = boundaries_list[0]
        extracted["boundaries"] = {
            "north": b.get("boundaryN"),
            "south": b.get("boundaryS"),
            "east": b.get("boundaryE"),
            "west": b.get("boundaryW"),
        }
        extracted["schedule"]["schedule_no"] = b.get("scheduleNo")
    
    # Extract schedule from property details
    extracted["schedule"] = {
        **extracted.get("schedule", {}),
        "survey_no": extracted["property_details"].get("survey_no"),
        "house_no": extracted["property_details"].get("house_no"),
        "extent": extracted["property_details"].get("extent"),
        "village": extracted["property_details"].get("village"),
        "district": extracted["property_details"].get("district"),
        "state": extracted["property_details"].get("state"),
    }
    
    # Extract sections text
    sections = report.get("sections", [])
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                content = section.get("content", "")
                if content:
                    extracted["sections_text"].append(content)
    
    # Extract documents scrutinized
    req_docs = report.get("requiredDocuments", [])
    if isinstance(req_docs, list):
        for doc in req_docs:
            if isinstance(doc, dict):
                extracted["documents_scrutinized"].append({
                    "type": doc.get("docType"),
                    "number": doc.get("docNumber"),
                    "date": doc.get("docDate"),
                    "subtype": doc.get("subType"),
                })
    
    return extracted


def build_fingerprint(merged_case: Dict) -> str:
    """
    Build a fingerprint string for the current case.
    Used for RAG retrieval to find similar precedents.
    Updated for actual NirnAI format.
    IMPROVED: Uses better EC extraction and includes more matching criteria.
    """
    parts = []
    
    # Extract from all three sources
    attachments = merged_case.get('attachments', [])
    ec_details = merged_case.get('encumbranceDetails', [])
    report = merged_case.get('reportJson', {})
    
    att_extracted = extract_from_attachments(attachments)
    ec_extracted = extract_from_encumbrance_details(ec_details)
    report_extracted = extract_from_report_json(report)
    
    # Get first EC transaction for details
    ec_transactions = ec_extracted.get('transactions', [])
    ec_txn = ec_transactions[0] if ec_transactions else {}
    
    # Add state/location (prefer report as it's structured)
    state = report_extracted['property_details'].get('state')
    if state:
        parts.append(f"State: {state}")
    
    district = report_extracted['property_details'].get('district')
    if district:
        parts.append(f"District: {district}")
    
    sro = report_extracted['property_details'].get('sro') or ec_extracted.get('sro')
    if sro:
        parts.append(f"SRO: {sro}")
    
    # Add property identifiers
    survey_no = report_extracted['property_details'].get('survey_no') or att_extracted.get('survey_no') or ec_extracted.get('survey_no')
    if survey_no:
        parts.append(f"Survey: {survey_no}")
    
    village = report_extracted['property_details'].get('village') or att_extracted.get('village')
    if village:
        parts.append(f"Village: {village}")
    
    extent = report_extracted['property_details'].get('extent') or att_extracted.get('extent') or ec_extracted.get('extent')
    if extent:
        parts.append(f"Extent: {extent}")
    
    # Add deed type - prefer EC as it's most reliable
    deed_type = ec_txn.get('deed_type') or report_extracted['property_details'].get('deed_type') or att_extracted.get('deed_type')
    if deed_type:
        parts.append(f"Deed: {deed_type}")
    
    # Add document number - prefer EC's properly extracted number
    doc_no = ec_txn.get('doc_no') or report_extracted['property_details'].get('doc_no') or att_extracted.get('doc_no')
    if doc_no:
        parts.append(f"DocNo: {doc_no}")
    
    # Add mortgage flag
    if ec_extracted.get('mortgage_flag'):
        parts.append("Mortgage: Active")
    
    # Add owner/applicant
    owner = report_extracted['property_details'].get('owner') or report_extracted['property_details'].get('applicant')
    if owner:
        parts.append(f"Owner: {owner}")
    
    # Add market value context (helps find similar value range cases)
    mkt_value = ec_extracted.get('market_value') or att_extracted.get('market_value')
    if mkt_value:
        try:
            val = int(mkt_value)
            if val < 100000:
                parts.append("ValueRange: <1L")
            elif val < 500000:
                parts.append("ValueRange: 1-5L")
            elif val < 1000000:
                parts.append("ValueRange: 5-10L")
            else:
                parts.append("ValueRange: >10L")
        except:
            pass
    
    # Add mutation status (important for finding similar cases)
    mutation = report_extracted['property_details'].get('mutation')
    if mutation:
        parts.append(f"Mutation: {mutation}")
    
    return " | ".join(parts)


def build_current_case_extract(merged_case: Dict) -> Dict:
    """
    Build a token-efficient extract of the current case for LLM prompts.
    This is what goes into the LLM, NOT the full JSON.
    Updated for actual NirnAI format.
    IMPROVED: Includes market value comparison, execution vs registration dates,
    and better boundary extraction.
    """
    attachments = merged_case.get('attachments', [])
    ec_details = merged_case.get('encumbranceDetails', [])
    report = merged_case.get('reportJson', {})
    
    att_extracted = extract_from_attachments(attachments)
    ec_extracted = extract_from_encumbrance_details(ec_details)
    report_extracted = extract_from_report_json(report)
    
    # Get EC transaction details for document comparison
    ec_transactions = ec_extracted.get('transactions', [])
    ec_doc_no = ec_transactions[0].get('doc_no') if ec_transactions else None
    ec_exec_date = ec_transactions[0].get('execution_date') if ec_transactions else None
    ec_reg_date = ec_transactions[0].get('registration_date') if ec_transactions else None
    
    extract = {
        "case_info": {
            "code": report_extracted['property_details'].get('code'),
            "branch": report.get('branch'),
            "lan": report.get('lan'),
            "policy": report.get('policy'),
            "loan_amount": report_extracted['property_details'].get('loan_amount'),
        },
        "owner_applicant": {
            "applicant": report_extracted['property_details'].get('applicant'),
            "owner_in_report": report_extracted['property_details'].get('owner'),
            "executant_from_deed": att_extracted.get('executant'),
            "claimant_from_deed": att_extracted.get('claimant'),
            "executant_from_ec": ec_transactions[0].get('executant') if ec_transactions else None,
            "claimant_from_ec": ec_transactions[0].get('claimant') if ec_transactions else None,
            "relationship": report.get('mortgagorRelationship'),
        },
        "title_deed": {
            "doc_no_report": report_extracted['property_details'].get('doc_no'),
            "doc_no_from_deed": att_extracted.get('doc_no'),
            "doc_no_from_ec": ec_doc_no,
            "deed_type_report": report_extracted['property_details'].get('deed_type'),
            "deed_type_from_deed": att_extracted.get('deed_type'),
            "deed_type_from_ec": ec_transactions[0].get('deed_type') if ec_transactions else None,
            "sro_report": report_extracted['property_details'].get('sro'),
            "sro_from_ec": ec_extracted.get('sro'),
            "document_age": report_extracted['property_details'].get('document_age'),
        },
        "dates": {
            "execution_date_from_deed": att_extracted.get('execution_date'),
            "execution_date_from_ec": ec_exec_date,
            "registration_date_from_deed": att_extracted.get('registration_date'),
            "registration_date_from_ec": ec_reg_date,
            "note": "Execution date is when deed was signed; Registration date is when it was registered at SRO",
        },
        "values": {
            "market_value_from_deed": att_extracted.get('market_value'),
            "market_value_from_ec": ec_extracted.get('market_value'),
            "consideration_value_from_ec": ec_extracted.get('consideration_value'),
            "note": "Market value is govt assessed value; Consideration value is declared transaction value",
        },
        "schedule": {
            "survey_no_report": report_extracted['schedule'].get('survey_no'),
            "survey_no_from_deed": att_extracted.get('survey_no'),
            "survey_no_from_ec": ec_extracted.get('survey_no'),
            "house_no_report": report_extracted['property_details'].get('house_no') or report_extracted['property_details'].get('flat_no'),
            "house_no_from_deed": att_extracted.get('house_no'),
            "house_no_from_ec": ec_extracted.get('house_no'),
            "flat_no": report_extracted['property_details'].get('flat_no'),
            "plot_no": report_extracted['property_details'].get('plot_no'),
            "assessment_no": report_extracted['property_details'].get('assessment_no'),
            "village": report_extracted['schedule'].get('village'),
            "taluk": report_extracted['property_details'].get('taluk'),
            "district": report_extracted['schedule'].get('district'),
            "state": report_extracted['schedule'].get('state'),
            "extent_report": report_extracted['schedule'].get('extent'),
            "extent_from_deed": att_extracted.get('extent'),
            "extent_from_ec": ec_extracted.get('extent'),
        },
        "boundaries": {
            "from_report": report_extracted.get('boundaries', {}),
            "from_deed": att_extracted.get('boundaries', {}),
            "from_ec": ec_extracted.get('boundaries', {}),
        },
        "ec_summary": {
            "transactions_count": len(ec_transactions),
            "transactions": ec_transactions[:5],  # Limit
            "mortgage_flag": ec_extracted.get('mortgage_flag'),
            "property_description": _truncate_text(ec_extracted.get('property_description', ''), 300),
        },
        "report_sections": [
            _truncate_text(s, 500) for s in report_extracted.get('sections_text', [])[:5]
        ],
        "documents_scrutinized": report_extracted.get('documents_scrutinized', [])[:10],
        "mutation_status": report_extracted['property_details'].get('mutation'),
        "accessibility": report_extracted['property_details'].get('accessibility'),
        # Use cleaned deed content instead of raw noisy text
        "source_doc_snippet": _truncate_text(att_extracted.get('cleaned_text', '') or att_extracted.get('raw_text', ''), 1500),
    }
    
    return extract


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


# =============================================================================
# DEDUPLICATION
# =============================================================================

def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate string similarity ratio."""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def deduplicate_issues(issues: List[Dict]) -> List[Dict]:
    """
    Deduplicate issues that point to the same underlying mismatch.
    Uses location + evidence similarity to detect duplicates.
    """
    if not issues:
        return []
    
    deduplicated = []
    
    for issue in issues:
        is_duplicate = False
        
        for existing in deduplicated:
            # Same location
            if issue.get('location', '').lower() == existing.get('location', '').lower():
                # Check evidence similarity
                issue_evidence = str(issue.get('evidence', {}).get('from_report', ''))
                existing_evidence = str(existing.get('evidence', {}).get('from_report', ''))
                
                if similarity_ratio(issue_evidence, existing_evidence) > 0.7:
                    is_duplicate = True
                    # Keep the one with higher severity
                    severity_order = {'critical': 3, 'major': 2, 'minor': 1}
                    if severity_order.get(issue.get('severity', 'minor'), 0) > \
                       severity_order.get(existing.get('severity', 'minor'), 0):
                        # Replace with higher severity issue
                        deduplicated.remove(existing)
                        deduplicated.append(issue)
                    break
        
        if not is_duplicate:
            deduplicated.append(issue)
    
    return deduplicated


def renumber_issues(issues_by_section: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Renumber issue IDs to be consistent within each section.
    E.g., PD-01, PD-02 for property_details, EC-01 for encumbrance_certificate.
    """
    section_prefixes = {
        'property_details': 'PD',
        'schedule_of_property': 'SP',
        'documents_scrutinized': 'DS',
        'encumbrance_certificate': 'EC',
        'flow_of_title': 'FT',
        'mutation_and_tax': 'MT',
        'conclusion_and_remarks': 'CR',
        'layout_and_flowchart': 'LF',
    }
    
    result = {}
    
    for section, issues in issues_by_section.items():
        prefix = section_prefixes.get(section, 'XX')
        renumbered = []
        
        for i, issue in enumerate(issues, 1):
            issue_copy = issue.copy()
            issue_copy['id'] = f"{prefix}-{i:02d}"
            renumbered.append(issue_copy)
        
        result[section] = renumbered
    
    return result


def get_evidence_snippet(merged_case: Dict, source: str, search_terms: List[str]) -> Optional[str]:
    """
    Search for evidence snippet in the merged case JSON.
    Updated for actual NirnAI format.
    IMPROVED: Filters out stamp paper noise from attachments.
    
    Args:
        merged_case: The full merged case JSON
        source: 'report', 'attachments', 'ec', or 'deed' (cleaned attachments)
        search_terms: Terms to search for
    
    Returns:
        Matching snippet or None
    """
    if source == 'report':
        data = merged_case.get('reportJson', {})
        data_str = json.dumps(data, default=str)
    elif source == 'attachments':
        # Use raw attachments
        data = merged_case.get('attachments', [])
        data_str = "\n".join(data) if isinstance(data, list) else str(data)
    elif source == 'deed':
        # Use cleaned/filtered attachments (deed content only)
        data = merged_case.get('attachments', [])
        raw_text = "\n".join(data) if isinstance(data, list) else str(data)
        data_str = _filter_stamp_paper_noise(raw_text)
    elif source == 'ec':
        data = merged_case.get('encumbranceDetails', [])
        data_str = json.dumps(data, default=str)
    else:
        data = merged_case
        data_str = json.dumps(data, default=str)
    
    # Stamp paper noise patterns to avoid in snippets
    noise_patterns = [
        r'(?i)twenty\s*rupees?',
        r'(?i)india\s*non\s*judicial',
        r'(?i)satyameva?\s*jayate?',
        r'(?i)denomination',
        r'(?i)stamp\s*s\.?\s*no',
    ]
    
    for term in search_terms:
        if not term or len(term) < 2:
            continue
            
        if term.lower() in data_str.lower():
            # Find context around the term
            idx = data_str.lower().find(term.lower())
            start = max(0, idx - 75)
            end = min(len(data_str), idx + len(term) + 125)
            snippet = data_str[start:end]
            
            # Check if this snippet is mostly noise
            is_noise = False
            for noise_pattern in noise_patterns:
                if re.search(noise_pattern, snippet):
                    # This might be stamp paper text - try to find a better match
                    # Look for the next occurrence
                    next_idx = data_str.lower().find(term.lower(), idx + len(term))
                    if next_idx > 0:
                        start = max(0, next_idx - 75)
                        end = min(len(data_str), next_idx + len(term) + 125)
                        snippet = data_str[start:end]
                        # Check again
                        if not any(re.search(p, snippet) for p in noise_patterns):
                            is_noise = False
                            break
                    is_noise = True
                    break
            
            # If still noise, skip this term and try the next
            if is_noise:
                continue
            
            # Clean up JSON artifacts
            snippet = re.sub(r'[{}\[\]"]', ' ', snippet)
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            
            # Final validation - snippet should have meaningful content
            if len(snippet) > 20:
                return f"...{snippet}..."
    
    return None


def compare_values(val1: Any, val2: Any, tolerance: float = 0.05) -> Dict:
    """
    Compare two values and return comparison result.
    Handles numeric comparison with tolerance and string comparison.
    
    Args:
        val1: First value
        val2: Second value
        tolerance: Tolerance for numeric comparison (default 5%)
    
    Returns:
        Dict with 'match', 'val1', 'val2', and 'difference' (if numeric)
    """
    result = {
        "val1": val1,
        "val2": val2,
        "match": False,
        "difference": None,
    }
    
    if val1 is None or val2 is None:
        result["match"] = val1 == val2
        return result
    
    # Try numeric comparison
    try:
        # Clean numeric values
        num1 = float(re.sub(r'[^\d.]', '', str(val1)))
        num2 = float(re.sub(r'[^\d.]', '', str(val2)))
        
        if num1 == 0 and num2 == 0:
            result["match"] = True
        elif num1 == 0 or num2 == 0:
            result["match"] = False
            result["difference"] = abs(num1 - num2)
        else:
            diff_pct = abs(num1 - num2) / max(num1, num2)
            result["match"] = diff_pct <= tolerance
            result["difference"] = abs(num1 - num2)
            result["difference_pct"] = round(diff_pct * 100, 2)
        
        return result
    except (ValueError, TypeError):
        pass
    
    # String comparison
    str1 = str(val1).lower().strip()
    str2 = str(val2).lower().strip()
    
    result["match"] = str1 == str2 or similarity_ratio(str1, str2) > 0.85
    
    return result
