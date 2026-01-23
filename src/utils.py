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


def extract_from_attachments(attachments: List[str]) -> Dict:
    """
    Extract key fields from attachments (OCR'd document text).
    This parses the raw text to find key information.
    """
    extracted = {
        "raw_text": "",
        "doc_no": None,
        "deed_type": None,
        "executant": None,
        "claimant": None,
        "extent": None,
        "survey_no": None,
        "village": None,
        "mandal": None,
        "district": None,
        "state": None,
        "boundaries": {},
        "registration_date": None,
        "sro": None,
    }
    
    if not attachments:
        return extracted
    
    # Combine all attachment text
    full_text = "\n".join(attachments) if isinstance(attachments, list) else str(attachments)
    extracted["raw_text"] = full_text[:5000]  # Limit for token control
    
    # Extract document number (patterns like "DOC.NO. 39/2026" or "Doc No/Year: 39/2026")
    doc_match = re.search(r'(?:DOC\.?\s*NO\.?|Doc\s*No[/\s]*Year)[:\s]*(\d+[/\s]*(?:of\s*)?\d+)', full_text, re.IGNORECASE)
    if doc_match:
        extracted["doc_no"] = normalize_doc_no(doc_match.group(1))
    
    # Extract deed type
    deed_patterns = [
        r'(Settlement\s*Deed|Gift\s*Settlement|Sale\s*Deed|Gift\s*Deed|Partition\s*Deed|Release\s*Deed|Mortgage\s*Deed)',
    ]
    for pattern in deed_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted["deed_type"] = match.group(1).strip()
            break
    
    # Extract names (executant/claimant patterns)
    # Look for "(DE)" for executant and "(DR)" for claimant in registration text
    de_match = re.search(r'\(DE\)\s*([A-Za-z\s]+?)(?:\(|$|\n)', full_text)
    if de_match:
        extracted["executant"] = de_match.group(1).strip()
    
    dr_match = re.search(r'\(DR\)\s*([A-Za-z\s]+?)(?:\(|$|\n)', full_text)
    if dr_match:
        extracted["claimant"] = dr_match.group(1).strip()
    
    # Extract survey number
    survey_match = re.search(r'(?:Survey\s*(?:No\.?|Number)?|Sy\.?\s*No\.?)[:\s]*(\d+[/\-]?\d*)', full_text, re.IGNORECASE)
    if survey_match:
        extracted["survey_no"] = survey_match.group(1)
    
    # Extract extent
    extent_match = re.search(r'(\d+)\s*(?:Sq\.?\s*(?:Yds?|Ft|M)|Square\s*(?:Yards?|Feet|Meters?))', full_text, re.IGNORECASE)
    if extent_match:
        extracted["extent"] = extent_match.group(0)
    
    # Extract location
    village_match = re.search(r'(?:Village|Vill)[:\s]*([A-Za-z\s]+?)(?:,|\n|Mandal|District)', full_text, re.IGNORECASE)
    if village_match:
        extracted["village"] = village_match.group(1).strip()
    
    mandal_match = re.search(r'Mandal[:\s]*([A-Za-z\s]+?)(?:,|\n|District)', full_text, re.IGNORECASE)
    if mandal_match:
        extracted["mandal"] = mandal_match.group(1).strip()
    
    district_match = re.search(r'District[:\s]*([A-Za-z\s]+?)(?:,|\n|State|registered)', full_text, re.IGNORECASE)
    if district_match:
        extracted["district"] = district_match.group(1).strip()
    
    # Extract boundaries
    boundary_patterns = {
        'north': r'(?:North|N)[:\s]*([^,\n\[\]]+?)(?:,|\n|\[|South|East|West|$)',
        'south': r'(?:South|S)[:\s]*([^,\n\[\]]+?)(?:,|\n|\[|North|East|West|$)',
        'east': r'(?:East|E)[:\s]*([^,\n\[\]]+?)(?:,|\n|\[|North|South|West|$)',
        'west': r'(?:West|W)[:\s]*([^,\n\[\]]+?)(?:,|\n|\[|North|South|East|$)',
    }
    
    for direction, pattern in boundary_patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted["boundaries"][direction] = match.group(1).strip()
    
    # Extract SRO
    sro_match = re.search(r'(?:SRO|Sub-?Registrar)[:\s]*([A-Za-z\.\s]+?)(?:\(|\n|,)', full_text, re.IGNORECASE)
    if sro_match:
        extracted["sro"] = sro_match.group(1).strip()
    
    return extracted


def extract_from_encumbrance_details(ec_details: List[Dict]) -> Dict:
    """
    Extract key fields from encumbranceDetails array.
    Handles the actual NirnAI EC format.
    """
    extracted = {
        "transactions": [],
        "mortgage_flag": False,
        "property_description": None,
        "sro": None,
    }
    
    if not ec_details or not isinstance(ec_details, list):
        return extracted
    
    for entry in ec_details:
        if not isinstance(entry, dict):
            continue
        
        # Extract property description
        desc = entry.get("description", "")
        if desc and not extracted["property_description"]:
            extracted["property_description"] = desc
        
        # Parse identifiers for doc number and SRO
        identifiers = entry.get("identifiers", "")
        doc_match = re.search(r'(\d+/\d+)', identifiers)
        sro_match = re.search(r'SRO\s*([A-Za-z\.\s]+?)(?:\(|$)', identifiers, re.IGNORECASE)
        
        doc_no = doc_match.group(1) if doc_match else None
        if sro_match and not extracted["sro"]:
            extracted["sro"] = sro_match.group(1).strip()
        
        # Parse deed value for deed type
        deed_value = entry.get("deedValue", "")
        deed_type = None
        if "Gift" in deed_value:
            deed_type = "Gift Settlement"
        elif "Sale" in deed_value:
            deed_type = "Sale Deed"
        elif "Mortgage" in deed_value:
            deed_type = "Mortgage Deed"
            extracted["mortgage_flag"] = True
        
        # Parse dates
        dates_str = entry.get("dates", "")
        reg_date = None
        date_match = re.search(r'\(R\)\s*([\d\-]+)', dates_str)
        if date_match:
            reg_date = date_match.group(1)
        
        # Parse parties
        parties = entry.get("parties", "")
        
        txn = {
            "doc_no": normalize_doc_no(doc_no) if doc_no else None,
            "deed_type": deed_type,
            "registration_date": reg_date,
            "parties": parties,
            "description": desc[:200] if desc else None,
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
    """
    parts = []
    
    # Extract from all three sources
    attachments = merged_case.get('attachments', [])
    ec_details = merged_case.get('encumbranceDetails', [])
    report = merged_case.get('reportJson', {})
    
    att_extracted = extract_from_attachments(attachments)
    ec_extracted = extract_from_encumbrance_details(ec_details)
    report_extracted = extract_from_report_json(report)
    
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
    survey_no = report_extracted['property_details'].get('survey_no') or att_extracted.get('survey_no')
    if survey_no:
        parts.append(f"Survey: {survey_no}")
    
    village = report_extracted['property_details'].get('village') or att_extracted.get('village')
    if village:
        parts.append(f"Village: {village}")
    
    extent = report_extracted['property_details'].get('extent') or att_extracted.get('extent')
    if extent:
        parts.append(f"Extent: {extent}")
    
    # Add deed type
    deed_type = report_extracted['property_details'].get('deed_type') or att_extracted.get('deed_type')
    if deed_type:
        parts.append(f"Deed: {deed_type}")
    
    # Add document number
    doc_no = report_extracted['property_details'].get('doc_no') or att_extracted.get('doc_no')
    if doc_no:
        parts.append(f"DocNo: {doc_no}")
    
    # Add mortgage flag
    if ec_extracted.get('mortgage_flag'):
        parts.append("Mortgage: Active")
    
    # Add owner/applicant
    owner = report_extracted['property_details'].get('owner') or report_extracted['property_details'].get('applicant')
    if owner:
        parts.append(f"Owner: {owner}")
    
    return " | ".join(parts)


def build_current_case_extract(merged_case: Dict) -> Dict:
    """
    Build a token-efficient extract of the current case for LLM prompts.
    This is what goes into the LLM, NOT the full JSON.
    Updated for actual NirnAI format.
    """
    attachments = merged_case.get('attachments', [])
    ec_details = merged_case.get('encumbranceDetails', [])
    report = merged_case.get('reportJson', {})
    
    att_extracted = extract_from_attachments(attachments)
    ec_extracted = extract_from_encumbrance_details(ec_details)
    report_extracted = extract_from_report_json(report)
    
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
            "executant_from_docs": att_extracted.get('executant'),
            "claimant_from_docs": att_extracted.get('claimant'),
        },
        "title_deed": {
            "doc_no_report": report_extracted['property_details'].get('doc_no'),
            "doc_no_source": att_extracted.get('doc_no'),
            "deed_type": report_extracted['property_details'].get('deed_type'),
            "sro": report_extracted['property_details'].get('sro'),
            "document_age": report_extracted['property_details'].get('document_age'),
        },
        "schedule": {
            "survey_no_report": report_extracted['schedule'].get('survey_no'),
            "survey_no_source": att_extracted.get('survey_no'),
            "house_no": report_extracted['property_details'].get('house_no'),
            "flat_no": report_extracted['property_details'].get('flat_no'),
            "plot_no": report_extracted['property_details'].get('plot_no'),
            "assessment_no": report_extracted['property_details'].get('assessment_no'),
            "village": report_extracted['schedule'].get('village'),
            "taluk": report_extracted['property_details'].get('taluk'),
            "district": report_extracted['schedule'].get('district'),
            "state": report_extracted['schedule'].get('state'),
            "extent_report": report_extracted['schedule'].get('extent'),
            "extent_source": att_extracted.get('extent'),
        },
        "boundaries": {
            "from_report": report_extracted.get('boundaries', {}),
            "from_docs": att_extracted.get('boundaries', {}),
        },
        "ec_summary": {
            "transactions_count": len(ec_extracted.get('transactions', [])),
            "transactions": ec_extracted.get('transactions', [])[:5],  # Limit
            "mortgage_flag": ec_extracted.get('mortgage_flag'),
            "property_description": _truncate_text(ec_extracted.get('property_description', ''), 300),
        },
        "report_sections": [
            _truncate_text(s, 500) for s in report_extracted.get('sections_text', [])[:5]
        ],
        "documents_scrutinized": report_extracted.get('documents_scrutinized', [])[:10],
        "mutation_status": report_extracted['property_details'].get('mutation'),
        "accessibility": report_extracted['property_details'].get('accessibility'),
        "source_doc_snippet": _truncate_text(att_extracted.get('raw_text', ''), 1500),
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
    
    Args:
        merged_case: The full merged case JSON
        source: 'report', 'attachments', or 'ec'
        search_terms: Terms to search for
    
    Returns:
        Matching snippet or None
    """
    if source == 'report':
        data = merged_case.get('reportJson', {})
    elif source == 'attachments':
        data = merged_case.get('attachments', [])
    elif source == 'ec':
        data = merged_case.get('encumbranceDetails', [])
    else:
        data = merged_case
    
    data_str = json.dumps(data, default=str) if not isinstance(data, str) else data
    
    for term in search_terms:
        if term.lower() in data_str.lower():
            # Find context around the term
            idx = data_str.lower().find(term.lower())
            start = max(0, idx - 50)
            end = min(len(data_str), idx + len(term) + 100)
            snippet = data_str[start:end]
            
            # Clean up JSON artifacts
            snippet = re.sub(r'[{}\[\]"]', ' ', snippet)
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            
            return f"...{snippet}..."
    
    return None
