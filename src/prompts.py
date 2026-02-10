"""
Prompt templates for NirnAI Two-Stage LLM Review.
Contains Generator and Critic prompts with condensed SOP checklist.
"""

# =============================================================================
# SOP CHECKLIST (Condensed for token efficiency)
# =============================================================================

SOP_CHECKLIST = """
## SOP RULES FOR TITLE VERIFICATION

### A. PROPERTY DETAILS
- Owner name must match latest title deed and EC (spelling, initials, surname)
- Doc No. and Year must match deed exactly
- Deed type (Sale/Gift/Partition/etc.) must match deed
- SRO name must match deed and EC
- House number, assessment number, survey number must match deed
- Village, Mandal, District, State must be complete and correct
- Extent with units must match deed and be consistent across all sections

### B. SCHEDULE OF PROPERTY  
- Include house no, assessment no, survey no, village, mandal, district, state as per deed
- Boundaries (N/S/E/W) must match deed exactly
- If EC boundaries differ, treat deed as primary and note mismatch in Additional Remarks
- Relationship between parties must match deed recitals
- Conditional settlements (life interest etc.) must be captured

### C. DOCUMENTS SCRUTINIZED
- List must include ALL documents relied upon
- Must be ordered oldest to newest
- Each item needs accurate description, date, type (Original/Photocopy/Online/Scanned)

### D. ENCUMBRANCE CERTIFICATE
- EC period must cover required policy period
- All relevant doc nos, years, deed types from EC must be reflected
- All mortgages and receipt deeds affecting property must be captured
- Active mortgages: conclusion must state "subject to discharge/release"
- If EC from FI, add note in Additional Remarks

### E. FLOW OF TITLE
- Title chain must be continuous from earliest link document to current owner
- Each step: correct registration number, date, parties, extent, identifiers
- Deaths must be supported by death certificates
- Representatives (PoA, sons) must be clearly stated
- Extent at each stage must match the deed for that stage

### F. MUTATION AND TAX
- Mutation data must be consistent with HTR
- Tax receipts must correctly capture payer name, receipt number, period
- If tax in deceased person's name, mention in Additional Remarks

### G. CONCLUSION AND REMARKS
- Must restate correct owner, extent, identifiers, location
- Must mention any encumbrances, conditional interests, EC limitations as "subject to"
- Additional Remarks must include: boundary mismatches, conditional settlements, presumptions, FI-supplied EC note

### H. STATE-SPECIFIC CONSIDERATIONS
- **TAMIL NADU**: 
  - EC uses Tamil script for party names and boundaries
  - Direction abbreviations: (வ)=North, (ெத)=South, (கி)=East, (ேம)=West
  - Document format: "Docno/Docyear: NNNN/YYYY"
  - Survey numbers may include multiple numbers: "225/2, 228/1B2B"
  - Plot/Site numbers are primary property identifiers in urban areas
  - Deed types: "Conveyance" = "Sale Deed", "Gift Settlement" = "Gift Deed"
  - Party extraction uses "Executant(s):" and "Claimant(s):" format
- **TELANGANA/ANDHRA PRADESH**:
  - EC uses [N]: [S]: [E]: [W]: format for boundaries
  - Party format: (DE) for executant, (DR) for claimant
  - Value format: "Mkt. Value: Rs. X, Cons. Value: Rs. Y"
- **KARNATAKA**:
  - May include Kannada script in some fields
"""

# =============================================================================
# GENERATOR PROMPT TEMPLATE (Stage 1: High-Recall Issue Detection)
# =============================================================================

GENERATOR_PROMPT_TEMPLATE = """You are a senior Indian real-estate title lawyer and L2 checker specializing in loan-against-property title verification.

## YOUR TASK
Review the MAKER-DRAFTED report against source documents. Generate ALL potential issues (high recall). Include even uncertain issues - they will be filtered later.

## CRITICAL REQUIREMENT: EVIDENCE-FIRST
For EVERY issue, you MUST provide BOTH:
1. evidence.from_report: Exact snippet from the report showing the issue
2. evidence.from_source_docs: Exact snippet from translated_docs and/or EC proving the mismatch

NO ISSUE IS VALID WITHOUT BOTH EVIDENCE SNIPPETS.

## STATE-SPECIFIC HANDLING
The case extract includes `detected_state` - use this for format-specific parsing:

**Tamil Nadu cases:**
- EC parties may be in Tamil script (பழனிசாமி, சரஸ்வதி, etc.) - compare meaning/transliteration, not exact script
- Boundaries use Tamil abbreviations: (வ)=North, (ெத)=South, (கி)=East, (ேம)=West
- "Conveyance" and "Sale Deed" are equivalent deed types
- "Gift Settlement" and "Gift Deed" are equivalent
- Document numbers use "Docno/Docyear: 4960/2011" format
- Survey numbers may include multiple numbers like "225/2, 228/1B2B"
- DO NOT confuse document numbers (NNNN/YYYY format) with survey numbers

**Telangana/AP cases:**
- Boundaries use [N]: [S]: [E]: [W]: format
- Parties use (DE) and (DR) format

## SOP RULES
{sop_checklist}

## RETRIEVED PRECEDENTS (Similar Historical Cases)
{precedent_snippets}

Use precedents to:
- If a precedent shows this pattern was accepted, consider lower severity
- If precedent shows recurring high-risk pattern, prioritize it
- Check precedent exceptions for known false-positives to avoid

## CURRENT CASE EXTRACT
The case extract below contains data from three sources:
- `source_doc_snippet`: Raw text from scanned/translated documents (attachments)
- `ec_summary`: Encumbrance certificate transactions
- `report_sections`: The maker's drafted report content

Key fields for comparison:
- `owner_applicant`: Compare owner_in_report with executant/claimant from EC (may be in Tamil script)
- `title_deed`: Compare doc_no_report with doc_no_from_ec (format may differ)
- `schedule`: Compare survey_no, house_no, plot_no across sources
- `boundaries`: Compare from_report with from_ec (direction labels may differ by state)

{case_extract}

## OUTPUT FORMAT
Return a JSON object with this exact structure:
```json
{{
  "overall_summary": "Brief summary of report quality",
  "overall_risk_level": "OK | NEEDS_FIX_BEFORE_RELEASE | BLOCKER",
  "sections": {{
    "property_details": [ISSUE...],
    "schedule_of_property": [ISSUE...],
    "documents_scrutinized": [ISSUE...],
    "encumbrance_certificate": [ISSUE...],
    "flow_of_title": [ISSUE...],
    "mutation_and_tax": [ISSUE...],
    "conclusion_and_remarks": [ISSUE...],
    "layout_and_flowchart": [ISSUE...]
  }}
}}
```

Each ISSUE must be:
```json
{{
  "id": "TEMP-01",
  "severity": "critical | major | minor",
  "location": "section name and field/paragraph",
  "rule": "SOP rule reference",
  "message_for_maker": "What is wrong",
  "suggested_fix": "How to fix it",
  "evidence": {{
    "from_report": "EXACT snippet from report",
    "from_source_docs": "EXACT snippet from translated_docs/EC"
  }}
}}
```

If a section has no issues, use an empty array.

Generate CANDIDATE_REVIEW JSON now. Be thorough - flag all potential issues.
"""

# =============================================================================
# CRITIC PROMPT TEMPLATE (Stage 2: Prune, Deduplicate, Calibrate)
# =============================================================================

CRITIC_PROMPT_TEMPLATE = """You are a senior quality control reviewer for legal opinion reports.

## YOUR TASK
Review the CANDIDATE_REVIEW from Stage 1 and produce the FINAL REVIEW_OBJECT.

## YOUR RESPONSIBILITIES
1. **Remove weak issues**: Delete any issue that:
   - Missing evidence.from_report (no exact snippet from report)
   - Missing evidence.from_source_docs (no exact snippet from source)
   - Evidence doesn't actually prove a mismatch
   - Issue is too generic without specific proof

2. **Merge duplicates**: Combine issues pointing to the same underlying mismatch

3. **Calibrate severity using precedents**:
   - If precedents show this variation is commonly accepted → downgrade or remove
   - If precedents show this is a recurring risk → keep or upgrade severity
   - Check precedent exceptions for known false-positives

4. **Ensure consistent IDs**: Renumber as PD-01, PD-02, SP-01, EC-01, etc.

5. **Empty sections**: If no valid issues remain, output empty array []

## STATE-SPECIFIC FALSE POSITIVE REMOVAL
When evaluating evidence for these cases, remove issues that are actually format differences, not real mismatches:

**Tamil Nadu cases:**
- Owner name in Tamil script (EC) vs English transliteration (report) is NOT a mismatch if they represent the same person
- "Conveyance" (EC) vs "Sale Deed" (report) is NOT a mismatch - they are equivalent
- "Gift Settlement" (EC) vs "Gift Deed" (report) is NOT a mismatch - they are equivalent
- Boundary format differences (Tamil abbreviations vs English) are NOT mismatches if content matches
- Document number format "4960/2011" vs "4960 of 2011" is NOT a mismatch
- DO NOT flag survey number issues if the "evidence" is actually a document number (NNNN/YYYY format)
- DO NOT use stamp paper text or OCR noise as evidence

**Telangana/AP cases:**
- Boundary format [N]: vs "North:" is NOT a mismatch if content matches

**Common false positives to REMOVE:**
- Survey number "4960" when it's actually document number 4960/2011 (wrong field)
- Boundary evidence from stamp paper noise (Rs., judicial, denomination)
- Owner mismatch when comparing report owner to EC executant (wrong comparison - should compare to claimant)

## PRECEDENTS (for severity calibration)
{precedent_snippets}

## CURRENT CASE EXTRACT (for reference)
{case_extract}

## CANDIDATE_REVIEW FROM STAGE 1
{candidate_review}

## OUTPUT FORMAT
Return ONLY the final REVIEW_OBJECT JSON:
```json
{{
  "overall_summary": "Concise summary of final assessment",
  "overall_risk_level": "CLEAR_TO_RELEASE | NEEDS_FIX_BEFORE_RELEASE | REJECT",
  "sections": {{
    "property_details": [ISSUE...],
    "schedule_of_property": [ISSUE...],
    "documents_scrutinized": [ISSUE...],
    "encumbrance_certificate": [ISSUE...],
    "flow_of_title": [ISSUE...],
    "mutation_and_tax": [ISSUE...],
    "conclusion_and_remarks": [ISSUE...],
    "layout_and_flowchart": [ISSUE...]
  }}
}}
```

Each final ISSUE:
```json
{{
  "id": "PD-01",
  "severity": "critical | major | minor",
  "location": "specific location in report",
  "rule": "SOP rule reference",
  "message_for_maker": "Clear explanation of the error",
  "suggested_fix": "Specific correction suggestion",
  "evidence": {{
    "from_report": "Exact snippet from report",
    "from_source_docs": "Exact snippet from source documents"
  }}
}}
```

IMPORTANT:
- Output ONLY valid JSON, no markdown formatting or explanation
- Every issue must have BOTH evidence snippets with exact text
- Remove all issues that don't meet evidence requirements
- Use precedent exceptions to avoid false positives
- Validate that evidence snippets are from the correct fields and not OCR noise

Generate FINAL REVIEW_OBJECT JSON now.
"""


def format_generator_prompt(
    case_extract: str,
    precedent_snippets: str,
    sop_checklist: str = SOP_CHECKLIST,
) -> str:
    """Format the generator prompt with case data."""
    return GENERATOR_PROMPT_TEMPLATE.format(
        sop_checklist=sop_checklist,
        precedent_snippets=precedent_snippets,
        case_extract=case_extract,
    )


def format_critic_prompt(
    case_extract: str,
    precedent_snippets: str,
    candidate_review: str,
) -> str:
    """Format the critic prompt with candidate review."""
    return CRITIC_PROMPT_TEMPLATE.format(
        precedent_snippets=precedent_snippets,
        case_extract=case_extract,
        candidate_review=candidate_review,
    )
