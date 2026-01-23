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
