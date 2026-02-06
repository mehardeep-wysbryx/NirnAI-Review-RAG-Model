#!/usr/bin/env python3
"""
NirnAI Streamlit App
- Upload and manage precedent JSONs
- Run reviews on test cases
- Download results
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import shutil

# Load environment variables - support both .env file (local) and Streamlit secrets (cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available in cloud

# Load API keys from Streamlit secrets if available (for cloud deployment)
if hasattr(st, 'secrets'):
    if 'OPENAI_API_KEY' in st.secrets:
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    if 'ANTHROPIC_API_KEY' in st.secrets:
        os.environ['ANTHROPIC_API_KEY'] = st.secrets['ANTHROPIC_API_KEY']

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="NirnAI - Legal Document Review",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6c5ce7, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 5.1rem;
        color: #636e72;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        color: #856404;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        background-color: #f8f9fa;
        border-radius: 5px 5px 0 0;
        color: #000000 !important;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #000000 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c5ce7;
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    .stTabs button[data-baseweb="tab"] {
        color: #000000 !important;
    }
    .stTabs button[aria-selected="true"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
PRECEDENTS_DIR = Path("data/precedents")
OUTPUTS_DIR = Path("outputs")

# Ensure directories exist
PRECEDENTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_precedent_count():
    """Count existing precedent files."""
    return len(list(PRECEDENTS_DIR.glob("*.json")))


def get_next_precedent_id():
    """Generate next precedent ID."""
    existing = list(PRECEDENTS_DIR.glob("*.json"))
    if not existing:
        return 1
    
    # Extract numbers from existing files
    numbers = []
    for f in existing:
        name = f.stem
        # Try to extract number from end of filename
        parts = name.split("_")
        for part in reversed(parts):
            if part.isdigit():
                numbers.append(int(part))
                break
    
    return max(numbers, default=0) + 1


def generate_filename(json_data, index):
    """Generate automatic filename from JSON metadata."""
    meta = json_data.get("meta", {})
    
    # Extract key identifiers
    state = meta.get("state", "unknown").lower().replace(" ", "_")[:2]  # First 2 chars
    deed_types = meta.get("deed_types", ["unknown"])
    deed = deed_types[0] if deed_types else "unknown"
    deed = deed.lower().replace(" ", "_")[:10]  # First 10 chars
    
    # Format: {state}_{deed_type}_{number}.json
    filename = f"{state}_{deed}_{index:03d}.json"
    return filename


def load_precedent_store():
    """Load the precedent store (cached)."""
    from src.ingest import PrecedentStore
    return PrecedentStore()


def run_review(merged_case, verbose=False):
    """Run the review pipeline."""
    from src.review import ReviewPipeline
    pipeline = ReviewPipeline()
    return pipeline.review(merged_case, verbose=verbose)


# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("### ⚖️ NirnAI")
    st.markdown("---")
    
    # Stats
    st.markdown("#### 📊 Database Stats")
    precedent_count = get_precedent_count()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precedents", precedent_count)
    with col2:
        output_count = len(list(OUTPUTS_DIR.glob("*.json")))
        st.metric("Reviews", output_count)
    
    st.markdown("---")
    
    # API Key Status
    st.markdown("#### 🔑 API Status")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        st.success(f"✅ OpenAI: {openai_key[:10]}...")
    elif anthropic_key:
        st.success(f"✅ Anthropic: {anthropic_key[:10]}...")
    else:
        st.error("❌ No API key found")
        st.info("Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env file")
    
    st.markdown("---")
    
    # Refresh Vector Store
    if st.button("🔄 Refresh Vector Store", use_container_width=True):
        with st.spinner("Re-ingesting precedents..."):
            try:
                store = load_precedent_store()
                store.ingest_directory(str(PRECEDENTS_DIR))
                st.success(f"✅ Ingested {precedent_count} precedents!")
            except Exception as e:
                st.error(f"Error: {e}")


# ============== MAIN CONTENT ==============
st.markdown('<p class="main-header">⚖️ NirnAI Legal Review</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">RAG-Powered Document Review System</p>', unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Precedents", "🔍 Run Review", "📁 Manage Data"])

# ============== TAB 1: UPLOAD PRECEDENTS ==============
with tab1:
    st.markdown("### 📤 Upload Precedent JSONs")
    st.markdown("Upload historical case JSONs to build the precedent database for RAG retrieval.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Select JSON files",
            type=["json"],
            accept_multiple_files=True,
            help="Upload one or more precedent JSON files"
        )
    
    with col2:
        st.markdown("#### Options")
        auto_name = st.checkbox("Auto-generate filenames", value=True, 
                               help="Automatically name files based on metadata")
        auto_ingest = st.checkbox("Auto-ingest to ChromaDB", value=True,
                                 help="Immediately add to vector store")
    
    if uploaded_files:
        st.markdown(f"#### 📁 {len(uploaded_files)} file(s) selected")
        
        # Preview uploaded files
        preview_data = []
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                content = json.load(uploaded_file)
                uploaded_file.seek(0)  # Reset file pointer
                
                meta = content.get("meta", {})
                preview_data.append({
                    "Original Name": uploaded_file.name,
                    "State": meta.get("state", "N/A"),
                    "District": meta.get("district", "N/A"),
                    "Deed Types": ", ".join(meta.get("deed_types", ["N/A"])),
                    "Has Review Notes": "✅" if content.get("review_notes") else "❌"
                })
            except json.JSONDecodeError:
                preview_data.append({
                    "Original Name": uploaded_file.name,
                    "State": "⚠️ Invalid JSON",
                    "District": "-",
                    "Deed Types": "-",
                    "Has Review Notes": "-"
                })
        
        st.dataframe(preview_data, use_container_width=True)
        
        # Upload button
        if st.button("📥 Upload & Save", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            saved_files = []
            errors = []
            next_id = get_next_precedent_id()
            
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    content = json.load(uploaded_file)
                    
                    # Generate filename
                    if auto_name:
                        filename = generate_filename(content, next_id + i)
                    else:
                        filename = uploaded_file.name
                    
                    # Save file
                    filepath = PRECEDENTS_DIR / filename
                    with open(filepath, "w") as f:
                        json.dump(content, f, indent=2)
                    
                    saved_files.append(filename)
                    
                except Exception as e:
                    errors.append(f"{uploaded_file.name}: {str(e)}")
            
            progress_bar.progress(1.0)
            status_text.empty()
            
            # Results
            if saved_files:
                st.success(f"✅ Successfully saved {len(saved_files)} file(s)!")
                with st.expander("View saved files"):
                    for f in saved_files:
                        st.code(f)
                
                # Auto-ingest
                if auto_ingest:
                    with st.spinner("Ingesting to ChromaDB..."):
                        try:
                            store = load_precedent_store()
                            store.ingest_directory(str(PRECEDENTS_DIR))
                            st.success("✅ Vector store updated!")
                        except Exception as e:
                            st.error(f"Ingestion error: {e}")
            
            if errors:
                st.error(f"❌ {len(errors)} file(s) failed:")
                for err in errors:
                    st.text(err)


# ============== TAB 2: RUN REVIEW ==============
with tab2:
    st.markdown("### 🔍 Run Document Review")
    st.markdown("Upload a merged case JSON to run the two-stage LLM review.")
    
    # Check API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        st.error("⚠️ No API key configured. Please add OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            test_file = st.file_uploader(
                "Upload merged case JSON",
                type=["json"],
                help="Upload a merged_case.json file for review"
            )
        
        with col2:
            st.markdown("#### Settings")
            verbose = st.checkbox("Verbose output", value=True)
            save_output = st.checkbox("Save to outputs/", value=True)
        
        if test_file:
            try:
                merged_case = json.load(test_file)
                
                # Preview
                with st.expander("📋 Preview uploaded case", expanded=False):
                    st.json(merged_case)
                
                # Run review
                if st.button("🚀 Run Review", type="primary", use_container_width=True):
                    
                    # Progress tracking
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    try:
                        status_text.text("🔄 Initializing pipeline...")
                        progress_bar.progress(10)
                        
                        status_text.text("📚 Loading precedent store...")
                        progress_bar.progress(20)
                        
                        status_text.text("🔍 Extracting case fingerprint...")
                        progress_bar.progress(30)
                        
                        status_text.text("🧠 Retrieving similar precedents...")
                        progress_bar.progress(40)
                        
                        status_text.text("🤖 Running Generator LLM (Stage 1)...")
                        progress_bar.progress(50)
                        
                        # Actually run the review
                        result = run_review(merged_case, verbose=verbose)
                        
                        status_text.text("✅ Running Critic LLM (Stage 2)...")
                        progress_bar.progress(80)
                        
                        status_text.text("📝 Finalizing review...")
                        progress_bar.progress(100)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📋 Review Results")
                        
                        # Extract issues from sections (flatten nested structure)
                        sections = result.get("sections", {})
                        all_issues = []
                        for section_name, section_issues in sections.items():
                            if isinstance(section_issues, list):
                                for issue in section_issues:
                                    issue["section"] = section_name  # Add section info
                                    all_issues.append(issue)
                        
                        # Map severity: critical->HIGH, major->MEDIUM, minor->LOW
                        severity_map = {"critical": "HIGH", "major": "MEDIUM", "minor": "LOW"}
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Issues", len(all_issues))
                        with col2:
                            high = len([i for i in all_issues if severity_map.get(i.get("severity", "").lower(), "") == "HIGH"])
                            st.metric("🔴 High Severity", high)
                        with col3:
                            medium = len([i for i in all_issues if severity_map.get(i.get("severity", "").lower(), "") == "MEDIUM"])
                            st.metric("🟡 Medium Severity", medium)
                        with col4:
                            low = len([i for i in all_issues if severity_map.get(i.get("severity", "").lower(), "") == "LOW"])
                            st.metric("🟢 Low Severity", low)
                        
                        # Recommendation based on risk level
                        risk_level = result.get("overall_risk_level", "N/A")
                        rec_map = {
                            "CLEAR_TO_RELEASE": ("APPROVE", "green"),
                            "NEEDS_FIX_BEFORE_RELEASE": ("HOLD", "orange"),
                            "REJECT": ("REJECT", "red"),
                        }
                        rec, rec_color = rec_map.get(risk_level, (risk_level, "gray"))
                        
                        st.markdown(f"**Recommendation:** :{rec_color}[**{rec}**]")
                        
                        # Summary
                        summary = result.get("overall_summary") or result.get("summary")
                        if summary:
                            st.markdown("**Summary:**")
                            st.info(summary)
                        
                        # Issues list grouped by section
                        if all_issues:
                            st.markdown("#### 📝 Issues Identified")
                            
                            # Group by section for better organization
                            section_names = {
                                "property_details": "Property Details",
                                "schedule_of_property": "Schedule of Property",
                                "documents_scrutinized": "Documents Scrutinized",
                                "encumbrance_certificate": "Encumbrance Certificate",
                                "flow_of_title": "Flow of Title",
                                "mutation_and_tax": "Mutation & Tax",
                                "conclusion_and_remarks": "Conclusion & Remarks",
                                "layout_and_flowchart": "Layout & Flowchart",
                            }
                            
                            for issue in all_issues:
                                severity = issue.get("severity", "unknown").lower()
                                severity_display = severity_map.get(severity, severity.upper())
                                severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity_display, "⚪")
                                
                                section = issue.get("section", "unknown")
                                section_display = section_names.get(section, section.replace("_", " ").title())
                                
                                location = issue.get("location", "N/A")
                                message = issue.get("message_for_maker", "No description")[:60]
                                
                                with st.expander(f"{severity_emoji} [{issue.get('id', 'N/A')}] {section_display} - {message}..."):
                                    st.markdown(f"**ID:** {issue.get('id', 'N/A')}")
                                    st.markdown(f"**Section:** {section_display}")
                                    st.markdown(f"**Location:** {location}")
                                    st.markdown(f"**Severity:** {severity_emoji} {severity_display}")
                                    st.markdown(f"**Rule:** {issue.get('rule', 'N/A')}")
                                    st.markdown(f"**Message:** {issue.get('message_for_maker', 'N/A')}")
                                    st.markdown(f"**Suggested Fix:** {issue.get('suggested_fix', 'N/A')}")
                                    
                                    if issue.get("evidence"):
                                        st.markdown("**Evidence:**")
                                        evidence = issue.get("evidence")
                                        if isinstance(evidence, dict):
                                            st.markdown(f"- **From Report:** {evidence.get('from_report', 'N/A')}")
                                            st.markdown(f"- **From Source Docs:** {evidence.get('from_source_docs', 'N/A')}")
                                        else:
                                            st.json(evidence)
                        
                        # Full JSON
                        with st.expander("📄 View Full JSON Output"):
                            st.json(result)
                        
                        # Save output
                        if save_output:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename = f"review_{timestamp}.json"
                            output_path = OUTPUTS_DIR / output_filename
                            
                            with open(output_path, "w") as f:
                                json.dump(result, f, indent=2)
                            
                            st.success(f"✅ Saved to: {output_path}")
                        
                        # Download button
                        st.markdown("---")
                        st.download_button(
                            label="📥 Download Review JSON",
                            data=json.dumps(result, indent=2),
                            file_name=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Review failed: {str(e)}")
                        st.exception(e)
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid JSON.")


# ============== TAB 3: MANAGE DATA ==============
with tab3:
    st.markdown("### 📁 Manage Precedent Data")
    
    # List existing precedents
    precedent_files = sorted(PRECEDENTS_DIR.glob("*.json"))
    
    if not precedent_files:
        st.info("No precedent files found. Upload some in the 'Upload Precedents' tab.")
    else:
        st.markdown(f"#### 📚 {len(precedent_files)} Precedent Files")
        
        # Search/filter
        search = st.text_input("🔍 Search files", placeholder="Type to filter...")
        
        # Display files
        filtered_files = [f for f in precedent_files if search.lower() in f.name.lower()] if search else precedent_files
        
        # Create dataframe for display
        file_data = []
        for f in filtered_files:
            try:
                with open(f) as file:
                    content = json.load(file)
                    meta = content.get("meta", {})
                    file_data.append({
                        "Filename": f.name,
                        "State": meta.get("state", "N/A"),
                        "District": meta.get("district", "N/A"),
                        "Deed Types": ", ".join(meta.get("deed_types", [])),
                        "Size": f"{f.stat().st_size / 1024:.1f} KB"
                    })
            except:
                file_data.append({
                    "Filename": f.name,
                    "State": "Error",
                    "District": "-",
                    "Deed Types": "-",
                    "Size": f"{f.stat().st_size / 1024:.1f} KB"
                })
        
        st.dataframe(file_data, use_container_width=True)
        
        # Actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # View file
            selected_file = st.selectbox("Select file to view", [f.name for f in filtered_files])
            if selected_file and st.button("👁️ View Contents"):
                filepath = PRECEDENTS_DIR / selected_file
                with open(filepath) as f:
                    content = json.load(f)
                st.json(content)
        
        with col2:
            # Delete file
            delete_file = st.selectbox("Select file to delete", [f.name for f in filtered_files], key="delete_select")
            if delete_file and st.button("🗑️ Delete File", type="secondary"):
                filepath = PRECEDENTS_DIR / delete_file
                if st.checkbox("Confirm deletion", key="confirm_delete"):
                    os.remove(filepath)
                    st.success(f"Deleted {delete_file}")
                    st.rerun()
        
        with col3:
            # Download all
            if st.button("📦 Download All as ZIP"):
                import io
                import zipfile
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in precedent_files:
                        zf.write(f, f.name)
                
                st.download_button(
                    label="📥 Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="precedents.zip",
                    mime="application/zip"
                )
        
        # Bulk operations
        st.markdown("---")
        st.markdown("#### ⚡ Bulk Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Re-ingest All to ChromaDB", use_container_width=True):
                with st.spinner("Re-ingesting all precedents..."):
                    try:
                        store = load_precedent_store()
                        store.ingest_directory(str(PRECEDENTS_DIR))
                        st.success(f"✅ Successfully ingested {len(precedent_files)} precedents!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("🗑️ Clear All Precedents", type="secondary", use_container_width=True):
                if st.checkbox("⚠️ I understand this will delete ALL precedent files", key="confirm_clear_all"):
                    for f in precedent_files:
                        os.remove(f)
                    st.success("All precedent files deleted!")
                    st.rerun()


# ============== FOOTER ==============
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #636e72; font-size: 0.9rem;">
        <p>NirnAI - RAG-Powered Legal Document Review</p>
    </div>
    """,
    unsafe_allow_html=True
)
