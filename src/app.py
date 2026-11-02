"""Public React + TypeScript interface hosted by Streamlit Community Cloud."""
from pathlib import Path
import os
import sys

# Public queries use the bundled corpus and never enable external tracing.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import streamlit.components.v1 as components
from public_search import handle_request

st.set_page_config(page_title='Hybrid Search — Evidence Explorer', page_icon='🔎', layout='wide')
st.markdown('''<style>
.block-container {padding: 0 1rem; max-width: 1500px;}
[data-testid="stHeader"] {background: transparent;}
</style>''', unsafe_allow_html=True)
assets = Path(__file__).resolve().parents[1] / 'frontend' / 'streamlit_build'
if not (assets / 'index.html').is_file():
    st.error('Frontend build missing. Run npm --prefix frontend run build:streamlit, then restart the app.')
    st.stop()

@st.cache_resource
def public_engine():
    from public_search import create_engine
    return create_engine()

component = components.declare_component('hybrid_search_explorer', path=str(assets))
request = component(response=st.session_state.get('search_response'), default=None, key='evidence_explorer')
if isinstance(request, dict) and request.get('id') != st.session_state.get('last_search_id'):
    # The previous request is marked before execution to prevent duplicate work
    # during reruns. Component errors are returned to its own UI.
    st.session_state['last_search_id'] = request.get('id')
    st.session_state['search_response'] = handle_request(request, public_engine)
    st.rerun()
